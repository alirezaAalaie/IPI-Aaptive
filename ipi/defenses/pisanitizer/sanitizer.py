"""
PISanitizer engine — the sanitization loop itself.

Port of ``code/defense/PISanitizer-main/PISanitizer/pisanitizer.py`` (with the
configurability of ``methods/pisanitizer.py``).

IPI adaptations vs original
---------------------------
* Upstream is a module-level function holding the model in two globals, loaded
  on first call and never released. This is a class, so a run can share weights
  with the victim model (``PISanitizer.from_local_llm``) instead of holding a
  second 16 GB copy — which is the difference between fitting on one Kaggle GPU
  and not.
* Upstream's batch path mutates its shared config dict when it fills in
  ``smooth_win``, so sample 1's smoothing window silently applies to samples
  2..N. Config is resolved per call here.
* Upstream generates 32 (batch path) or 1 (quick-usage path) tokens before
  reading attention, but only ever uses the *first* generated token's query
  row. One token is generated here — the two paths' attention signals are
  identical, and the 32-token version costs 31 wasted decode steps per round.
* The forward pass that extracts hidden states runs on the sequence *including*
  the generated token, exactly as upstream: the attribution row is that token's
  query position.
* Sanitization is reported through :class:`SanitizationTrace` rather than
  printed, so an evaluation can record what was cut without scraping stdout.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from .attention import hidden_states_for, infer_model_type, token_attention_signal
from .config import (
    ANCHOR_SUFFIX,
    CONTEXT_PREFIX,
    DEFAULT_SANITIZER_MODEL,
    LLAMA3_DELIMITERS,
    MAX_ITERATIONS,
    PADDING_TOKEN,
    SANITIZATION_INSTRUCTIONS,
    resolve_config,
    smoothing_window,
)
from .peaks import group_peaks

log = logging.getLogger(__name__)


@dataclass
class RemovedSpan:
    """One deleted token span, in the coordinates of that round's context."""
    iteration: int
    start: int
    end: int
    text: str

    def __repr__(self) -> str:
        preview = self.text if len(self.text) <= 60 else self.text[:57] + "..."
        return f"RemovedSpan(iter={self.iteration}, [{self.start}:{self.end}], {preview!r})"


@dataclass
class SanitizationTrace:
    """
    What one :meth:`PISanitizer.sanitize` call did.

    ``attn_signals`` holds the smoothed per-token signal for each round, which
    is the quantity this defense and every other attention-based defense are
    ultimately arguing about — keep it when comparing signals across methods.
    """
    original: str
    sanitized: str
    removed: List[RemovedSpan] = field(default_factory=list)
    iterations: int = 0
    n_removed_tokens: int = 0
    attn_signals: List[List[float]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.n_removed_tokens > 0

    def summary(self) -> str:
        return (
            f"PISanitizer: {self.iterations} round(s), "
            f"{self.n_removed_tokens} token(s) removed in "
            f"{len(self.removed)} span(s)"
        )


class PISanitizer:
    """
    Attention-guided prompt sanitizer (Geng et al., arXiv:2511.10720).

    Wraps a local model whose attention can be reconstructed and exposes
    :meth:`sanitize`, which deletes the token spans that most drive
    instruction-following behaviour out of an untrusted context.

    The sanitizer model is independent of the model being defended: the backend
    LLM can be an API model. Upstream always uses one Llama-3.1-8B-Instruct for
    both.

    Args:
        model:        A loaded causal LM. Omit to load ``model_name``.
        tokenizer:    Its tokenizer. Omit to load ``model_name``.
        model_name:   Model to load when ``model``/``tokenizer`` are omitted.
        config:       Overrides for :data:`~.config.DEFAULT_CONFIG`.
        delimiters:   Chat markers used to build the detection prompt. The
                      default is the raw Llama-3 set upstream hard-codes;
                      override for a non-Llama sanitizer.
        model_type:   Attention-family override (``llama``, ``qwen2``, ...).
        device_map:   Passed to ``from_pretrained`` when loading.
        torch_dtype:  Passed to ``from_pretrained`` when loading.

    Example::

        san = PISanitizer()                      # loads Llama-3.1-8B-Instruct
        clean = san.sanitize(untrusted_context)

        san = PISanitizer.from_local_llm(llm)    # reuse the victim's weights
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = DEFAULT_SANITIZER_MODEL,
        config: Optional[dict] = None,
        delimiters: Sequence[str] = LLAMA3_DELIMITERS,
        model_type: Optional[str] = None,
        device_map: Any = "auto",
        torch_dtype: Any = None,
    ):
        if (model is None) != (tokenizer is None):
            raise ValueError("Pass both model and tokenizer, or neither.")

        if model is None:
            model, tokenizer = self._load(model_name, device_map, torch_dtype)

        self.model = model
        self.tokenizer = tokenizer
        self.config = resolve_config(config)
        if len(delimiters) != 3:
            raise ValueError("delimiters must be (system, user, assistant)")
        self.delimiters = tuple(delimiters)
        self.model_type = model_type or infer_model_type(model)

        if self.model_type != "llama" and tuple(delimiters) == LLAMA3_DELIMITERS:
            log.warning(
                "[PISanitizer] Sanitizer family is %r but the detection prompt "
                "is being built with Llama-3 chat markers. Pass delimiters= for "
                "this model, or the anchor prompt will not parse as a "
                "conversation and the attention signal will be meaningless.",
                self.model_type,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    @staticmethod
    def _load(model_name: str, device_map: Any, torch_dtype: Any):
        import transformers

        log.info("[PISanitizer] Loading sanitizer model %s", model_name)
        kwargs: dict = {"device_map": device_map}
        kwargs["torch_dtype"] = torch_dtype if torch_dtype is not None else "auto"
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True,
        )
        return model, tokenizer

    @classmethod
    def from_local_llm(cls, llm: Any, **kwargs) -> "PISanitizer":
        """
        Build a sanitizer over an already-loaded ``LocalLLM``'s weights.

        Sanitizing with the same model being defended is upstream's setup and
        halves the memory a run needs. Note the tokenizer is shared, so this
        is only safe when the victim has not been given extra vocabulary
        (a StruQ/SecAlign/DefensiveToken checkpoint has).
        """
        return cls(model=llm.hf_model, tokenizer=llm.tokenizer, **kwargs)

    # -- detection prompt ----------------------------------------------------

    def _anchor_prompts(self, target_instruction: str = "") -> Tuple[str, str]:
        anchor = SANITIZATION_INSTRUCTIONS[int(self.config["anchor_prompt"])]
        return anchor.format(target_inst=target_instruction), ANCHOR_SUFFIX

    def build_detection_prompt(self, target_instruction: str = "") -> Tuple[str, str]:
        """
        Return the ``(prefix, suffix)`` the context gets wrapped in.

        The ``" X" * 500`` padding on both sides is load-bearing: it holds the
        context away from the prompt boundaries where attention sinks live.
        """
        anchor, suffix = self._anchor_prompts(target_instruction)
        sys_d, user_d, asst_d = self.delimiters
        prefix = (
            sys_d + anchor + user_d + CONTEXT_PREFIX
            + PADDING_TOKEN * int(self.config["start_offset"])
        )
        tail = PADDING_TOKEN * int(self.config["end_offset"]) + suffix + asst_d
        return prefix, tail

    # -- the loop ------------------------------------------------------------

    def sanitize(self, context: str, target_instruction: str = "") -> str:
        """Return ``context`` with its highest-attention spans deleted."""
        return self.sanitize_with_trace(context, target_instruction).sanitized

    def sanitize_with_trace(
        self,
        context: str,
        target_instruction: str = "",
    ) -> SanitizationTrace:
        """
        Run the full sanitization loop and report what was removed.

        Args:
            context:            The untrusted text to sanitize.
            target_instruction: Only used by ``anchor_prompt=3``, which anchors
                                on the real user task instead of a generic
                                "obey anything" instruction.
        """
        import torch

        trace = SanitizationTrace(original=context, sanitized=context)
        if not context or not context.strip():
            return trace

        tok = self.tokenizer
        prefix, tail = self.build_detection_prompt(target_instruction)
        prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        tail_ids = tok(tail, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        context_ids = tok(context, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        # Fixed for the whole call, from the initial length — matching upstream,
        # where it is computed once and reused across rounds.
        smooth_win = smoothing_window(len(context_ids), self.config["smooth_win"])

        for iteration in range(1, MAX_ITERATIONS + 1):
            trace.iterations = iteration
            if len(context_ids) == 0:
                break

            input_ids = torch.cat([prefix_ids, context_ids, tail_ids]).unsqueeze(0)
            input_ids = input_ids.to(self.model.device)
            ctx_start = len(prefix_ids)
            ctx_end = ctx_start + len(context_ids)
            n_input = input_ids.shape[1]

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
                hidden_states = hidden_states_for(self.model, generated)

            # Attribution row = the query position of the newly generated token.
            signal = token_attention_signal(
                self.model, hidden_states,
                attribution_start=n_input + 1,
                attribution_end=n_input + 2,
                n_context_tokens=n_input,
                mode=str(self.config["mode"]),
                model_type=self.model_type,
            )

            smoothed, spans = group_peaks(
                signal[ctx_start:ctx_end].tolist(),
                smooth_win=smooth_win,
                max_gap=int(self.config["max_gap"]),
                threshold=float(self.config["threshold"]),
            )
            trace.attn_signals.append(list(smoothed))

            remove_idx: List[int] = []
            for start, end in spans:
                start = max(0, start)
                end = min(len(context_ids) - 1, end)
                if end < start:
                    continue
                span_text = tok.decode(context_ids[start:end + 1], skip_special_tokens=True)
                trace.removed.append(RemovedSpan(iteration, start, end, span_text))
                remove_idx.extend(range(start, end + 1))
                log.info("[PISanitizer] round %d removing [%d:%d] %r",
                         iteration, start, end, span_text[:80])

            if not remove_idx:
                break

            trace.n_removed_tokens += len(remove_idx)
            keep = torch.ones(len(context_ids), dtype=torch.bool)
            keep[remove_idx] = False
            context_ids = copy.deepcopy(context_ids)[keep]
            # Rounds carry the *ids* forward, not a re-tokenization of the
            # decoded text (upstream does the same). Re-encoding would let the
            # tokens on either side of a cut merge, shifting every later index.
            trace.sanitized = tok.decode(context_ids, skip_special_tokens=True)

        log.info("[PISanitizer] %s", trace.summary())
        return trace

    def __repr__(self) -> str:
        return (
            f"PISanitizer(model={getattr(self.model, 'name_or_path', '?')!r}, "
            f"mode={self.config['mode']!r})"
        )
