"""
Victim — abstract interface for any victim LLM or defended pipeline.

Any system to be evaluated against IPI attacks must subclass Victim and
implement at minimum generate(). Additional methods are optional and raise
informative errors if called on a victim that does not support them.

Required for ALL attacks:
  generate(messages)           — called by TAP / PAIR / RS / Beam-RS / BEAST
                                 for response generation.

Required for RS / Beam-RS:
  get_first_token_logprobs()   — logprob-guided search; raises
                                 LogprobNotSupportedError by default.

Required for BEAST (white-box local access):
  tokenizer, hf_model, _device — HuggingFace model internals.
  generate_n_tokens_batch()    — batch forward pass for beam search.
  attack_objective_targeted()  — perplexity objective for BEAST scoring.
  apply_chat_template()        — chat template formatting.
  All raise LocalOnlyError by default.

Instruction / data channels
---------------------------
  set_channels(instruction, data)  — pin the trusted/untrusted split for prompts
                                     built by hand. Prompts built by ``ipi.harness``
                                     carry their own split (``ipi/channels.py``),
                                     and that one wins. No defense parses a prompt
                                     to find the untrusted span.
  resolve_channels(messages)       — what a defense calls to get it.

Attributes
----------
  system_prompt  (str)  — prepended as a system turn in every messages list.
                          Set as an instance attr in subclass __init__. Default "".
  model_name     (str)  — model identifier. Used by RS for adv_init selection. Default "".
  backend        (str)  — "api" | "local". Override as class variable in subclass.
                          RS uses this for probability-threshold selection.
  max_bs         (int)  — batch size for local operations. Default 50.

Example — custom attention-tracker defense
------------------------------------------
    from ipi.victim import Victim

    class AttentionTrackerVictim(Victim):
        backend = "local"

        def __init__(self, model_path: str, system_prompt: str = ""):
            self._model = AttentionTrackerModel(model_path)
            self.system_prompt = system_prompt
            self.model_name    = model_path

        def generate(self, messages, max_tokens=200, temperature=0.0):
            return self._model.run_with_defense(messages)

        def get_first_token_logprobs(self, messages, n_top=20):
            return self._model.get_logprobs(messages, n=n_top)

        # Expose HF internals for BEAST support:
        @property
        def tokenizer(self): return self._model.tokenizer
        @property
        def hf_model(self):  return self._model.hf_model
        @property
        def _device(self):   return self._model.device
        def generate_n_tokens_batch(self, *a, **kw): return self._model.generate_batch(*a, **kw)
        def attack_objective_targeted(self, *a, **kw): return self._model.objective(*a, **kw)
        def apply_chat_template(self, *a, **kw): return self._model.tokenizer.apply_chat_template(*a, **kw)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Optional, Sequence

from .channels import (
    ChanneledPrompt, PreRenderedMessages, PreRenderedPromptError, channels_of,
)
from .llm_unified import LogprobNotSupportedError, LocalOnlyError

log = logging.getLogger(__name__)


class Victim(ABC):
    """
    Abstract interface for any victim / defended LLM pipeline.

    Subclass this to plug a custom defense into the IPI attack benchmark.
    Implement generate() to support all attacks.
    Implement get_first_token_logprobs() to support RS / Beam-RS.
    Implement the BEAST properties / methods to support BEAST.
    """

    # Class variables — override as class variables or instance attrs in subclasses.
    backend: ClassVar[str] = "api"   # "api" | "local"
    system_prompt: str = ""
    model_name: str = ""
    max_bs: int = 50

    # ------------------------------------------------------------------
    # Abstract — required for all attacks
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response from an OpenAI-format messages list.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            max_tokens:  Override default generation length. None = use default.
            temperature: Override default sampling temperature. None = use default.

        Returns:
            Generated response text.
        """

    # ------------------------------------------------------------------
    # Optional — needed for RS / Beam-RS logprob-guided search
    # ------------------------------------------------------------------

    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        """
        Return log-probabilities for the top-n most likely first tokens.

        Implement this to support RS and Beam-RS attacks.

        Returns:
            dict mapping token string → log-probability (negative float).

        Raises:
            LogprobNotSupportedError if not implemented.
        """
        raise LogprobNotSupportedError(
            f"{type(self).__name__} does not support get_first_token_logprobs(). "
            "Implement this method to use RS / Beam-RS attacks, or use an "
            "APILLM provider that supports logprobs (openai, deepseek, metis_openai)."
        )

    # ------------------------------------------------------------------
    # Optional — needed for BEAST (white-box local model access)
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """HuggingFace tokenizer. Required for BEAST and token-level RS."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not expose a tokenizer. "
            "BEAST and token-level RS require a local-model victim. "
            "Use TargetLLM(LocalLLM(...)) or set backend='local' in your Victim subclass."
        )

    @property
    def hf_model(self):
        """Raw HuggingFace model. Required for BEAST beam-search logit computation."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not expose hf_model. "
            "BEAST requires direct model access (backend='local')."
        )

    @property
    def _device(self):
        """Torch device of the local model. Required for BEAST."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not expose _device."
        )

    def generate_n_tokens_batch(self, *args, **kwargs):
        """Batch forward pass returning logits. Required for BEAST beam search."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not implement generate_n_tokens_batch(). "
            "BEAST requires this method for white-box logit access."
        )

    def attack_objective_targeted(self, *args, **kwargs):
        """Compute perplexity-based attack objective over target_str. Required for BEAST."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not implement attack_objective_targeted()."
        )

    def apply_chat_template(self, *args, **kwargs):
        """Apply the model's HuggingFace chat template. Required for BEAST."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not implement apply_chat_template()."
        )

    # ------------------------------------------------------------------
    # Instruction / data channels
    # ------------------------------------------------------------------
    #
    # A defense that reformats, filters, wraps, marks or sanitizes the untrusted
    # span needs to know where that span is. It is never recovered from the
    # rendered text: ``harness`` builds a ``ChanneledPrompt`` and renders it to
    # ``ChanneledMessages``, which carry the split with them all the way into
    # ``preprocess_messages``. See ``ipi/channels.py``.

    _pinned_channels: Optional[ChanneledPrompt] = None
    _channel_pin_shadowed: bool = False
    _channel_guess_warned: bool = False

    def set_channels(self, instruction: str, data: str, system: str = "") -> None:
        """
        Pin the split for prompts that carry none — hand-built messages, or an
        empty list plus ``preprocess_messages([])`` (what the DefensiveToken
        notebook does to render a prompt without generating).

        A prompt that carries its own split wins over this pin: the carried one
        is per-prompt, this one is sticky and would otherwise apply the first
        scenario's data to all 360.
        """
        self.set_channeled_prompt(
            ChanneledPrompt(system=system, instruction=instruction, data=data))

    def set_channeled_prompt(self, prompt: Optional[ChanneledPrompt]) -> None:
        """``set_channels`` for a fully-specified prompt (framing, epilogue, ...)."""
        self._pinned_channels = prompt
        self._channel_pin_shadowed = False

    def clear_channels(self) -> None:
        self._pinned_channels = None
        self._channel_pin_shadowed = False

    def resolve_channels(
        self,
        messages: Sequence[Mapping[str, Any]],
        warn: bool = True,
    ) -> ChanneledPrompt:
        """
        The instruction/data split for ``messages``. Never returns None: the
        last resort is ``ChanneledPrompt.from_messages``, which treats the whole
        final user turn as data rather than leaving a defense inert.

        Both warnings here fire **once per victim**. A search attack calls this a
        few thousand times per scenario, and a per-call warning would be a log
        flood that everyone learns to ignore — which is the same as no warning.

        Raises:
            PreRenderedPromptError: if ``messages`` is already a structured defense's
                rendered prompt. Guessing a split for it puts this defense's text
                after the response delimiter; see ``PreRenderedMessages``.
        """
        if isinstance(messages, PreRenderedMessages):
            raise PreRenderedPromptError(
                f"{type(self).__name__} is chained under "
                f"{messages.rendered_by or 'a structured defense'}, whose output is "
                "already rendered for the model — there is no instruction/data split "
                "left to read. Put the structured defense innermost, e.g. "
                f"{type(self).__name__}({messages.rendered_by or 'StruQDefense'}(target)) "
                f"rather than {messages.rendered_by or 'StruQDefense'}({type(self).__name__}(target))."
            )
        carried = channels_of(messages)
        if carried is not None:
            if self._pinned_channels is not None and not self._channel_pin_shadowed:
                self._channel_pin_shadowed = True
                log.warning(
                    "[channels] %s has channels pinned by set_channels(), but this "
                    "prompt carries its own split — using the carried one. Call "
                    "clear_channels() if the pin is stale.", type(self).__name__)
            return carried
        if self._pinned_channels is not None:
            return self._pinned_channels
        first = warn and not self._channel_guess_warned
        self._channel_guess_warned = True
        return ChanneledPrompt.from_messages(messages, warn=first)

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def __call__(self, messages_or_injection) -> str:
        """
        Dual-mode callable.

        str input  → wraps as a user message (with system_prompt if set).
        list input → forwarded to generate() as-is.
        """
        if isinstance(messages_or_injection, str):
            msgs: list[dict] = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": messages_or_injection})
        elif isinstance(messages_or_injection, list):
            msgs = messages_or_injection
        else:
            raise TypeError(
                f"Expected str or list[dict], got {type(messages_or_injection).__name__}"
            )
        return self.generate(msgs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"model={self.model_name!r}, backend={self.backend!r})"
        )
