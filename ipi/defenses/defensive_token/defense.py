"""
DefensiveToken defense wrapper for `ipi` benchmark evaluation.

Port of the inference path in ``code/defense/DefensiveToken-main/demo.py``.

DefensiveToken is a *test-time* defense: five special tokens with optimised
embeddings are prepended to the prompt. Everything else — weights, tokenizer
vocabulary beyond those five entries — is the public model. That gives it the
same instruction/data channel structure as StruQ and SecAlign, so it reuses
:class:`~ipi.defenses.channels.StructuredChannelDefense`: read the split off the
prompt, filter the data channel, hand the model one pre-rendered string.

What differs from StruQ/SecAlign is the rendering. Those models are fine-tuned
on bare Alpaca-format strings; DefensiveToken's models keep a chat template
(the Meta-SecAlign-style one installed by ``apply_defensive_tokens``), with the
trusted instruction in the ``system`` turn and untrusted data in the ``user``
turn.

IPI adaptations vs original
---------------------------
* Upstream's demo hands the defense a hand-written ``(instruction, data)`` pair.
  Here the same pair travels with the prompt as a ``ChanneledPrompt``
  (``ipi/channels.py``), or is pinned with ``set_channels`` when the caller
  builds messages by hand. Same split StruQ/SecAlign read, so a
  defense-vs-defense row compares defenses and not prompt-parsing luck.
* Upstream filters ``tokenizer.all_special_tokens`` out of the concatenated
  untrusted string. Same set here, but applied to the recovered data channel
  and via ``recursive_filter`` (repeat-until-fixpoint) rather than a single
  pass — deleting one token can splice its neighbours into a fresh occurrence.
  ``all_special_tokens`` is read live, so it includes the five DefensiveTokens
  themselves: an attacker cannot smuggle them into the data channel.
* ``enabled=False`` renders the *same* prompt without the five tokens. That is
  the paper's "skip DefensiveTokens" mode and the correct undefended control —
  it isolates the tokens, not the prompt format.
* Prompts carrying no data channel are emitted as a system turn only,
  rather than routing the whole prompt into the untrusted ``user`` turn.

Preserved upstream quirk
------------------------
The rendered prompt goes to the model as a ``RAW_ROLE`` turn, which
``LocalLLM._build_local_prompt_ids`` encodes with ``add_special_tokens=True``.
For the two Llama-3 templates — the only ones that emit ``bos_token`` — that
yields *two* BOS tokens, because the tokenizer's post-processor prepends one to
a string that already contains it. This is not an oversight: upstream's
``demo.py`` calls ``tokenizer(input_string)``, whose default is likewise
``add_special_tokens=True``, so the released embeddings were exercised against
exactly this token sequence. Do not "fix" it — that would put the optimised
tokens in a context they were never evaluated in.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence, Tuple

from ...victim import Victim
from ..channels import StructuredChannelDefense
from .config import DEFENSIVE_TOKEN_NAMES

log = logging.getLogger(__name__)


class DefensiveTokenDefense(StructuredChannelDefense):
    """
    DefensiveToken defense wrapper (Chen et al., AISec 2025).

    Requires a **local** target whose model and tokenizer have been patched by
    :func:`~ipi.defenses.defensive_token.build.apply_defensive_tokens` (or that
    was loaded from a ``-5DefensiveTokens`` checkpoint). The constructor
    verifies this rather than trusting it: an unpatched tokenizer renders
    ``add_defensive_tokens`` as an undefined Jinja variable, which is falsy, so
    the defense would run with no defense in it and report a suspiciously good
    ASR.

    Args:
        target:                 Underlying Victim carrying the patched model.
        enabled:                Emit the five tokens. ``False`` gives the
                                paper's stock-utility mode / undefended control.
        apply_defensive_filter: Strip special tokens from the data channel.
                                ``False`` isolates the filter's contribution.

    Example::

        llm = LocalLLM(model="meta-llama/Llama-3.1-8B-Instruct", device_map={"": 0})
        apply_defensive_tokens(llm.hf_model, llm.tokenizer,
                               model_name="meta-llama/Llama-3.1-8B-Instruct")
        defended = DefensiveTokenDefense(TargetLLM(llm))
    """

    def __init__(
        self,
        target: Victim,
        enabled: bool = True,
        apply_defensive_filter: bool = True,
    ):
        super().__init__(target, apply_defensive_filter=apply_defensive_filter)
        if self.backend != "local":
            raise ValueError(
                "DefensiveTokenDefense needs a local target: the defense is five "
                "embedding rows in the model's vocabulary, which an API backend "
                f"cannot expose (got backend={self.backend!r})."
            )
        self.enabled = enabled
        self._verify_target()

    # -- fidelity checks -----------------------------------------------------

    def _verify_target(self) -> None:
        """Fail loudly if the target was never patched."""
        tokenizer = self.tokenizer

        vocab = tokenizer.get_vocab()
        missing = [t for t in DEFENSIVE_TOKEN_NAMES if t not in vocab]
        if missing:
            raise ValueError(
                f"Target tokenizer is missing {missing} — it was never patched. "
                "Call ipi.defenses.defensive_token.apply_defensive_tokens(model, "
                "tokenizer, model_name=...) first, or load a "
                "'-5DefensiveTokens' checkpoint."
            )

        probe = [
            {"role": "system", "content": "instruction"},
            {"role": "user", "content": "data"},
        ]
        try:
            on = self._render(probe, add_defensive_tokens=True)
            off = self._render(probe, add_defensive_tokens=False)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim below
            raise ValueError(
                "Target tokenizer could not render the DefensiveToken chat "
                f"template ({exc}). Re-run apply_defensive_tokens with "
                "set_chat_template=True."
            ) from exc

        if DEFENSIVE_TOKEN_NAMES[0] not in on or on == off:
            raise ValueError(
                "Target tokenizer's chat template ignores add_defensive_tokens, "
                "so the tokens would never reach the model. This is the model's "
                "stock template, not the DefensiveToken one — re-run "
                "apply_defensive_tokens with set_chat_template=True."
            )

    # -- StructuredChannelDefense hooks --------------------------------------

    @property
    def filtered_tokens(self) -> Tuple[str, ...]:  # type: ignore[override]
        """
        Read live off the tokenizer, so the five DefensiveTokens are included
        once the target is patched (demo.py: ``recursive_filter(..., all_special_tokens)``).
        """
        return tuple(t for t in self.tokenizer.all_special_tokens if t)

    def _render(self, conversation: Sequence[Mapping[str, Any]], add_defensive_tokens: bool) -> str:
        return self.tokenizer.apply_chat_template(
            list(conversation),
            tokenize=False,
            add_generation_prompt=True,
            add_defensive_tokens=add_defensive_tokens,
        )

    def format_prompt(self, instruction: str, data: str) -> str:
        conversation = [{"role": "system", "content": instruction}]
        if data and data.strip():
            conversation.append({"role": "user", "content": data})
        return self._render(conversation, add_defensive_tokens=self.enabled)

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        return f"DefensiveTokenDefense(tokens={state}, target={self.target!r})"
