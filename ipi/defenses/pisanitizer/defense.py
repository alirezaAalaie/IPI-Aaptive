"""
PISanitizer defense wrapper for `ipi` benchmark evaluation.

Unlike StruQ / SecAlign / DefensiveToken, PISanitizer does not restructure the
prompt for a specially-trained model — it *edits the untrusted text in place*
and hands the backend model its ordinary prompt. So this subclasses
``DefendedVictim`` directly rather than ``StructuredChannelDefense``: the
recovered data channel is sanitized and substituted back into the original
messages, leaving message roles and structure untouched.

That also means the defended model can be an **API** model. Only the sanitizer
needs local weights and attention access.

IPI adaptations vs original
---------------------------
* Upstream is evaluated by handing the sanitizer a hand-split
  ``(input_prompt, context)`` pair from its own dataset builders. Here the pair
  is the ``ChanneledPrompt`` the harness built (``ipi/channels.py``), carried on
  the messages themselves — the same split StruQ/SecAlign/DefensiveToken read, so
  defense-vs-defense rows differ by defense and not by prompt parsing.
  ``set_channels`` pins it for a prompt built by hand.
* Upstream sanitizes the LongBench-style context, which is thousands of tokens.
  Short IPI contexts are still handled, but note the peak-finding floor
  (``height=0.005``, ``threshold=0.01``) was tuned against long contexts where
  attention per token is small; on a 40-token context, ordinary tokens clear
  those thresholds easily. See ``min_context_chars``.
* The sanitizer only ever sees the *data* channel, never the trusted
  instruction, so it cannot delete the user's real task. Upstream passes the
  context alone for the same reason.
"""
from __future__ import annotations

import logging
from typing import Optional

from ...victim import Victim
from ..base import DefendedVictim
from .sanitizer import PISanitizer, SanitizationTrace

log = logging.getLogger(__name__)


class PISanitizerDefense(DefendedVictim):
    """
    PISanitizer defense (Geng et al., arXiv:2511.10720).

    Sanitizes the untrusted data channel before the victim model sees it.

    Args:
        target:             The victim being defended. May be an API model.
        sanitizer:          A :class:`PISanitizer`. Built with defaults if
                            omitted, which loads Llama-3.1-8B-Instruct.
        min_context_chars:  Skip sanitization for data channels shorter than
                            this many *characters* (not tokens — the guard runs
                            before tokenization). The method's thresholds were
                            tuned on long contexts; on very short ones the peak
                            finder has no baseline to stand out from. ``0``
                            disables the guard.
        pass_user_task:     Give the sanitizer the trusted instruction as
                            ``target_instruction``. Only meaningful with
                            ``anchor_prompt=3``, which anchors on the real task.

    Example::

        san = PISanitizer.from_local_llm(llm)
        defended = PISanitizerDefense(TargetLLM(APILLM("gpt-4o-mini")), sanitizer=san)

        result = AttackEvaluator(target=defended, attacker=IgnoreAttacker()).run(ds)
        print(defended.last_trace.summary())
    """

    def __init__(
        self,
        target: Victim,
        sanitizer: Optional[PISanitizer] = None,
        min_context_chars: int = 0,
        pass_user_task: bool = False,
    ):
        super().__init__(target)
        self.sanitizer = sanitizer if sanitizer is not None else PISanitizer()
        self.min_context_chars = min_context_chars
        self.pass_user_task = pass_user_task
        self.last_trace: Optional[SanitizationTrace] = None

    # Channel plumbing (``set_channels`` / ``clear_channels`` /
    # ``resolve_channels``) is inherited from ``Victim``.

    # -- Victim plumbing -----------------------------------------------------

    def preprocess_messages(self, messages: list[dict]) -> list[dict]:
        prompt = self.resolve_channels(messages)
        instruction, data = prompt.trusted_instruction, prompt.data

        if not data or not data.strip():
            log.warning(
                "[PISanitizer] No untrusted data channel found in this prompt; "
                "nothing to sanitize. Call set_channels() to supply the split."
            )
            self.last_trace = None
            return messages

        if self.min_context_chars and len(data) < self.min_context_chars:
            log.info(
                "[PISanitizer] Data channel is %d chars (< min_context_chars=%d); "
                "skipping sanitization.", len(data), self.min_context_chars,
            )
            self.last_trace = None
            return messages

        trace = self.sanitizer.sanitize_with_trace(
            data, target_instruction=instruction if self.pass_user_task else "",
        )
        self.last_trace = trace

        if not trace.changed:
            return messages

        # Substituting the sanitized span back used to mean finding it verbatim in
        # some message, and the defense went INERT — silently, bar a warning — when
        # the recovered span did not occur in exactly one turn. Re-rendering the
        # channel it came from cannot miss.
        return prompt.with_data(trace.sanitized).to_messages()

    def __repr__(self) -> str:
        return f"PISanitizerDefense(sanitizer={self.sanitizer!r}, target={self.target!r})"
