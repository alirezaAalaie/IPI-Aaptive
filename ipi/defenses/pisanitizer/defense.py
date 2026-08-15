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
  ``(input_prompt, context)`` pair from its own dataset builders. Here the
  untrusted span is recovered from the harness messages list with
  ``split_instruction_data`` — the same recovery StruQ/SecAlign/DefensiveToken
  use, so defense-vs-defense rows differ by defense and not by prompt parsing.
  ``set_channels`` pins it when the caller knows the split.
* Upstream sanitizes the LongBench-style context, which is thousands of tokens.
  Short IPI contexts are still handled, but note the peak-finding floor
  (``height=0.005``, ``threshold=0.01``) was tuned against long contexts where
  attention per token is small; on a 40-token context, ordinary tokens clear
  those thresholds easily. See ``min_context_tokens``.
* The sanitizer only ever sees the *data* channel, never the trusted
  instruction, so it cannot delete the user's real task. Upstream passes the
  context alone for the same reason.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ...victim import Victim
from ..base import DefendedVictim
from ..channels import split_instruction_data
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
        min_context_tokens: Skip sanitization for data channels shorter than
                            this many characters. The method's thresholds were
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
        min_context_tokens: int = 0,
        pass_user_task: bool = False,
    ):
        super().__init__(target)
        self.sanitizer = sanitizer if sanitizer is not None else PISanitizer()
        self.min_context_tokens = min_context_tokens
        self.pass_user_task = pass_user_task
        self._channel_override: Optional[Tuple[str, str]] = None
        self.last_trace: Optional[SanitizationTrace] = None

    # -- explicit channel control (mirrors StructuredChannelDefense) ----------

    def set_channels(self, instruction: str, data: str) -> None:
        """Pin the instruction/data split instead of recovering it."""
        self._channel_override = (instruction, data)

    def clear_channels(self) -> None:
        self._channel_override = None

    def resolve_channels(self, messages: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
        if self._channel_override is not None:
            return self._channel_override
        return split_instruction_data(messages)

    # -- Victim plumbing -----------------------------------------------------

    @staticmethod
    def _substitute(messages: List[dict], original: str, sanitized: str) -> bool:
        """Replace the first occurrence of ``original`` across ``messages``."""
        for msg in messages:
            content = msg.get("content") or ""
            if original in content:
                msg["content"] = content.replace(original, sanitized, 1)
                return True
        return False

    def preprocess_messages(self, messages: list[dict]) -> list[dict]:
        instruction, data = self.resolve_channels(messages)

        if not data or not data.strip():
            log.warning(
                "[PISanitizer] No untrusted data channel found in this prompt; "
                "nothing to sanitize. Call set_channels() to supply the split."
            )
            self.last_trace = None
            return messages

        if self.min_context_tokens and len(data) < self.min_context_tokens:
            log.info(
                "[PISanitizer] Data channel is %d chars (< min_context_tokens=%d); "
                "skipping sanitization.", len(data), self.min_context_tokens,
            )
            self.last_trace = None
            return messages

        trace = self.sanitizer.sanitize_with_trace(
            data, target_instruction=instruction if self.pass_user_task else "",
        )
        self.last_trace = trace

        if not trace.changed:
            return messages

        new_msgs = copy.deepcopy(messages)
        if not self._substitute(new_msgs, data, trace.sanitized):
            log.warning(
                "[PISanitizer] Sanitized the data channel but could not locate it "
                "verbatim in any message, so the original prompt is being sent "
                "unchanged — the defense is INERT for this prompt. This happens "
                "when the recovered channel spans several turns. Use "
                "set_channels() with a span that occurs in one message."
            )
            return messages
        return new_msgs

    def __repr__(self) -> str:
        return f"PISanitizerDefense(sanitizer={self.sanitizer!r}, target={self.target!r})"
