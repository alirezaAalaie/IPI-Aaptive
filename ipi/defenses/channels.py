"""
Instruction / data channel utilities shared by the structured-prompt defenses
(StruQ, SecAlign).

Both defenses need to know which part of a prompt is the *trusted instruction*
and which part is *untrusted data*. The IPI harness does not carry that split
explicitly: ``make_scenario_target_fn`` (``ipi/evaluator.py``) flattens a
scenario into an OpenAI-style messages list whose shape depends on the dataset.
Guessing "system = instruction, user = data" is wrong for every shape the
harness emits — in the AgentDojo shape the legitimate task lives *inside* the
user turn, and in the BIPIA shape the untrusted context lives inside the
*system* turn. This module recovers the split from the shapes actually emitted,
and lets a caller override it outright via ``set_channels``.

Also hosts the two pieces of upstream machinery both defenses share:

  * ``recursive_filter``            — StruQ/SecAlign ``test.py`` defensive filter.
  * ``format_with_other_delimiters``— StruQ/SecAlign ``struq.py`` delimiter
                                      randomisation used to build Completion
                                      training samples.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base import DefendedVictim
from ..victim import Victim

log = logging.getLogger(__name__)

#: Role marker for a fully pre-rendered prompt. ``LocalLLM._build_local_prompt_ids``
#: encodes a lone ``{"role": RAW_ROLE}`` turn verbatim instead of running it
#: through the model's chat template. StruQ/SecAlign models are fine-tuned on
#: bare Alpaca-format strings, so wrapping their structured query in a chat
#: template would feed them a format they were never aligned on.
RAW_ROLE = "raw"


# ---------------------------------------------------------------------------
# Defensive filter (StruQ test.py:81 / SecAlign test.py:81)
# ---------------------------------------------------------------------------

def recursive_filter(s: str, filtered_tokens: Sequence[str]) -> str:
    """
    Strip every token in ``filtered_tokens`` from ``s``, repeating until no
    token remains. The repetition matters: deleting a token can splice its
    neighbours into a fresh occurrence (``'#[INST]##'`` -> ``'##'``).

    This is the second half of the StruQ defense — the structured prompt only
    holds if the attacker cannot write the delimiters into the data channel.
    """
    filtered = False
    while not filtered:
        for f in filtered_tokens:
            if f in s:
                s = s.replace(f, '')
        filtered = True
        for f in filtered_tokens:
            if f in s:
                filtered = False
    return s


# ---------------------------------------------------------------------------
# Delimiter randomisation (StruQ struq.py:10 / SecAlign struq.py:10)
# ---------------------------------------------------------------------------

def format_with_other_delimiters(
    text: str,
    delimiters: Mapping[str, Sequence[str]],
    other_delm_tokens: Mapping[str, Sequence[str]],
    other_delm_for_test: int = 2,
    test: bool = False,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Replace any real delimiter occurring in ``text`` with a randomly sampled
    stand-in (``|USER COMMAND|:``, ``<<bot answer>>:``, ...).

    Upstream splits ``other_delm_tokens`` into a train pool (everything but the
    last ``other_delm_for_test`` entries) and a disjoint test pool (the last
    ``other_delm_for_test``), so evaluation uses delimiter styles never seen in
    training. ``test`` selects the pool.

    Direct port of ``format_with_other_delimiters`` with ``np.random`` swapped
    for an injectable ``random.Random`` so dataset generation is reproducible
    without touching global numpy state.
    """
    rng = rng or random
    test_idx = -other_delm_for_test

    def pool(name: str) -> Sequence[str]:
        toks = other_delm_tokens[name]
        return toks[test_idx:] if test else toks[:test_idx]

    mark = rng.choice(pool('mark')) + ':'

    def sample_delm(delm_name: str) -> str:
        role_name = 'user' if delm_name in ('inst', 'inpt') else 'asst'
        role = rng.choice(pool(role_name))
        delm = rng.choice(pool(delm_name))

        p = rng.random()
        if p < 1 / 3:
            return (role + delm).upper()
        elif p < 2 / 3:
            return (role + delm).lower()
        else:
            return role + delm

    for delm in delimiters.values():
        # Skip the chat-model delimiter sets — they contain '' / ' ', which
        # str.replace would splatter over every character boundary.
        if '' in delm or ' ' in delm:
            continue
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text


# ---------------------------------------------------------------------------
# Instruction / data channel recovery
# ---------------------------------------------------------------------------

# Shape emitted by make_scenario_target_fn for AgentDojo / generic scenarios
# (ipi/evaluator.py:426). The legitimate task and the poisoned environment data
# share one user turn.
_AGENTDOJO_RE = re.compile(r"\AUser Task:\n(.*?)\n\nContext:\n(.*)\Z", re.DOTALL)
_ENV_RE = re.compile(r"\A<env>\n?(.*?)\n?</env>\Z", re.DOTALL)

# Shapes emitted by _format_bipia_prompt_helper (ipi/evaluator.py:69). The
# untrusted context follows one of these markers, and runs either to the end of
# the turn or up to the trailing instruction.
_BIPIA_CONTEXT_MARKERS = (
    "Fix the bug in my code based on the guidance provided in the following Stack Overflow answer:\n",
    "Fix the bug based on the guidance provided in the following Stack Overflow answer:\n",
    "Here is the content of the email:\n",
    "Here is the content of the news:\n",
    "Here is the content of the form:\n",
    "Here is the news:\n",
)
_BIPIA_CONTEXT_TERMINATORS = ("\n\nQuestion: ", "\n\nConcisely reply")


def _split_bipia(text: str) -> Optional[Tuple[str, str]]:
    """Return ``(trusted_remainder, untrusted_context)`` or None if no match."""
    hit = min(
        ((text.find(m), m) for m in _BIPIA_CONTEXT_MARKERS if m in text),
        default=None,
    )
    if hit is None:
        return None
    marker_start, marker = hit
    start = marker_start + len(marker)

    ends = [text.find(t, start) for t in _BIPIA_CONTEXT_TERMINATORS]
    ends = [e for e in ends if e != -1]
    end = min(ends) if ends else len(text)

    trusted = (text[:start] + text[end:]).strip()
    return trusted, text[start:end].strip()


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p).strip()


def split_instruction_data(messages: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    """
    Recover ``(instruction, data)`` from a harness messages list.

    ``instruction`` is everything the pipeline owner controls (agent system
    prompt, tool schema, the user's legitimate task). ``data`` is the untrusted
    blob an attacker can write into. Returns ``data == ""`` when the split
    cannot be determined, which routes the caller to the ``prompt_no_input``
    template rather than silently demoting the real task into the data channel.
    """
    system = _join(*[m.get("content", "") or "" for m in messages if m.get("role") == "system"])
    user = _join(*[m.get("content", "") or "" for m in messages if m.get("role") == "user"])

    # 1. AgentDojo / generic scenario: "User Task:\n...\n\nContext:\n..."
    m = _AGENTDOJO_RE.match(user)
    if m:
        task, data = m.group(1).strip(), m.group(2).strip()
        env = _ENV_RE.match(data)
        if env:
            data = env.group(1).strip()
        return _join(system, task), data

    # 2. BIPIA: untrusted context sits inside the system turn (require_system)
    #    or inside the single user turn (no-system variant).
    found = _split_bipia(system)
    if found:
        return _join(found[0], user), found[1]
    found = _split_bipia(user)
    if found:
        return _join(system, found[0]), found[1]

    # 3. Unknown shape. A system turn plus a user turn is the one case where
    #    "system trusted / user untrusted" is a defensible reading.
    if system and user:
        return system, user

    log.warning(
        "[channels] Could not locate an untrusted data channel in this messages "
        "list; routing the whole prompt through the instruction channel. The "
        "structured-query defense is inert for this prompt — call set_channels() "
        "to supply the split explicitly."
    )
    return (user or system), ""


# ---------------------------------------------------------------------------
# Shared base for StruQ / SecAlign wrappers
# ---------------------------------------------------------------------------

class StructuredChannelDefense(DefendedVictim):
    """
    Base for defenses that reformat a prompt into separate instruction and data
    channels, filter the data channel, and hand the result to the model as a
    pre-rendered (non-chat-templated) string.

    Subclasses implement :meth:`format_prompt`.
    """

    #: Tokens stripped from the data channel before formatting.
    filtered_tokens: Sequence[str] = ()

    def __init__(self, target: Victim, apply_defensive_filter: bool = True):
        super().__init__(target)
        self.apply_defensive_filter = apply_defensive_filter
        self._channel_override: Optional[Tuple[str, str]] = None

    # -- explicit channel control -------------------------------------------

    def set_channels(self, instruction: str, data: str) -> None:
        """
        Pin the instruction/data split instead of recovering it from messages.

        Use this whenever the caller knows the split — it is always more
        reliable than parsing. Stays in effect until :meth:`clear_channels`.
        """
        self._channel_override = (instruction, data)

    def clear_channels(self) -> None:
        self._channel_override = None

    def resolve_channels(self, messages: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
        if self._channel_override is not None:
            return self._channel_override
        return split_instruction_data(messages)

    # -- subclass hook -------------------------------------------------------

    def format_prompt(self, instruction: str, data: str) -> str:
        raise NotImplementedError

    # -- Victim plumbing -----------------------------------------------------

    def _wrap(self, prompt: str) -> List[Dict[str, str]]:
        """
        Emit the rendered prompt as a single turn. Local backends get RAW_ROLE
        so the chat template is bypassed (these models are fine-tuned on bare
        Alpaca-format strings); API backends get a normal user turn, since they
        have no way to consume a raw prompt.
        """
        role = RAW_ROLE if self.backend == "local" else "user"
        return [{"role": role, "content": prompt}]

    def preprocess_messages(self, messages: list[dict]) -> list[dict]:
        instruction, data = self.resolve_channels(messages)
        if data and self.apply_defensive_filter and self.filtered_tokens:
            data = recursive_filter(data, self.filtered_tokens)
        return self._wrap(self.format_prompt(instruction, data))
