"""
The two pieces of upstream StruQ/SecAlign machinery both structured defenses
share, plus the base class that renders a :class:`~ipi.channels.ChanneledPrompt`
into a structured query.

  * ``recursive_filter``            — StruQ/SecAlign ``test.py`` defensive filter.
  * ``format_with_other_delimiters``— StruQ/SecAlign ``struq.py`` delimiter
                                      randomisation used to build Completion
                                      training samples.
  * ``StructuredChannelDefense``    — instruction channel + filtered data channel
                                      + one pre-rendered turn.

Channel *recovery* used to live here — ``split_instruction_data`` matching
``_AGENTDOJO_RE`` / ``_ENV_RE`` / BIPIA markers against the rendered prompt, and
``transform_data_channel`` doing the same match to rewrite the untrusted span in
place. Both are gone. The split now travels with the prompt (``ipi/channels.py``)
exactly as StruQ keeps ``{instruction}`` and ``{input}`` as separate fields from
its dataset all the way to ``format_map``, and every defense here reads it via
``Victim.resolve_channels``. There is nothing left to parse, and no shared regex
that six defenses silently depend on.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Mapping, Optional, Sequence

from .base import DefendedVictim
from ..channels import ChanneledPrompt
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

    # Channel plumbing — ``set_channels`` / ``clear_channels`` /
    # ``resolve_channels`` — is inherited from ``Victim``. A prompt built by
    # ``ipi.harness`` carries its own split; ``set_channels`` pins one for a
    # prompt built by hand.

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
        prompt: ChanneledPrompt = self.resolve_channels(messages)
        instruction, data = prompt.trusted_instruction, prompt.data
        if data and self.apply_defensive_filter and self.filtered_tokens:
            data = recursive_filter(data, self.filtered_tokens)
        # A plain list, deliberately: the returned prompt is the model's own
        # rendering (delimiters, RAW_ROLE), and a defense chained *under* this one
        # must not re-render the channels and throw that away.
        return self._wrap(self.format_prompt(instruction, data))
