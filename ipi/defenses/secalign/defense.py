"""
SecAlign Defense Wrapper for `ipi` Benchmark Evaluation.

Port of the inference path in
``code/defense/SecAlign-main/test.py`` (``form_llm_input`` + ``recursive_filter``).
"""
from __future__ import annotations

from typing import Sequence

from ...victim import Victim
from ..channels import StructuredChannelDefense
from .config import DELIMITERS, FILTERED_TOKENS, PROMPT_FORMAT


class SecAlignDefense(StructuredChannelDefense):
    """
    SecAlign defense wrapper.

    SecAlign shares StruQ's structured prompt and defensive filter; what differs
    is how the model got there — preference optimisation (DPO) against a
    rejected response that follows the injection, rather than SFT.

    The instruction/data split travels with the prompt (see ``ipi/channels.py``)
    rather than being parsed back out of it; call :meth:`set_channels` for a
    prompt built by hand.

    Args:
        target:                 Underlying Victim carrying the SecAlign model.
        frontend_delimiters:    Must match the scheme the model was aligned on.
        apply_defensive_filter: Strip ``FILTERED_TOKENS`` from the data channel.
    """

    filtered_tokens: Sequence[str] = FILTERED_TOKENS

    def __init__(
        self,
        target: Victim,
        frontend_delimiters: str = "TextTextText",
        apply_defensive_filter: bool = True,
    ):
        super().__init__(target, apply_defensive_filter=apply_defensive_filter)
        if frontend_delimiters not in DELIMITERS:
            raise ValueError(
                f"Unknown frontend_delimiters {frontend_delimiters!r}; "
                f"expected one of {sorted(DELIMITERS)}."
            )
        self.frontend_delimiters = frontend_delimiters
        self.prompt_dict = PROMPT_FORMAT[frontend_delimiters]

    def format_prompt(self, instruction: str, data: str) -> str:
        key = "prompt_input" if (data and data.strip()) else "prompt_no_input"
        return self.prompt_dict[key].format_map({"instruction": instruction, "input": data})
