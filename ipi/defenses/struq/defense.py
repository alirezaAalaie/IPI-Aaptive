"""
StruQ Defense Wrapper for the ipi framework.

Port of the inference path in ``code/defense/StruQ-main/test.py``
(``form_llm_input`` + ``recursive_filter``).
"""
from __future__ import annotations

from typing import Sequence

from ...victim import Victim
from ..channels import StructuredChannelDefense
from .config import FILTERED_TOKENS, PROMPT_FORMAT, STRUQ_DELIMITERS


def format_struq_prompt(
    instruction: str,
    input_text: str = "",
    delimiter_scheme: str = "SpclSpclSpcl",
) -> str:
    """
    Render a structured query.

    This is the raw formatter and does **not** sanitise ``input_text`` — dataset
    construction deliberately places delimiters in the data channel. Inference
    goes through :class:`StruQDefense`, which filters first.

    Args:
        instruction:      Trusted instruction (the instruction channel).
        input_text:       Untrusted data (the data channel). Empty selects the
                          ``prompt_no_input`` template.
        delimiter_scheme: Key into ``STRUQ_DELIMITERS``.
    """
    prompt_dict = PROMPT_FORMAT.get(delimiter_scheme) or PROMPT_FORMAT["SpclSpclSpcl"]
    key = "prompt_input" if (input_text and input_text.strip()) else "prompt_no_input"
    return prompt_dict[key].format_map({"instruction": instruction, "input": input_text})


class StruQDefense(StructuredChannelDefense):
    """
    StruQ defense wrapper.

    StruQ has two halves, and both are needed:

    1. **Structured query** — instruction and untrusted data go in separately
       delimited channels (``[MARK] [INST][COLN]`` vs ``[MARK] [INPT][COLN]``),
       and the model is fine-tuned to only ever obey the instruction channel.
    2. **Defensive filter** — the delimiter tokens are stripped from the data
       channel, so the attacker cannot close the channel and open a fake
       instruction one.

    The instruction/data split travels with the prompt (see ``ipi/channels.py``)
    rather than being parsed back out of it; call :meth:`set_channels` for a
    prompt built by hand.

    Args:
        target:                 Underlying Victim (a LocalLLM carrying the StruQ
                                model/adapter, wrapped in TargetLLM).
        delimiter_scheme:       Must match what the model was trained with.
        apply_defensive_filter: Strip ``FILTERED_TOKENS`` from the data channel.
                                Only turn this off to measure what the filter
                                is worth on its own.
        use_struq_template:     False passes messages through untouched, for an
                                undefended control run.
    """

    filtered_tokens: Sequence[str] = FILTERED_TOKENS

    def __init__(
        self,
        target: Victim,
        delimiter_scheme: str = "SpclSpclSpcl",
        apply_defensive_filter: bool = True,
        use_struq_template: bool = True,
    ):
        super().__init__(target, apply_defensive_filter=apply_defensive_filter)
        if delimiter_scheme not in STRUQ_DELIMITERS:
            raise ValueError(
                f"Unknown delimiter_scheme {delimiter_scheme!r}; "
                f"expected one of {sorted(STRUQ_DELIMITERS)}."
            )
        self.delimiter_scheme = delimiter_scheme
        self.use_struq_template = use_struq_template

    def format_prompt(self, instruction: str, data: str) -> str:
        return format_struq_prompt(
            instruction=instruction,
            input_text=data,
            delimiter_scheme=self.delimiter_scheme,
        )

    def preprocess_messages(self, messages: list[dict]) -> list[dict]:
        if not self.use_struq_template:
            return messages
        return super().preprocess_messages(messages)
