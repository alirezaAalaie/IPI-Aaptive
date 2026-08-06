"""
StruQ Defense Wrapper for ipi framework.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Any

from ..base import DefendedVictim
from ...victim import Victim
from .config import STRUQ_DELIMITERS, SYS_INPUT_PREFIX, SYS_NO_INPUT_PREFIX


def format_struq_prompt(
    instruction: str,
    input_text: str = "",
    delimiter_scheme: str = "SpclSpclSpcl",
) -> str:
    """
    Formats instruction and input using StruQ structured query delimiters.

    Args:
        instruction: Trusted instruction / prompt.
        input_text: Untrusted data / context.
        delimiter_scheme: Delimiter configuration key (e.g. 'SpclSpclSpcl' or 'TextTextText').

    Returns:
        Structured query string with separate instruction, input, and response channels.
    """
    delm = STRUQ_DELIMITERS.get(delimiter_scheme, STRUQ_DELIMITERS["SpclSpclSpcl"])
    if input_text and input_text.strip():
        return (
            f"{SYS_INPUT_PREFIX}"
            f"{delm[0]}\n{instruction}\n\n"
            f"{delm[1]}\n{input_text}\n\n"
            f"{delm[2]}\n"
        )
    else:
        return (
            f"{SYS_NO_INPUT_PREFIX}"
            f"{delm[0]}\n{instruction}\n\n"
            f"{delm[2]}\n"
        )


class StruQDefense(DefendedVictim):
    """
    StruQ Defense Wrapper for `ipi` Benchmark.

    StruQ separates instructions and untrusted data into distinct channels using special
    structured delimiters (e.g., `[MARK] [INST] [COLN]` vs `[MARK] [INPT] [COLN]`).
    When combined with a StruQ fine-tuned model, instructions embedded in the input
    channel are ignored by the model.

    Args:
        target: Underlying Victim instance (e.g. LocalLLM loaded with StruQ model/adapter).
        delimiter_scheme: Delimiter scheme name ('SpclSpclSpcl', 'TextTextText', etc.).
        use_struq_template: Whether to reformat messages into StruQ structured channels.
    """

    def __init__(
        self,
        target: Victim,
        delimiter_scheme: str = "SpclSpclSpcl",
        use_struq_template: bool = True,
    ):
        super().__init__(target)
        self.delimiter_scheme = delimiter_scheme
        self.use_struq_template = use_struq_template

    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.use_struq_template:
            return messages

        new_msgs = deepcopy(messages)
        user_content = ""
        system_content = ""

        for msg in new_msgs:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")

        inst = system_content if system_content else "Follow the instructions in the prompt."
        inp = user_content

        formatted_prompt = format_struq_prompt(
            instruction=inst,
            input_text=inp,
            delimiter_scheme=self.delimiter_scheme,
        )
        return [{"role": "user", "content": formatted_prompt}]
