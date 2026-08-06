"""
SecAlign Defense Wrapper for `ipi` Benchmark Evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from ..base import DefendedVictim
from ...victim import Victim
from .config import PROMPT_FORMAT


class SecAlignDefense(DefendedVictim):
    """
    SecAlign Defense Wrapper for `ipi` Benchmark.
    Evaluates SecAlign fine-tuned models on prompt injection benchmarks.
    """

    def __init__(self, target: Victim, frontend_delimiters: str = "TextTextText"):
        super().__init__(target)
        self.frontend_delimiters = frontend_delimiters
        self.prompt_dict = PROMPT_FORMAT.get(frontend_delimiters, PROMPT_FORMAT["TextTextText"])

    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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

        sample_dict = {"instruction": inst, "input": inp}
        if inp:
            formatted = self.prompt_dict["prompt_input"].format_map(sample_dict)
        else:
            formatted = self.prompt_dict["prompt_no_input"].format_map(sample_dict)

        return [{"role": "user", "content": formatted}]
