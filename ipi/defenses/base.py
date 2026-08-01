"""
DefendedVictim — Base wrapper class for defended target models.
"""
from __future__ import annotations

import logging
from typing import Optional
from ..victim import Victim

log = logging.getLogger(__name__)


class DefendedVictim(Victim):
    """
    Base wrapper for any defended victim LLM pipeline.

    Wraps an underlying Victim instance (e.g. TargetLLM or LocalLLM) and applies
    preprocessing and postprocessing hooks. Conforms strictly to the Victim interface
    so it can be evaluated against all 13 attacks in the benchmark.
    """

    def __init__(self, target: Victim):
        self.target = target

    @property
    def backend(self) -> str:
        return getattr(self.target, "backend", "api")

    @property
    def system_prompt(self) -> str:
        return getattr(self.target, "system_prompt", "")

    @system_prompt.setter
    def system_prompt(self, value: str):
        if hasattr(self.target, "system_prompt"):
            self.target.system_prompt = value

    @property
    def system_prompt_template(self) -> str:
        return getattr(self.target, "system_prompt_template", "")

    @system_prompt_template.setter
    def system_prompt_template(self, value: str):
        if hasattr(self.target, "system_prompt_template"):
            self.target.system_prompt_template = value

    @property
    def model_name(self) -> str:
        return getattr(self.target, "model_name", "")

    @property
    def max_bs(self) -> int:
        return getattr(self.target, "max_bs", 50)

    # Hooks for subclasses
    def preprocess_messages(self, messages: list[dict]) -> list[dict]:
        """Transform input messages before passing to the target model."""
        return messages

    def postprocess_response(self, response: str) -> str:
        """Transform or validate generated output before returning."""
        return response

    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        processed = self.preprocess_messages(messages)
        self.last_input_messages = processed
        raw_response = self.target.generate(
            processed, max_tokens=max_tokens, temperature=temperature
        )
        self.last_response = raw_response
        return self.postprocess_response(raw_response)


    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        processed = self.preprocess_messages(messages)
        return self.target.get_first_token_logprobs(processed, n_top=n_top)

    # HF model internals delegation for white-box attacks (BEAST, GCG, AutoDAN)
    @property
    def tokenizer(self):
        return self.target.tokenizer

    @property
    def hf_model(self):
        return self.target.hf_model

    @property
    def _device(self):
        return self.target._device

    def generate_n_tokens_batch(self, *args, **kwargs):
        return self.target.generate_n_tokens_batch(*args, **kwargs)

    def attack_objective_targeted(self, *args, **kwargs):
        return self.target.attack_objective_targeted(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.target.apply_chat_template(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(target={self.target!r})"
