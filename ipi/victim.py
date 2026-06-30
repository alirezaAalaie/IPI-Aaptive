"""
Victim — abstract interface for any victim LLM or defended pipeline.

Any system to be evaluated against IPI attacks must subclass Victim and
implement at minimum generate(). Additional methods are optional and raise
informative errors if called on a victim that does not support them.

Required for ALL attacks:
  generate(messages)           — called by TAP / PAIR / RS / Beam-RS / BEAST
                                 for response generation.

Required for RS / Beam-RS:
  get_first_token_logprobs()   — logprob-guided search; raises
                                 LogprobNotSupportedError by default.

Required for BEAST (white-box local access):
  tokenizer, hf_model, _device — HuggingFace model internals.
  generate_n_tokens_batch()    — batch forward pass for beam search.
  attack_objective_targeted()  — perplexity objective for BEAST scoring.
  apply_chat_template()        — chat template formatting.
  All raise LocalOnlyError by default.

Attributes
----------
  system_prompt  (str)  — prepended as a system turn in every messages list.
                          Set as an instance attr in subclass __init__. Default "".
  model_name     (str)  — model identifier. Used by RS for adv_init selection. Default "".
  backend        (str)  — "api" | "local". Override as class variable in subclass.
                          RS uses this for probability-threshold selection.
  max_bs         (int)  — batch size for local operations. Default 50.

Example — custom attention-tracker defense
------------------------------------------
    from ipi.victim import Victim

    class AttentionTrackerVictim(Victim):
        backend = "local"

        def __init__(self, model_path: str, system_prompt: str = ""):
            self._model = AttentionTrackerModel(model_path)
            self.system_prompt = system_prompt
            self.model_name    = model_path

        def generate(self, messages, max_tokens=200, temperature=0.0):
            return self._model.run_with_defense(messages)

        def get_first_token_logprobs(self, messages, n_top=20):
            return self._model.get_logprobs(messages, n=n_top)

        # Expose HF internals for BEAST support:
        @property
        def tokenizer(self): return self._model.tokenizer
        @property
        def hf_model(self):  return self._model.hf_model
        @property
        def _device(self):   return self._model.device
        def generate_n_tokens_batch(self, *a, **kw): return self._model.generate_batch(*a, **kw)
        def attack_objective_targeted(self, *a, **kw): return self._model.objective(*a, **kw)
        def apply_chat_template(self, *a, **kw): return self._model.tokenizer.apply_chat_template(*a, **kw)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from .llm_unified import LogprobNotSupportedError, LocalOnlyError

log = logging.getLogger(__name__)


class Victim(ABC):
    """
    Abstract interface for any victim / defended LLM pipeline.

    Subclass this to plug a custom defense into the IPI attack benchmark.
    Implement generate() to support all attacks.
    Implement get_first_token_logprobs() to support RS / Beam-RS.
    Implement the BEAST properties / methods to support BEAST.
    """

    # Class variables — override as class variables or instance attrs in subclasses.
    backend: ClassVar[str] = "api"   # "api" | "local"
    system_prompt: str = ""
    model_name: str = ""
    max_bs: int = 50

    # ------------------------------------------------------------------
    # Abstract — required for all attacks
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response from an OpenAI-format messages list.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            max_tokens:  Override default generation length. None = use default.
            temperature: Override default sampling temperature. None = use default.

        Returns:
            Generated response text.
        """

    # ------------------------------------------------------------------
    # Optional — needed for RS / Beam-RS logprob-guided search
    # ------------------------------------------------------------------

    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        """
        Return log-probabilities for the top-n most likely first tokens.

        Implement this to support RS and Beam-RS attacks.

        Returns:
            dict mapping token string → log-probability (negative float).

        Raises:
            LogprobNotSupportedError if not implemented.
        """
        raise LogprobNotSupportedError(
            f"{type(self).__name__} does not support get_first_token_logprobs(). "
            "Implement this method to use RS / Beam-RS attacks, or use an "
            "APILLM provider that supports logprobs (openai, deepseek, metis_openai)."
        )

    # ------------------------------------------------------------------
    # Optional — needed for BEAST (white-box local model access)
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """HuggingFace tokenizer. Required for BEAST and token-level RS."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not expose a tokenizer. "
            "BEAST and token-level RS require a local-model victim. "
            "Use TargetLLM(LocalLLM(...)) or set backend='local' in your Victim subclass."
        )

    @property
    def hf_model(self):
        """Raw HuggingFace model. Required for BEAST beam-search logit computation."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not expose hf_model. "
            "BEAST requires direct model access (backend='local')."
        )

    @property
    def _device(self):
        """Torch device of the local model. Required for BEAST."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not expose _device."
        )

    def generate_n_tokens_batch(self, *args, **kwargs):
        """Batch forward pass returning logits. Required for BEAST beam search."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not implement generate_n_tokens_batch(). "
            "BEAST requires this method for white-box logit access."
        )

    def attack_objective_targeted(self, *args, **kwargs):
        """Compute perplexity-based attack objective over target_str. Required for BEAST."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not implement attack_objective_targeted()."
        )

    def apply_chat_template(self, *args, **kwargs):
        """Apply the model's HuggingFace chat template. Required for BEAST."""
        raise LocalOnlyError(
            f"{type(self).__name__} does not implement apply_chat_template()."
        )

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def __call__(self, messages_or_injection) -> str:
        """
        Dual-mode callable.

        str input  → wraps as a user message (with system_prompt if set).
        list input → forwarded to generate() as-is.
        """
        if isinstance(messages_or_injection, str):
            msgs: list[dict] = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": messages_or_injection})
        elif isinstance(messages_or_injection, list):
            msgs = messages_or_injection
        else:
            raise TypeError(
                f"Expected str or list[dict], got {type(messages_or_injection).__name__}"
            )
        return self.generate(msgs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"model={self.model_name!r}, backend={self.backend!r})"
        )
