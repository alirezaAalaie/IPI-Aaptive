"""
PerplexityConstraint — drop candidates a perplexity filter would catch.

Port of ``easyjailbreak/constraint/PerplexityConstraint.py``. Computes strided perplexity
with a causal LM and keeps only text below ``threshold``.

Why an *attack* wants this
--------------------------
Perplexity filtering is a published **defense** against gradient attacks (Alon & Kamfonas
2023): GCG's output is high-entropy token soup, so a windowed PPL check catches it almost
perfectly. Used here as a constraint it inverts into the adaptive-attack move — the search
only keeps candidates that would survive that defense, so the ASR you report is against a
defended pipeline rather than against a filter you happened not to run. That makes it the
one component in this package that is a defense mechanism used offensively; do not confuse
it with `ipi/defenses/`, which is what the victim runs.

Needs ``torch`` + ``transformers``, imported lazily — ``import ipi`` must work without them.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..datasets import Instance
from .base import ConstraintBase

log = logging.getLogger(__name__)

__all__ = ["PerplexityConstraint"]


class PerplexityConstraint(ConstraintBase):
    """
    Keep candidates whose perplexity is at or below ``threshold``.

    Args:
        eval_model:     A local ``Victim`` (``hf_model`` + ``tokenizer``), or any object
                        exposing those two attributes. Upstream requires its
                        ``WhiteBoxModelBase``; our ``Victim`` contract replaces it.
        threshold:      Perplexity ceiling. Upstream's default is 500. The right value is
                        model- and corpus-specific — calibrate it on clean text from the
                        same distribution before reporting a number that depends on it.
        prompt_pattern: Which instance text is scored. Default ``"{query}"``.
        attr_name:      Attributes the pattern references. Default ``["query"]``.
        max_length:     Context window used per stride.
        stride:         Step between windows (the HF strided-perplexity recipe).
    """

    def __init__(self, eval_model, threshold: float = 500.0,
                 prompt_pattern: Optional[str] = None,
                 attr_name: Optional[list[str]] = None,
                 max_length: int = 512, stride: int = 512):
        if threshold <= 0:
            raise ValueError(f"threshold must be greater than 0, got {threshold}")
        tokenizer = getattr(eval_model, "tokenizer", None)
        model = getattr(eval_model, "hf_model", None) or getattr(eval_model, "model", None)
        if tokenizer is None or model is None:
            raise ValueError(
                "PerplexityConstraint needs a local model exposing `tokenizer` and "
                "`hf_model` (our Victim contract) — got "
                f"{type(eval_model).__name__} with neither.")
        self.eval_model = eval_model
        self.ppl_tokenizer = tokenizer
        self.ppl_model = model
        self.threshold = threshold
        self.max_length = max_length
        self.stride = stride
        self.prompt_pattern = "{query}" if prompt_pattern is None else prompt_pattern
        self.attr_name = list(attr_name or ["query"])

    def _format(self, instance: Instance) -> str:
        out = self.prompt_pattern
        for attr in self.attr_name:
            value = getattr(instance, attr, "") or ""
            if isinstance(value, (list, tuple)):
                value = value[-1] if value else ""
            out = out.replace("{" + attr + "}", str(value))
        return out

    def perplexity(self, text: str) -> float:
        """Strided perplexity, the HF recipe upstream uses."""
        import torch

        input_ids = torch.tensor(
            self.ppl_tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        if input_ids.size(1) < 2:
            return float("inf")

        eval_loss = []
        end_loc = 0
        with torch.no_grad():
            for i in range(0, input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, input_ids.size(1))
                trg_len = end_loc - i
                window = input_ids[:, begin_loc:end_loc].to(self.ppl_model.device)
                target_ids = window.clone()
                target_ids[:, :-trg_len] = -100
                outputs = self.ppl_model(window, labels=target_ids)
                eval_loss.append(outputs[0] * trg_len)
            return float(torch.exp(torch.stack(eval_loss).sum() / end_loc).item())

    def judge(self, text: str) -> bool:
        """True when the text is fluent enough to slip past a perplexity filter."""
        return self.perplexity(text) <= self.threshold

    def keep(self, instance: Instance, **kwargs) -> bool:
        return self.judge(self._format(instance))

    def __repr__(self) -> str:
        return f"PerplexityConstraint(threshold={self.threshold})"
