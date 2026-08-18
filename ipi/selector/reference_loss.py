"""
ReferenceLossSelector — pick the candidate whose prompt makes the target most likely.

Port of ``easyjailbreak/selector/ReferenceLossSelector.py``, the white-box selector GCG
and BEAST use: score every candidate by cross-entropy of ``reference_responses[0]``
given the prompt, and keep the lowest. No victim queries are spent — it is a forward
pass, which is why gradient attacks can afford hundreds of candidates per step.

Args on the model
-----------------
Upstream takes its ``WhiteBoxModelBase`` and calls ``model_utils.encode_trace``. We take
a ``Victim`` with ``backend == "local"`` exposing ``hf_model`` + ``tokenizer`` — our own
contract, the same one ``attacks/{gcg,autodan,beast}.py`` optimise against — and build
the trace with ``tokenizer.apply_chat_template``, so it works with any modern HF model
rather than a hard-coded conversation template.

``torch`` is imported lazily: ``import ipi`` must work without it.

Universal vs per-instance
-------------------------
``is_universal=False`` (default) selects independently within each parent's group, so a
batch of candidates from different scenarios does not collapse to one winner.
``is_universal=True`` sums loss across the whole batch per ``jailbreak_prompt`` and picks
the single prompt that is best on average — the "one suffix for every goal" setting.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..datasets import AttackDataset, Instance
from .base import SelectPolicy

log = logging.getLogger(__name__)

__all__ = ["ReferenceLossSelector"]


class ReferenceLossSelector(SelectPolicy):
    """
    Select by teacher-forced loss on the reference response.

    Args:
        victim:       A local ``Victim`` (``hf_model`` + ``tokenizer``, ``backend == "local"``).
        batch_size:   Candidates scored per forward batch. ``None`` = one batch.
        is_universal: Score each ``jailbreak_prompt`` across the whole dataset rather
                      than within each parent's group.
    """

    def __init__(self, victim, batch_size: Optional[int] = None,
                 is_universal: bool = False):
        super().__init__(None)
        if getattr(victim, "backend", None) != "local":
            raise ValueError(
                "ReferenceLossSelector needs a local Victim (backend='local') exposing "
                f"hf_model + tokenizer; got backend={getattr(victim, 'backend', None)!r}")
        for attr in ("hf_model", "tokenizer"):
            if getattr(victim, attr, None) is None:
                raise ValueError(f"victim.{attr} is required by ReferenceLossSelector")
        self.victim = victim
        self.batch_size = batch_size
        self.is_universal = is_universal

    # ------------------------------------------------------------------

    def _prompt_text(self, instance: Instance) -> str:
        """The text the victim sees, up to where it starts generating."""
        tokenizer = self.victim.tokenizer
        jailbreak_prompt = instance.jailbreak_prompt or "{query}"
        content = jailbreak_prompt.replace("{query}", instance.query or "")

        messages = []
        system_prompt = getattr(self.victim, "system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(f"[{m['role'].upper()}] {m['content']}"
                             for m in messages) + "\n[ASSISTANT] "

    def _loss(self, instance: Instance) -> float:
        """Mean cross-entropy of the reference response given the prompt."""
        import torch

        tokenizer = self.victim.tokenizer
        model = self.victim.hf_model
        device = next(model.parameters()).device

        if not instance.reference_responses:
            raise ValueError(
                f"instance {instance.id!r} has no reference_responses to score against")
        if len(instance.reference_responses) > 1:
            log.warning(
                "ReferenceLossSelector uses reference_responses[0]; instance %r carries "
                "%d — the rest are ignored.", instance.id, len(instance.reference_responses))

        prompt_text = self._prompt_text(instance)
        full_text = prompt_text + instance.reference_responses[0]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False,
                                      return_tensors="pt").to(device)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False,
                                    return_tensors="pt").to(device)
        prompt_len = prompt_ids.shape[1]
        if full_ids.shape[1] <= prompt_len:
            return float("inf")

        with torch.no_grad():
            logits = model(input_ids=full_ids, use_cache=False).logits

        target_len = full_ids.shape[1] - prompt_len
        logits_slice = logits[0, prompt_len - 1: prompt_len - 1 + target_len, :]
        target_slice = full_ids[0, prompt_len: prompt_len + target_len]
        loss = torch.nn.functional.cross_entropy(logits_slice, target_slice)
        return float(loss.item())

    # ------------------------------------------------------------------

    def select(self, dataset: AttackDataset) -> AttackDataset:
        """The lowest-loss group of candidates. Writes ``instance._loss`` on the way."""
        if not self.is_universal and len(dataset.group_by_parents()) > 1:
            return AttackDataset.merge([
                self.select(AttackDataset(group))
                for group in dataset.group_by_parents().values()
            ])

        for instance in dataset:
            instance._loss = self._loss(instance)

        best_group = None
        best_loss = None
        for group in dataset.group_by(lambda i: i.jailbreak_prompt).values():
            total_loss = sum(instance._loss for instance in group)
            if best_loss is None or total_loss < best_loss:
                best_loss, best_group = total_loss, group

        if best_group is None:
            return AttackDataset([])
        log.info("Loss selection: best loss = %s", best_loss)
        log.debug("Loss selection: best jailbreak prompt = %r", best_group[0].jailbreak_prompt)
        return AttackDataset(best_group)
