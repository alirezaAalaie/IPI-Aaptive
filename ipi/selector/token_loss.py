"""
TokenLossSelector — pick among candidates that differ only in one token span.

The white-box counterpart to ``ReferenceLossSelector``. That one takes text, renders a
chat prompt per instance and re-tokenizes it; this one takes candidates that are already
token ids and share a fixed prefix and suffix, so a batch is built by concatenation
alone. That difference is the whole reason GCG and BEAST could not use the text selector:
at 512 candidates per step, re-encoding the prompt 512 times is the run.

    full_ids = head_ids + <candidate adv ids> + tail_ids + target_ids
               \\________ fixed _________/                \\____ fixed ____/

``head_ids`` / ``tail_ids`` come from ``harness.split_optimization_prompt`` — the
victim's real prompt, IPI carrier and defense included — so the loss is computed on the
string the victim is actually given.

One objective, two attacks
--------------------------
GCG minimises cross-entropy of the target span; BEAST maximises
``-perplexity(target | prompt)``. Those are the same number with opposite signs, which is
why both recipes can share this selector. ``_loss`` is always the cross-entropy, lower is
better, and a caller that wants BEAST's convention negates it.

Uniform spans
-------------
Every candidate in a call must have the same adversarial length. Both callers satisfy
that — GCG's suffix is fixed-length and BEAST's beam grows one token at a time in
lockstep — and a mixed batch is refused rather than padded, because padding a span that
sits *before* the target would shift the target slice per row and silently score the
wrong positions.

``_build_batch`` is pure Python on purpose, the same split ``ReferenceLossSelector``
makes: the index arithmetic is the part that gets the off-by-one wrong, and keeping it
torch-free lets ``smoke_check.py`` verify it with no GPU and no torch installed.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..datasets import AttackDataset, Instance
from .base import SelectPolicy

log = logging.getLogger(__name__)

__all__ = ["TokenLossSelector", "ADV_IDS"]

#: Where a candidate carries its adversarial token span on the carrier.
ADV_IDS = "adv_ids"


class TokenLossSelector(SelectPolicy):
    """
    Score and rank token-level candidates by teacher-forced loss on the target span.

    Args:
        victim:     A local ``Victim`` (``hf_model`` + ``tokenizer``, ``backend == "local"``).
        head_ids:   Token ids before the adversarial span — prompt up to the injection.
        tail_ids:   Token ids after it — the close of the user turn plus the generation
                    prompt. BEAST calls this ``end_inst_token``.
        target_ids: The target string's ids. The loss is the mean CE over exactly these.
        batch_size: Candidates per forward pass. Unlike ``ReferenceLossSelector`` this
                    defaults to something finite (64), because its callers routinely
                    hand it 512 candidates and ``None`` would mean one batch of 512.
        keep:       How many candidates ``select()`` returns. 1 for GCG's greedy step,
                    ``k1`` for BEAST's beam.
    """

    def __init__(self, victim, head_ids: list[int], tail_ids: list[int],
                 target_ids: list[int], batch_size: int = 64, keep: int = 1):
        super().__init__(None)
        if getattr(victim, "backend", None) != "local":
            raise ValueError(
                "TokenLossSelector needs a local Victim (backend='local') exposing "
                f"hf_model + tokenizer; got backend={getattr(victim, 'backend', None)!r}")
        for attr in ("hf_model", "tokenizer"):
            if getattr(victim, attr, None) is None:
                raise ValueError(f"victim.{attr} is required by TokenLossSelector")
        if not target_ids:
            raise ValueError("target_ids is empty — there would be nothing to score")
        self.victim = victim
        self.head_ids = list(head_ids)
        self.tail_ids = list(tail_ids)
        self.target_ids = list(target_ids)
        self.batch_size = batch_size
        self.keep = keep

    # ------------------------------------------------------------------

    @staticmethod
    def adv_ids(instance: Instance) -> list[int]:
        """The candidate's adversarial span. Raises rather than scoring an empty one."""
        ids = instance.attack_attrs.get(ADV_IDS)
        if not ids:
            raise ValueError(
                f"instance {instance.id!r} carries no attack_attrs[{ADV_IDS!r}] — "
                "a token-level candidate must be created through a token mutation")
        return list(ids)

    def _build_batch(self, adv_rows: list[list[int]]) -> tuple[list[list[int]], int, int]:
        """
        ``(input_rows, target_start, target_len)`` — the index arithmetic, torch-free.

        Every row is ``head + adv + tail + target``. Because the adversarial spans are
        the same length, one ``target_start`` describes every row, and the batch needs no
        padding at all.
        """
        if not adv_rows:
            return [], 0, 0
        lengths = {len(row) for row in adv_rows}
        if len(lengths) != 1:
            raise ValueError(
                f"TokenLossSelector needs one adversarial length per batch, got {sorted(lengths)}")

        adv_len = lengths.pop()
        target_start = len(self.head_ids) + adv_len + len(self.tail_ids)
        rows = [
            self.head_ids + adv + self.tail_ids + self.target_ids
            for adv in adv_rows
        ]
        return rows, target_start, len(self.target_ids)

    def _score_batch(self, instances: list[Instance]) -> None:
        """One forward pass; writes ``instance._loss`` (mean CE over the target span)."""
        import torch
        import torch.nn.functional as F

        rows, target_start, target_len = self._build_batch(
            [self.adv_ids(i) for i in instances])
        if not rows:
            return

        model = self.victim.hf_model
        device = next(model.parameters()).device
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits

        # Position t's logits predict token t+1, so the target span's predictions start
        # one step earlier. Same shift as gradient.score_candidates.
        shift = target_start - 1
        tgt_logits = logits[:, shift: shift + target_len, :]
        tgt_ids = input_ids[:, target_start: target_start + target_len]
        b = input_ids.shape[0]
        losses = F.cross_entropy(
            tgt_logits.reshape(b * target_len, -1),
            tgt_ids.reshape(b * target_len),
            reduction="none",
        ).reshape(b, target_len).mean(dim=1)

        for instance, loss in zip(instances, losses.tolist()):
            instance._loss = float(loss)

    # ------------------------------------------------------------------

    def score(self, dataset: AttackDataset) -> AttackDataset:
        """Score every candidate, writing ``instance._loss``. Returns the dataset."""
        instances = list(dataset)
        for start in range(0, len(instances), self.batch_size):
            self._score_batch(instances[start: start + self.batch_size])
        return dataset

    def n_batches(self, n_candidates: int) -> int:
        """Forward passes ``score()`` would spend — for the ``n_forward_passes`` counter."""
        if n_candidates <= 0:
            return 0
        return (n_candidates + self.batch_size - 1) // self.batch_size

    def select(self, dataset: AttackDataset) -> AttackDataset:
        """The ``keep`` lowest-loss candidates, best first."""
        instances = list(self.score(dataset))
        if not instances:
            return AttackDataset([])
        ranked = sorted(instances, key=lambda i: i._loss)
        return AttackDataset(ranked[: self.keep])

    def update(self, dataset: AttackDataset):
        """Stateless — the loss is recomputed from the weights every step."""
        return

    def __repr__(self) -> str:
        return (f"TokenLossSelector(keep={self.keep}, batch_size={self.batch_size}, "
                f"target_len={len(self.target_ids)})")
