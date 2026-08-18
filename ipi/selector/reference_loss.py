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

Batching
--------
``batch_size`` is honoured: candidates are padded, stacked and scored in one forward
pass per batch, using upstream's masked-label formulation (``labels`` = ``-100``
everywhere except the reference span, then per-row mean over the unmasked positions).
That is mathematically identical to slicing each sequence on its own — with right
padding and causal attention, no real position can attend to a pad — so a batched score
and a one-at-a-time score are the same number.

This matters because the selector was written for GCG- and BEAST-scale candidate sets.
Scoring 512 candidates one at a time is ~512 forward passes; batching them at 64 is 8.
``batch_size=None`` means "one batch for the whole dataset", which is upstream's default
and will OOM on a large candidate set — set it explicitly for anything gradient-scale.

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
        batch_size:   Candidates scored per forward batch. ``None`` (upstream's default)
                      means one batch for the whole dataset — fine for a handful of
                      candidates, an OOM for a gradient-scale set. Set it explicitly there.
        is_universal: Score each ``jailbreak_prompt`` across the whole dataset rather
                      than within each parent's group.
        prompt_builder: ``Callable(injection: str) -> messages``, normally
                      ``harness.build_optimization_messages`` bound to the instance.
                      Without it the selector builds a bare ``[system][user]`` pair,
                      which is *not* what the victim is given for an IPI scenario — the
                      same mismatch that made GCG / BEAST / AutoDAN optimize a prompt
                      nobody sends. Pass it for any benchmark run.
    """

    def __init__(self, victim, batch_size: Optional[int] = None,
                 is_universal: bool = False, prompt_builder=None):
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
        self.prompt_builder = prompt_builder

    # ------------------------------------------------------------------

    def _prompt_text(self, instance: Instance) -> str:
        """
        The text the victim sees, up to where it starts generating.

        Goes through ``prompt_builder`` when one is given, so the loss is computed on
        the victim's real prompt — IPI carrier and defense included — rather than a bare
        ``[system][user]`` pair. The bare shape stays the default because the selector
        is also usable standalone, but scoring against it while judging success through
        ``harness.make_target_fn`` is the mismatch that invalidated the white-box rows.
        """
        from ..llm_unified import render_messages

        jailbreak_prompt = instance.jailbreak_prompt or "{query}"
        content = jailbreak_prompt.replace("{query}", instance.query or "")

        if self.prompt_builder is not None:
            messages = self.prompt_builder(content)
        else:
            messages = []
            system_prompt = getattr(self.victim, "system_prompt", "")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

        text, _ = render_messages(self.victim.tokenizer, messages)
        return text

    def _pad_id(self) -> int:
        """A token id safe to pad with. Value is irrelevant — pads are masked out."""
        tok = self.victim.tokenizer
        for candidate in (tok.pad_token_id, tok.eos_token_id):
            if candidate is not None:
                return int(candidate)
        return 0

    def _encode(self, instance: Instance) -> tuple[list[int], int, int]:
        """
        ``(full_ids, prompt_len, target_len)`` for one instance.

        ``target_len == 0`` marks a degenerate instance (the reference response encodes
        to nothing after the prompt); the caller scores it ``inf`` rather than dividing
        by zero.
        """
        tokenizer = self.victim.tokenizer
        if not instance.reference_responses:
            raise ValueError(
                f"instance {instance.id!r} has no reference_responses to score against")
        if len(instance.reference_responses) > 1:
            log.warning(
                "ReferenceLossSelector uses reference_responses[0]; instance %r carries "
                "%d — the rest are ignored.", instance.id, len(instance.reference_responses))

        prompt_text = self._prompt_text(instance)
        full_text = prompt_text + instance.reference_responses[0]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        return full_ids, prompt_len, max(0, len(full_ids) - prompt_len)

    def _build_batch(self, instances: list[Instance]):
        """
        Pad and label a batch — the index arithmetic, with no torch in sight.

        Split out from ``_score_batch`` on purpose: this is the part that gets the
        off-by-one wrong, and keeping it pure Python means ``smoke_check.py`` can verify
        it on a machine with no GPU and no torch installed.

        Returns ``(rows, input_rows, label_rows, mask_rows)`` where ``rows`` are the
        indices into ``instances`` that are actually scorable. A row whose reference span
        encodes to nothing is left out and scored ``inf`` by the caller — including it
        would divide by a zero token count.

        ``labels`` is ``-100`` everywhere except the reference span, upstream's
        formulation: ``cross_entropy`` ignores ``-100``, so neither the prompt nor the
        padding contributes to the loss.
        """
        pad_id = self._pad_id()
        encoded = [self._encode(instance) for instance in instances]
        rows = [i for i, (_, _, target_len) in enumerate(encoded) if target_len > 0]
        if not rows:
            return [], [], [], []

        max_len = max(len(encoded[i][0]) for i in rows)
        input_rows, label_rows, mask_rows = [], [], []
        for i in rows:
            full_ids, prompt_len, target_len = encoded[i]
            pad = max_len - len(full_ids)
            input_rows.append(full_ids + [pad_id] * pad)
            labels = [-100] * (len(full_ids) + pad)
            labels[prompt_len: prompt_len + target_len] = \
                full_ids[prompt_len: prompt_len + target_len]
            label_rows.append(labels)
            mask_rows.append([1] * len(full_ids) + [0] * pad)
        return rows, input_rows, label_rows, mask_rows

    def _score_batch(self, instances: list[Instance]) -> None:
        """
        Score a batch in one forward pass, writing ``instance._loss`` on each.

        Shift by one, then a per-row mean over the unmasked positions — identical to
        slicing each sequence individually, because right padding plus causal attention
        means no real position can attend to a pad.
        """
        import torch
        import torch.nn.functional as F

        rows, input_rows, label_rows, mask_rows = self._build_batch(instances)

        scorable = set(rows)
        for i, instance in enumerate(instances):
            if i not in scorable:
                instance._loss = float("inf")
        if not rows:
            return

        model = self.victim.hf_model
        device = next(model.parameters()).device
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        labels    = torch.tensor(label_rows, dtype=torch.long, device=device)
        attention = torch.tensor(mask_rows,  dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention,
                           use_cache=False).logits

        shift_logits = logits[:, :-1, :].transpose(1, 2)      # B x V x (L-1)
        shift_labels = labels[:, 1:]                          # B x (L-1)
        per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        valid = (shift_labels != -100)
        losses = (per_token * valid).sum(dim=1) / valid.sum(dim=1)

        for row, i in enumerate(rows):
            instances[i]._loss = float(losses[row].item())

    # ------------------------------------------------------------------

    def score(self, dataset: AttackDataset) -> AttackDataset:
        """
        Score every candidate, writing ``instance._loss``. Returns the dataset.

        ``select()`` keeps only the winner, which is what GCG and BEAST want. A genetic
        search needs the whole fitness vector — AutoDAN's roulette wheel is over all of
        it — so the scoring pass is public on its own. Same batching, same masked-label
        formulation; the only difference is that nothing is discarded.
        """
        instances = list(dataset)
        # `or 1` guards the empty dataset: range(0, 0, 0) raises rather than doing nothing.
        size = self.batch_size or len(instances) or 1
        for start in range(0, len(instances), size):
            self._score_batch(instances[start: start + size])
        return dataset

    def select(self, dataset: AttackDataset) -> AttackDataset:
        """The lowest-loss group of candidates. Writes ``instance._loss`` on the way."""
        if not self.is_universal and len(dataset.group_by_parents()) > 1:
            return AttackDataset.merge([
                self.select(AttackDataset(group))
                for group in dataset.group_by_parents().values()
            ])

        self.score(dataset)

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
