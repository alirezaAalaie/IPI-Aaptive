"""
AttackDataset — an ordered collection of ``Instance`` objects.

Ported from EasyJailbreak's ``JailbreakDataset``. Every component family takes and
returns one of these, which is what makes mutations, constraints, selectors and
evaluators composable.

Deviations from upstream (both deliberate)
------------------------------------------
  * **Not a ``torch.utils.data.Dataset``.** Upstream subclasses it and imports the
    HuggingFace ``datasets`` package at module import. ``import ipi`` must work on a
    machine with no torch (see CLAUDE.md — heavy deps are optional extras), so this is
    a plain sequence. Nothing in upstream's code path actually used the torch base
    class; it only supplied ``__getitem__``/``__len__``, which are defined here.
  * **No HuggingFace hub loading.** Upstream's ``__init__`` accepts a string and pulls
    from ``Lemhf14/EasyJailbreak_Datasets``. Our benchmark ships in ``ipi/data/``; a
    network fetch at dataset construction is not something we want on Kaggle.
"""
from __future__ import annotations

import json
import random
from typing import Iterable, Iterator, Optional, Union

from .instance import Instance

__all__ = ["AttackDataset"]


class AttackDataset:
    """
    A list of ``Instance`` objects with the slicing / merging helpers the components use.

    Args:
        dataset: A list (or any iterable) of ``Instance`` objects.
        shuffle: Shuffle on construction.
        seed:    RNG seed used when ``shuffle`` is True, for reproducibility.
                 (Upstream shuffles from the global RNG; we make it explicit.)
    """

    def __init__(
        self,
        dataset: Iterable[Instance] = (),
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        self._dataset: list[Instance] = list(dataset)
        bad = [type(x).__name__ for x in self._dataset if not isinstance(x, Instance)]
        if bad:
            raise TypeError(
                f"AttackDataset takes Instance objects; got {sorted(set(bad))}. "
                "Wrap raw records with Instance(**record) first."
            )
        self.shuffled = shuffle
        if shuffle:
            random.Random(seed).shuffle(self._dataset)

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key) -> Union[Instance, "AttackDataset"]:
        if isinstance(key, slice):
            return AttackDataset(self._dataset[key])
        return self._dataset[key]

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[Instance]:
        return iter(self._dataset)

    def __repr__(self) -> str:
        return f"AttackDataset(n={len(self._dataset)})"

    # ------------------------------------------------------------------
    # Mutation of the collection
    # ------------------------------------------------------------------

    def add(self, instance: Instance):
        """Append one instance."""
        if not isinstance(instance, Instance):
            raise TypeError(f"add() takes an Instance, got {type(instance).__name__}")
        self._dataset.append(instance)

    def extend(self, instances: Iterable[Instance]):
        """Append many instances."""
        for inst in instances:
            self.add(inst)

    def shuffle(self, seed: Optional[int] = None):
        """Shuffle in place."""
        random.Random(seed).shuffle(self._dataset)
        self.shuffled = True

    @classmethod
    def merge(cls, dataset_list: Iterable["AttackDataset"]) -> "AttackDataset":
        """Concatenate several datasets into one."""
        out: list[Instance] = []
        for ds in dataset_list:
            out.extend(ds._dataset)
        return cls(out)

    def subset(self, n: int, seed: int = 42) -> "AttackDataset":
        """A reproducible random subset of at most ``n`` instances."""
        rng = random.Random(seed)
        return AttackDataset(rng.sample(self._dataset, min(n, len(self._dataset))))

    def group_by(self, key) -> dict:
        """
        Group instances by ``key(instance)``.

        Used to aggregate results per scenario / per attack when one scenario has
        spawned many candidates.
        """
        out: dict = {}
        for inst in self._dataset:
            out.setdefault(key(inst), []).append(inst)
        return out

    def group_by_parents(self) -> dict:
        """Group instances by their first parent (identity-keyed)."""
        return self.group_by(lambda i: id(i.parents[0]) if i.parents else None)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dicts(self) -> list[dict]:
        """Dict view of every instance (lineage excluded)."""
        return [i.to_dict() for i in self._dataset]

    def save_to_json(self, path: str):
        """Write ``to_dicts()`` to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: str) -> "AttackDataset":
        """Load a JSON list of records, wrapping each in an ``Instance``."""
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return cls([Instance(**r) for r in records])

    @classmethod
    def load_jsonl(cls, path: str) -> "AttackDataset":
        """Load a JSONL file, wrapping each line in an ``Instance``."""
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Instance(**json.loads(line)))
        return cls(out)
