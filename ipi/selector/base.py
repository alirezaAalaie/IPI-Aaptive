"""
SelectPolicy — the ABC every selection strategy implements.

Mirrors ``easyjailbreak.selector.selector.SelectPolicy``. A policy owns a pool of
candidates and answers two questions across a search:

    select()          -> AttackDataset   which candidate(s) to spend the next query on
    update(dataset)                      what happened, so the next select() is better

The bookkeeping it reads lives on the ``Instance``: ``index`` (its slot in the policy's
reward array), ``visited_num`` (how often it was chosen), ``level`` (its depth in the
mutation tree) and ``children`` (its descendants). Those four fields are exactly why the
carrier had to exist — without them GPTFuzzer could only do flat UCB1 over a list.

The reward
----------
Every bandit policy here scores a candidate by ``Instance.num_jailbreak``, which is
``sum(eval_results)``. That is only meaningful when the evaluator wrote **booleans** — a
``*Judge`` / ``*Match`` from ``ipi.metrics``. Pairing one of these with a 1-10
``*GetScore`` evaluator sums scores instead of successes and silently rescales the
reward. Use ``SelectBasedOnScores`` for a graded signal, as upstream's TAP does.

Deviations from upstream
------------------------
  * **Seeded RNG.** Upstream draws from the global ``random`` / ``np.random``, so two
    runs of the same experiment differ. Each policy takes ``seed=``.
  * **``register()``.** Upstream extends its reward array by length but never assigns
    ``index`` to a candidate added mid-search — its fuzzer sets ``node.index`` inline,
    and a recipe that forgets silently reads another node's reward. Adding a candidate
    through the policy is the supported path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..datasets import AttackDataset, Instance

__all__ = ["SelectPolicy"]


class SelectPolicy(ABC):
    """
    Base class for a selection strategy over an ``AttackDataset``.

    Args:
        dataset: The candidate pool. Every instance gets ``visited_num = 0`` and a fresh
                 ``index`` (upstream behaviour — the index is the policy's reward slot).
        seed:    RNG seed for the policies that sample.
    """

    def __init__(self, dataset: Optional[AttackDataset] = None, seed: Optional[int] = None):
        self.dataset = dataset if dataset is not None else AttackDataset([])
        self.seed = seed
        for k, instance in enumerate(self.dataset):
            instance.visited_num = 0
            instance.index = k

    # ------------------------------------------------------------------
    # The seam
    # ------------------------------------------------------------------

    @abstractmethod
    def select(self) -> AttackDataset:
        """Pick the candidate(s) to try next."""

    def update(self, dataset: AttackDataset):
        """Fold the outcome of the last selection back into the policy's state."""

    def initial(self):
        """Reset any internal state."""

    # ------------------------------------------------------------------
    # Pool growth
    # ------------------------------------------------------------------

    def register(self, instance: Instance) -> Instance:
        """
        Add a candidate discovered mid-search, giving it its reward slot.

        Population-based attacks grow their pool as they go (GPTFuzzer adds every
        surviving mutant). Going through here guarantees ``index`` and the policy's
        per-candidate state stay in step; see the module docstring.
        """
        instance.index = len(self.dataset)
        if getattr(instance, "visited_num", None) is None:
            instance.visited_num = 0
        self.dataset.add(instance)
        self._grow(len(self.dataset))
        return instance

    def _grow(self, size: int):
        """Hook: extend any per-candidate arrays to ``size``. Overridden by subclasses."""

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self.dataset)})"
