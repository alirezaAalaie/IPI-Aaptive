"""
ConstraintBase — the ABC every filter implements.

Mirrors ``easyjailbreak.constraint.ConstraintBase.ConstraintBase``. A constraint runs
*after* a mutation and drops the candidates that are not worth a victim query:

    ConstraintBase.__call__(dataset) -> dataset      # filters, never rewrites

That is the whole contract, and the reason it is a separate family from ``mutation/``:
a mutation is 1-to-n and records lineage, a constraint is n-to-m and records nothing. The
payoff is budget — every candidate a constraint removes is a target query not spent, which
is what makes TAP's tree affordable at all.

Deviations from upstream
------------------------
  * ``__call__`` is concrete here, built on an abstract ``keep(instance)`` predicate, so a
    new filter is one method. Upstream makes ``__call__`` itself abstract and every
    subclass re-implements the same loop. Filters that need to see the whole batch at once
    (``DeleteOffTopic`` ranks and truncates) still override ``__call__``.
  * Seeded RNG, as in ``selector/`` — upstream draws from the global ``np.random``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..datasets import AttackDataset, Instance

__all__ = ["ConstraintBase"]


class ConstraintBase(ABC):
    """Filter an ``AttackDataset`` down to the candidates worth querying with."""

    def __call__(self, dataset: AttackDataset, *args, **kwargs) -> AttackDataset:
        """Keep the instances ``keep()`` accepts, in their original order."""
        return AttackDataset([i for i in dataset if self.keep(i, **kwargs)])

    @abstractmethod
    def keep(self, instance: Instance, **kwargs) -> bool:
        """True if this candidate survives the filter."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
