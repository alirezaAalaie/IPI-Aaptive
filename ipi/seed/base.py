"""
SeedBase — a store of attack seeds.

Mirrors ``easyjailbreak.seed.seed_base.SeedBase``.
"""
from __future__ import annotations

from typing import Iterator, Optional

__all__ = ["SeedBase"]


class SeedBase:
    """A base class that can store and generate attack seeds."""

    def __init__(self, seeds: Optional[list[str]] = None):
        self.seeds: list[str] = list(seeds) if seeds else []

    def __iter__(self) -> Iterator[str]:
        return iter(self.seeds)

    def __len__(self) -> int:
        return len(self.seeds)

    def new_seeds(self, **kwargs):
        """Generate new seeds, replacing the stored batch."""
        raise NotImplementedError
