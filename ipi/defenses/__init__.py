"""
IPI Defenses Package.
"""
from .base import DefendedVictim
from .in_context import (
    InstructionalDefense,
    ReminderDefense,
    SandwichDefense,
    SpotlightDefense,
    CompositeDefense,
)

__all__ = [
    "DefendedVictim",
    "InstructionalDefense",
    "ReminderDefense",
    "SandwichDefense",
    "SpotlightDefense",
    "CompositeDefense",
]
