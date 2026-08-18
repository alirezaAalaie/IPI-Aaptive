"""
ipi.seed — the prompt / seed registry.

Mirrors EasyJailbreak's ``easyjailbreak/seed/``, and like upstream it is the single
home for every fixed string an attack uses — injection payloads, attacker system
prompts, scorer prompts, constraint prompts, in-context demonstrations. There is no
separate ``prompts.py``.

    from ipi.seed import SeedTemplate, render, sample_population, ica_demos
"""
from .base import SeedBase
from .template import (
    SeedTemplate,
    PLACEHOLDER,
    TARGET_PLACEHOLDER,
    load_seed_templates,
    render,
    sample_population,
    ica_demos,
)

__all__ = [
    "SeedBase",
    "SeedTemplate",
    "PLACEHOLDER",
    "TARGET_PLACEHOLDER",
    "load_seed_templates",
    "render",
    "sample_population",
    "ica_demos",
]
