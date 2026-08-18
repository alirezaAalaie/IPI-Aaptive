"""
SeedTemplate — the prompt/seed registry.

This is the **single home for every fixed string an attack uses**: injection payload
templates, attacker system prompts, scorer system prompts, constraint prompts and
in-context demonstrations. ``ipi/prompts.py`` was dissolved into it.

Schema
------
``seed_templates.json`` is keyed three levels deep::

    {usage: {method: {variant: [templates]}}}

  usage    which model the text configures, or what it is
             attack     — the injected payload, or the attacker LLM's system prompt
             judge      — the scorer LLM's system prompt
             constraint — a filter LLM's system prompt
             demo       — in-context (goal, target) demonstration pairs
  method   the paper/recipe it belongs to (TAP, PAIR, AutoDAN, Gptfuzzer, ICA, ...)
  variant  ``original`` = verbatim from the reference implementation
           ``ipi``      = this repo's IPI-reframed text
           A method may carry a further IPI variant when it genuinely has more than
           one reframing: ``attack.TAP.ipi_universal`` is the task-universal
           prefix/suffix attacker prompt, picked by ``prompt_mode="ipi_universal"``.
           Ask for a variant by name — never index into a pool positionally.

EasyJailbreak's schema is ``{usage: {method: [templates]}}`` with usage ∈
{attack, judge}. Ours is a strict superset: the extra ``variant`` level exists because
we must carry *both* the paper-faithful text (for reproduction rows) and the
IPI-reframed text (for the attack rows) under the same method. Flattening the two axes
into one key is what previously pushed the TAP/PAIR prompts out of the registry and into
a separate ``prompts.py``.

Substitution
------------
``render()`` uses ``str.replace``, never ``str.format``. Many templates contain literal
braces — the JSON response examples in the TAP/PAIR attacker prompts, LaTeX in the
ReNeLLM scenarios — which would make ``format()`` raise.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from .base import SeedBase

__all__ = [
    "SeedTemplate", "PLACEHOLDER", "TARGET_PLACEHOLDER",
    "load_seed_templates", "render", "sample_population", "ica_demos",
]

PLACEHOLDER = "{query}"
TARGET_PLACEHOLDER = "{target_str}"

_TEMPLATE_FILE = "seed_templates.json"
_cache: Optional[dict] = None


def load_seed_templates(template_file: Optional[str] = None) -> dict:
    """
    Load the registry JSON. Cached when using the packaged default.

    Args:
        template_file: Path to an external JSON with the same schema.
                       None (default) loads the packaged ``seed_templates.json``.
    """
    global _cache
    if template_file is not None:
        with open(template_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if _cache is None:
        try:
            from importlib.resources import files
            raw = (files(__package__) / _TEMPLATE_FILE).read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - odd install layouts
            import os
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, _TEMPLATE_FILE), "r", encoding="utf-8") as f:
                raw = f.read()
        _cache = json.loads(raw)
    return _cache


class SeedTemplate(SeedBase):
    """
    Pulls templates out of the registry.

    Example:
        >>> SeedTemplate().new_seeds(method_list=["Gptfuzzer"])            # 77 verbatim
        >>> SeedTemplate().new_seeds(method_list=["AutoDAN"], variant="ipi")
        >>> SeedTemplate().new_seeds(prompt_usage="judge", method_list=["IPI"],
        ...                          variant="ipi")[0]                    # scorer prompt
    """

    def __init__(self, seeds: Optional[list[str]] = None):
        super().__init__(seeds)

    def new_seeds(
        self,
        seeds_num: Optional[int] = None,
        prompt_usage: str = "attack",
        method_list: Optional[list[str]] = None,
        variant: str = "original",
        template_file: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> list:
        """
        Pull a pool (or a random sample of it) from the registry.

        Args:
            seeds_num:    How many to return. None returns the whole pool.
            prompt_usage: ``attack`` | ``judge`` | ``constraint`` | ``demo``.
            method_list:  Method names, e.g. ``["Gptfuzzer"]``, ``["TAP"]``.
            variant:      ``original`` (verbatim upstream) or ``ipi``.
            template_file: Optional external JSON (same schema).
            seed:         RNG seed for reproducible sampling.

        Returns:
            List of templates (or of demo dicts, for ``prompt_usage="demo"``).

        Raises:
            AttributeError: if a requested usage/method/variant is absent — matches
                            upstream, which also fails loudly rather than silently
                            returning an empty pool.
            AssertionError: if ``seeds_num`` exceeds the pool size (upstream behaviour;
                            use ``sample_population`` when you need more).
        """
        self.seeds = []
        if method_list is None:
            method_list = ["IPI"]

        registry = load_seed_templates(template_file)
        section = registry.get(prompt_usage)
        if section is None:
            raise AttributeError(
                f"seed registry has no {prompt_usage!r} usage. Available: "
                f"{sorted(k for k in registry if not k.startswith('_'))}"
            )

        pool: list = []
        for method in method_list:
            if method not in section:
                raise AttributeError(
                    f"seed registry has no {prompt_usage!r} entry for method {method!r}. "
                    f"Available: {sorted(section)}"
                )
            variants = section[method]
            if variant not in variants:
                raise AttributeError(
                    f"{prompt_usage}.{method} has no {variant!r} variant. "
                    f"Available: {sorted(variants)}"
                )
            pool.extend(variants[variant])

        if seeds_num is None:
            self.seeds = pool
            return pool

        assert seeds_num > 0, "The seeds_num must be a positive integer."
        assert seeds_num <= len(pool), (
            "The number of seeds in the template pool is less than the number being "
            f"asked for ({seeds_num} > {len(pool)}). Use sample_population() if you need "
            "a population larger than the pool."
        )
        self.seeds = random.Random(seed).sample(pool, seeds_num)
        return self.seeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render(template: str, query: str, target_str: Optional[str] = None) -> str:
    """
    Substitute into a template's placeholders.

    Uses ``str.replace`` rather than ``str.format`` on purpose — see module docstring.

    Args:
        template:   Registry template.
        query:      Substituted for ``{query}``.
        target_str: Substituted for ``{target_str}`` when given. Only the
                    ``attack.TAP/PAIR`` ``original`` attacker prompts use it.
    """
    out = template.replace(PLACEHOLDER, query)
    if target_str is not None:
        out = out.replace(TARGET_PLACEHOLDER, target_str)
    return out


def sample_population(
    query: str,
    size: int,
    prompt_usage: str = "attack",
    method_list: Optional[list[str]] = None,
    variant: str = "ipi",
    extra_seeds: Optional[list[str]] = None,
    template_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Build a rendered population of exactly ``size`` seeds for GA / fuzzing.

    Unlike ``new_seeds``, ``size`` may exceed the pool. Distinct templates are used
    first; only once exhausted does it cycle (reshuffling each pass, so duplicates are
    spread out). A 64-member population from a 41-template pool therefore has 41
    distinct individuals rather than a handful repeated.
    """
    rng = random.Random(seed)
    pool = SeedTemplate().new_seeds(
        prompt_usage=prompt_usage,
        method_list=method_list or ["IPI"],
        variant=variant,
        template_file=template_file,
    )
    rendered = [render(t, query) for t in pool]
    rng.shuffle(rendered)

    population: list[str] = list(extra_seeds) if extra_seeds else []
    for s in rendered:
        if len(population) >= size:
            break
        population.append(s)
    while len(population) < size:
        rng.shuffle(rendered)
        for s in rendered:
            if len(population) >= size:
                break
            population.append(s)
    return population[:size]


def ica_demos(
    n: Optional[int] = None,
    variant: str = "ipi",
    template_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[tuple[str, str]]:
    """
    Return ``(goal, target)`` demonstration pairs for ICA.

    Args:
        n:       Number of pairs. None returns the whole pool.
        variant: ``ipi`` (default) — 10 authored demonstrations of an assistant
                 following an *injected* instruction, benign by construction.
                 ``original`` — the 30 verbatim AdvBench pairs the paper uses.
        seed:    RNG seed for reproducible sampling.

    Why ``ipi`` is the default: ICA works by showing an assistant that complies. The
    AdvBench pairs only restate the request ("Sure, here is a tutorial on …") and never
    actually carry out anything, and their subject matter is mismatched to IPI, where
    the barrier is instruction *provenance* rather than refusal. The IPI pool shows
    real, complete compliance with an injected instruction.
    """
    registry = load_seed_templates(template_file)
    raw = registry["demo"]["ICA"][variant]
    pairs = [(p["goal"], p["target"]) for p in raw]
    if n is None or n >= len(pairs):
        return pairs
    assert n > 0, "n must be a positive integer."
    return random.Random(seed).sample(pairs, n)
