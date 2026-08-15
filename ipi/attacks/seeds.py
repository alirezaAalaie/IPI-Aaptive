"""
Seed template registry — the EasyJailbreak ``SeedBase`` / ``SeedTemplate`` pattern.

Why this exists
---------------
Population-based attacks (GPTFuzzer, AutoDAN) and template attacks (ICA,
DeepInception, ReNeLLM) all need *seed prompts*. Before this module each attack
carried its own hard-coded list, which meant (a) no single place to audit what a
run actually used, (b) no way to swap in a different pool without editing source,
and (c) silent drift from the published templates.

EasyJailbreak solves this with a small registry: templates live in a JSON file
keyed by ``prompt_usage -> method -> [templates]``, and ``SeedTemplate.new_seeds()``
pulls a pool (or a random sample of it) by method name. We adopt that design.

Layout of ``seed_templates.json``
---------------------------------
    original/   verbatim copies from the reference implementations
                (Gptfuzzer: 77 seeds · ICA · DeepInception · ReNeLLM: 3 scenarios)
    ipi/        this repo's IPI-reframed pool (41 templates, 9 strategy families)
    advbench/   30 (goal, target) pairs sampled from AdvBench, used to build ICA
                in-context demonstrations the way the original paper does

All templates use ``{query}`` as the placeholder, matching the upstream convention.

Deviation from EasyJailbreak
----------------------------
``SeedTemplate.new_seeds()`` upstream *asserts* ``seeds_num <= len(pool)`` and
samples without replacement. We keep that behaviour for parity, but add
``sample_population()`` for the GA/fuzzing case where you need a population larger
than the pool — it fills with distinct templates first and only then cycles, so a
64-member population from a 41-template pool has 41 distinct individuals rather
than a handful repeated (this was a real bug in the previous AutoDAN seeding).
"""

from __future__ import annotations

import json
import random
from typing import Iterator, Optional

__all__ = [
    "SeedBase", "SeedTemplate", "PLACEHOLDER",
    "load_seed_templates", "render", "sample_population", "advbench_pairs",
]

PLACEHOLDER = "{query}"

_TEMPLATE_FILE = "seed_templates.json"
_cache: Optional[dict] = None


def load_seed_templates(template_file: Optional[str] = None) -> dict:
    """
    Load the seed-template JSON. Cached when using the packaged default.

    Args:
        template_file: Path to an external JSON file with the same schema.
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
        except Exception:  # pragma: no cover - fallback for odd install layouts
            import os
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, _TEMPLATE_FILE), "r", encoding="utf-8") as f:
                raw = f.read()
        _cache = json.loads(raw)
    return _cache


class SeedBase:
    """
    A base class that can store and generate attack seeds.

    Mirrors ``easyjailbreak.seed.SeedBase``.
    """

    def __init__(self, seeds: Optional[list[str]] = None):
        self.seeds: list[str] = list(seeds) if seeds else []

    def __iter__(self) -> Iterator[str]:
        return iter(self.seeds)

    def __len__(self) -> int:
        return len(self.seeds)

    def new_seeds(self, **kwargs):
        raise NotImplementedError


class SeedTemplate(SeedBase):
    """
    Generates seeds from the template registry.

    Example:
        >>> SeedTemplate().new_seeds(method_list=["DeepInception"])          # whole pool
        >>> SeedTemplate().new_seeds(seeds_num=5, method_list=["Gptfuzzer"]) # random 5
        >>> SeedTemplate().new_seeds(method_list=["IPI"], prompt_usage="ipi")
    """

    def __init__(self, seeds: Optional[list[str]] = None):
        super().__init__(seeds)

    def new_seeds(
        self,
        seeds_num: Optional[int] = None,
        prompt_usage: str = "original",
        method_list: Optional[list[str]] = None,
        template_file: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        """
        Pull templates from the registry.

        Args:
            seeds_num:    How many seeds to return. None returns the whole pool.
            prompt_usage: Top-level section — "original" (verbatim upstream) or "ipi".
            method_list:  Method names to pull, e.g. ["Gptfuzzer"], ["ReNeLLM"].
            template_file: Optional external JSON file (same schema).
            seed:         Optional RNG seed for reproducible sampling.

        Returns:
            List of template strings.

        Raises:
            AttributeError: if a requested method is absent (matches upstream).
            AssertionError:  if seeds_num exceeds the pool size (matches upstream).
        """
        self.seeds = []
        if method_list is None:
            method_list = ["IPI"]

        registry = load_seed_templates(template_file)
        section = registry.get(prompt_usage, {})

        pool: list[str] = []
        for method in method_list:
            if method not in section:
                raise AttributeError(
                    f"seed templates contain no {prompt_usage!r} pool for method "
                    f"{method!r}. Available: {sorted(k for k in section if not k.startswith('_'))}"
                )
            pool.extend(section[method])

        if seeds_num is None:
            self.seeds = pool
            return pool

        assert seeds_num > 0, "The seeds_num must be a positive integer."
        assert seeds_num <= len(pool), (
            "The number of seeds in the template pool is less than the number "
            f"being asked for ({seeds_num} > {len(pool)}). Use sample_population() "
            "if you need a population larger than the pool."
        )
        rng = random.Random(seed)
        self.seeds = rng.sample(pool, seeds_num)
        return self.seeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render(template: str, query: str) -> str:
    """
    Substitute ``query`` into a template's placeholder.

    Uses ``str.replace`` rather than ``str.format`` on purpose: many upstream
    templates contain literal braces (LaTeX in ReNeLLM, JSON in tool-output
    framings) that would make ``format`` raise.
    """
    return template.replace(PLACEHOLDER, query)


def sample_population(
    query: str,
    size: int,
    prompt_usage: str = "ipi",
    method_list: Optional[list[str]] = None,
    extra_seeds: Optional[list[str]] = None,
    template_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Build a rendered population of exactly ``size`` seeds for GA / fuzzing.

    Unlike ``new_seeds``, this permits ``size`` larger than the pool. Distinct
    templates are used first; only once they are exhausted does it cycle (with a
    reshuffle each pass, so duplicates are spread out rather than blocked).

    Args:
        query:       Goal string substituted into each template's placeholder.
        size:        Desired population size.
        prompt_usage/method_list/template_file: Passed to the registry.
        extra_seeds: Concrete seeds used verbatim and placed first.
        seed:        Optional RNG seed.
    """
    rng = random.Random(seed)
    pool = SeedTemplate().new_seeds(
        prompt_usage=prompt_usage,
        method_list=method_list or ["IPI"],
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


def advbench_pairs(
    n: Optional[int] = None,
    template_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[tuple[str, str]]:
    """
    Return (goal, target) pairs from the packaged AdvBench sample.

    These are the raw material for ICA's in-context demonstrations, matching the
    original paper's construction: each demo is a harmful request paired with an
    affirmative compliance prefix.

    Args:
        n:    Number of pairs. None returns all packaged pairs.
        seed: Optional RNG seed for reproducible sampling.
    """
    registry = load_seed_templates(template_file)
    pairs = [(p["goal"], p["target"]) for p in registry["advbench"]["pairs"]]
    if n is None:
        return pairs
    assert n > 0, "n must be a positive integer."
    if n >= len(pairs):
        return pairs
    return random.Random(seed).sample(pairs, n)
