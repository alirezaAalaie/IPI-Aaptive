"""
AutoDAN (Hierarchical Genetic Algorithm) adapted for Indirect Prompt Injection.

Original paper: Liu et al. (2023), "AutoDAN: Generating Stealthy Jailbreak Prompts on
Aligned Large Language Models"
arXiv: https://arxiv.org/abs/2310.04451

Algorithm: a genetic algorithm over readable natural-language injection templates.
  1. Seed a population from the template pool.
  2. Each generation:
     a. Score every candidate by cross-entropy loss(target_str | candidate).
        Lower loss = the victim is more likely to emit target_str.
     b. Keep the elites; draw the rest of the parents by softmax roulette on fitness.
     c. Recombine and mutate the parents.
     d. Every ``hga_period`` steps the HGA variant swaps the word-level operator in,
        driven by a momentum dictionary of which words appear in fit candidates.
  3. Stop on success, or when the budget runs out.

  run_autodan_ga   — pure GA (sentence crossover + mutation).
  run_autodan_hga  — HGA: alternates the GA step with word-level replacement.

Local-only: AutoDAN needs white-box access (forward passes for the loss). Use
``LocalLLM`` or any ``Victim`` with ``backend == "local"``.

Composition
-----------
The population is an ``AttackDataset`` of ``Instance``s — ``jailbreak_prompt`` is the
candidate template, ``eval_results[-1]`` its fitness. Every part comes from a family:

    fitness      ``selector.ReferenceLossSelector.score`` — one batched forward pass per
                 ``batch_size`` candidates, not one per candidate
    selection    ``selector.GeneticSelectPolicy`` — elites plus softmax roulette
    crossover    ``mutation.SentenceCrossOver``
    mutation     ``mutation.ReplaceWordsWithSynonyms`` (HGA's word-level operator), or
                 any ``llm_mutator`` callable
    seeds        ``seed.sample_population``
    success      ``metrics.check_ipi_success``

This recipe used to inline all five — its own crossover, synonym replacement, word
roulette, momentum dictionary and elite/roulette selection — duplicating operators that
already existed in ``ipi/mutation/`` and ``ipi/selector/``. Going through the families
also buys the batching: scoring was one forward pass *per candidate*, so a population of
64 cost 64 passes per generation.

IPI adaptations vs original
----------------------------
  target_str:
      Original: "Sure, here is a tutorial for making a bomb:" (a compliance prefix).
      IPI:      the expected agent action / tool call, e.g. "send_email(to='evil@x.com')".
                The CE loss over its tokens is the fitness.

  candidate population / seeds:
      Original: hand-crafted templates from ``assets/prompt_group.pth``.
      IPI:      the seed registry. ``seed_variant="ipi"`` → ``attack.AutoDAN.ipi``
                (41 IPI templates across 9 strategy families) plus an optional
                ``extra_seeds`` list; ``seed_variant="original"`` → the 128 upstream
                seeds verbatim, for the paper-reproduction row. No external .pth file,
                so the package stays self-contained.

  the prompt the loss is computed on:
      Original: FastChat's ``autodan_SuffixManager`` — one prompt used for both the loss
                and the generation, with ``[REPLACE]`` substituted by the instruction.
      IPI:      ``harness.build_optimization_messages`` — the victim's real prompt, IPI
                carrier and defense preprocessing included, so the GA ranks candidates by
                a loss on the string that is actually sent. It previously built its own
                bare ``[system][user]`` prompt *and* appended the goal a second time on
                top of the template's own ``{query}`` substitution.

  success evaluation:
      Original: a refusal-prefix keyword check.
      IPI:      ``check_ipi_success`` against the scenario's own target and eval mode.

  NLTK:
      The word-level operator needs nltk (stopwords + wordnet); it is imported lazily
      inside ``mutation.ReplaceWordsWithSynonyms`` and degrades to a pass-through if
      absent, so a missing corpus never kills a run mid-search.
      Install with: pip install nltk && python -m nltk.downloader stopwords wordnet punkt
"""

from __future__ import annotations

import gc
import logging
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..attacker import AdaptiveAttacker
from ..datasets import AttackDataset, Instance
from ..metrics import Evaluator, check_ipi_success
from ..mutation import ReplaceWordsWithSynonyms, SentenceCrossOver
from ..seed import sample_population
from ..selector import GeneticSelectPolicy, ReferenceLossSelector
from ..victim import Victim

log = logging.getLogger(__name__)

# Model/developer names AutoDAN's templates address; never swapped for a synonym.
_PROTECTED_WORDS = {
    "llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm",
    "mpt-chat", "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt",
    "modelkeeper", "prompt",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AutoDANResult:
    """Result of an AutoDAN GA / HGA run."""
    success: bool
    best_loss: float          # lowest cross-entropy loss reached
    injection: str            # the best candidate template, goal already substituted
    target_response: str      # the victim's response to it
    n_queries: int            # victim calls — one per generation, for the success check
    n_forward_passes: int     # scoring batches through the model — compute, not queries
    n_steps: int
    time_seconds: float
    goal: str
    target_str: str
    trace: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AutoDANResult(success={self.success}, best_loss={self.best_loss:.4f}, "
            f"n_steps={self.n_steps}, n_queries={self.n_queries})"
        )


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _make_population(
    goal: str,
    target_str: str,
    batch_size: int,
    extra_seeds: Optional[list[str]] = None,
    seed_variant: str = "ipi",
    seed: Optional[int] = None,
) -> AttackDataset:
    """
    Generation 0: one ``Instance`` per seed template.

    ``jailbreak_prompt`` is the template *with the goal already substituted*, which is
    what ``sample_population`` returns. ``reference_responses[0]`` is what
    ``ReferenceLossSelector`` scores against — the CE-loss target.
    """
    templates = sample_population(
        goal, batch_size, prompt_usage="attack", method_list=["AutoDAN"],
        variant=seed_variant, extra_seeds=extra_seeds, seed=seed)
    return AttackDataset([
        Instance(
            id=f"autodan-{i}",
            query=goal,
            jailbreak_prompt=template,
            reference_responses=[target_str],
        )
        for i, template in enumerate(templates)
    ])


def _score(loss_selector: ReferenceLossSelector, population: AttackDataset) -> None:
    """
    Write each candidate's fitness into ``eval_results``.

    ``ReferenceLossSelector`` writes ``_loss`` (lower is better); ``GeneticSelectPolicy``
    reads ``eval_results[-1]`` as higher-is-better. The negation is upstream's own
    ``score_list = [-x for x in score_list]``, and getting its sign wrong inverts the
    search silently — which is why it happens in exactly one place.
    """
    loss_selector.score(population)
    for instance in population:
        instance.eval_results = [-float(instance._loss)]


# ---------------------------------------------------------------------------
# HGA momentum word dictionary
# ---------------------------------------------------------------------------

def build_momentum_word_dict(
    word_dict: dict[str, float],
    population: AttackDataset,
    topk: int = -1,
) -> dict[str, float]:
    """
    Accumulate a word -> average-fitness map across generations.

    Port of upstream's ``construct_momentum_word_dict``: a word's score is the mean
    fitness of the candidates containing it, blended half-and-half with whatever the
    word already carried, so evidence persists across generations ("momentum"). It is
    what ``mutation.ReplaceWordsWithSynonyms`` draws its replacement probabilities from.

    Needs nltk for tokenisation and stopwords; without it the dictionary stays as it was
    and the operator degrades to a pass-through, which is the same failure mode the
    operator itself has.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except (ImportError, LookupError) as exc:
        from ..mutation.rule import _warn_nltk_missing
        _warn_nltk_missing("AutoDAN momentum dictionary", exc)
        return word_dict

    word_scores: dict[str, list[float]] = defaultdict(list)
    for instance in population:
        score = float(instance.eval_results[-1]) if instance.eval_results else 0.0
        if not np.isfinite(score):
            continue
        try:
            tokens = nltk.word_tokenize(instance.jailbreak_prompt or "")
        except LookupError:
            return word_dict
        for word in {
            w for w in tokens
            if w.lower() not in stop_words and w.lower() not in _PROTECTED_WORDS
        }:
            word_scores[word].append(score)

    for word, scores in word_scores.items():
        avg = sum(scores) / len(scores)
        word_dict[word] = (word_dict[word] + avg) / 2 if word in word_dict else avg

    ordered = OrderedDict(sorted(word_dict.items(), key=lambda kv: kv[1], reverse=True))
    return dict(list(ordered.items())[:topk] if topk != -1 else ordered.items())


# ---------------------------------------------------------------------------
# One generation
# ---------------------------------------------------------------------------

def _apply_llm_mutator(
    parents: list[Instance],
    mutation_rate: float,
    llm_mutator: Callable[[str], str],
    rng: random.Random,
) -> list[Instance]:
    """
    Rewrite a parent's template with probability ``mutation_rate``.

    The optional LLM-driven branch of upstream's ``apply_gpt_mutation``. Without a
    mutator the word-level operator does the job instead; the two are alternatives, as
    upstream's ``if_api`` switch makes them.
    """
    for parent in parents:
        if rng.random() >= mutation_rate:
            continue
        try:
            mutated = (llm_mutator(parent.jailbreak_prompt) or "").strip()
        except Exception as exc:
            log.debug("[AutoDAN] llm_mutator raised: %s", exc)
            continue
        if mutated:
            parent.jailbreak_prompt = mutated
    return parents


def _ga_generation(
    population: AttackDataset,
    policy: GeneticSelectPolicy,
    crossover: SentenceCrossOver,
    crossover_prob: float,
    mutation_rate: float,
    llm_mutator: Optional[Callable[[str], str]],
    rng: random.Random,
    batch_size: int,
) -> AttackDataset:
    """Elites + roulette parents, recombined pairwise, then mutated."""
    elites = policy.elites(population)
    parents = policy.roulette(population, n=batch_size - len(elites))

    offspring: list[Instance] = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
        if rng.random() < crossover_prob:
            # 1-to-2: the operator records both lineage edges for us.
            offspring.extend(crossover(AttackDataset([p1]), other_instance=p2))
        else:
            offspring.extend([crossover.new_child(p1), crossover.new_child(p2)])

    offspring = offspring[: batch_size - len(elites)]
    if llm_mutator is not None:
        offspring = _apply_llm_mutator(offspring, mutation_rate, llm_mutator, rng)
    return AttackDataset(elites + offspring)


def _hga_generation(
    population: AttackDataset,
    policy: GeneticSelectPolicy,
    synonym: ReplaceWordsWithSynonyms,
    word_dict: dict[str, float],
    mutation_rate: float,
    llm_mutator: Optional[Callable[[str], str]],
    rng: random.Random,
    batch_size: int,
) -> tuple[AttackDataset, dict[str, float]]:
    """
    The word-level half of HGA: refresh the momentum dictionary, then swap synonyms.

    Upstream sorts the population and takes everything past the elites as parents rather
    than drawing a roulette — the word roulette *inside* the operator is where the
    fitness weighting happens at this step.
    """
    word_dict = build_momentum_word_dict(word_dict, population)
    synonym.update(word_dict)

    ranked = sorted(population, key=policy._fitness, reverse=True)
    elites, parents = ranked[: policy.num_elites], ranked[policy.num_elites:]
    parents = [p for p in parents if (p.jailbreak_prompt or "").strip()]
    if len(parents) < batch_size - len(elites) and parents:
        parents += rng.choices(parents, k=batch_size - len(elites) - len(parents))

    # The operator takes no per-call rate: its word roulette already weights every
    # swap by the momentum score, which is where upstream's `crossover` knob acts.
    offspring = list(synonym(AttackDataset(parents)))
    if llm_mutator is not None:
        offspring = _apply_llm_mutator(offspring, mutation_rate, llm_mutator, rng)
    offspring = offspring[: batch_size - len(elites)]
    return AttackDataset(elites + offspring), word_dict


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def _run_autodan(
    goal: str,
    target_str: str,
    target_llm: Victim,
    *,
    variant: str,
    eval_target_fn: Optional[Callable[[str], str]],
    eval_mode: str,
    num_steps: int,
    batch_size: int,
    num_elites_frac: float,
    crossover_prob: float,
    num_points: int,
    mutation_rate: float,
    hga_period: int,
    extra_seeds: Optional[list[str]],
    seed_variant: str,
    llm_mutator: Optional[Callable[[str], str]],
    prompt_builder: Optional[Callable[[str], list[dict]]],
    score_batch_size: Optional[int],
    budget_seconds: Optional[float],
    seed: int,
    verbose: bool,
) -> AutoDANResult:
    """GA and HGA share every line except which operator drives a given generation."""
    if target_llm.backend != "local":
        raise ValueError(
            "AutoDAN requires backend='local' (white-box model access for the loss). "
            f"Current backend: {target_llm.backend!r}.")

    random.seed(seed)
    np.random.seed(seed)

    rng = random.Random(seed)
    num_elites = max(1, int(batch_size * num_elites_frac))

    population = _make_population(
        goal, target_str, batch_size, extra_seeds, seed_variant, seed=seed)
    loss_selector = ReferenceLossSelector(
        target_llm, batch_size=score_batch_size, prompt_builder=prompt_builder)
    policy = GeneticSelectPolicy(num_elites=num_elites, seed=seed)
    crossover = SentenceCrossOver(num_points=num_points, seed=seed)
    synonym = ReplaceWordsWithSynonyms(seed=seed)

    start_time = time.time()
    best_loss = float("inf")
    best_injection = population[0].jailbreak_prompt
    best_response = ""
    n_queries = 0
    n_forward = 0
    word_dict: dict[str, float] = {}
    trace: list[dict] = []
    step = 0

    for step in range(num_steps):
        if budget_seconds is not None and (time.time() - start_time) > budget_seconds:
            log.info("[AutoDAN-%s] budget exhausted at step %d.", variant.upper(), step)
            break

        # ---- Fitness: one batched forward pass per score_batch_size candidates ----
        _score(loss_selector, population)
        size = score_batch_size or len(population) or 1
        n_forward += (len(population) + size - 1) // size

        fittest = max(population, key=policy._fitness)
        step_loss = -policy._fitness(fittest)
        if step_loss < best_loss:
            best_loss = step_loss
            best_injection = fittest.jailbreak_prompt

        # ---- Did it work? One victim call per generation, on the fittest candidate ----
        injection = fittest.jailbreak_prompt
        response = (
            eval_target_fn(injection) if eval_target_fn is not None
            else target_llm(injection))
        n_queries += 1
        best_response = response
        success = check_ipi_success(response, target_str, eval_mode)

        if verbose:
            log.info("[AutoDAN-%s] step=%3d/%d  best_loss=%.4f  success=%s  t=%.1fs",
                     variant.upper(), step + 1, num_steps, step_loss, success,
                     time.time() - start_time)

        trace.append({
            "step": step + 1, "loss": step_loss, "best_loss": best_loss,
            "injection": injection[:80], "success": success,
            "level": fittest.level, "word_dict_size": len(word_dict),
        })

        if success:
            log.info("[AutoDAN-%s] success at step %d.", variant.upper(), step + 1)
            return AutoDANResult(
                success=True, best_loss=best_loss, injection=injection,
                target_response=response, n_queries=n_queries,
                n_forward_passes=n_forward, n_steps=step + 1,
                time_seconds=time.time() - start_time, goal=goal,
                target_str=target_str, trace=trace)

        # ---- Breed the next generation ----
        # HGA alternates: a full GA step every `hga_period` generations, the word-level
        # operator otherwise. Same polarity as upstream's `if j % args.iter == 0`.
        if variant == "ga" or step % hga_period == 0:
            population = _ga_generation(
                population, policy, crossover, crossover_prob, mutation_rate,
                llm_mutator, rng, batch_size)
        else:
            population, word_dict = _hga_generation(
                population, policy, synonym, word_dict, mutation_rate,
                llm_mutator, rng, batch_size)

        gc.collect()

    return AutoDANResult(
        success=check_ipi_success(best_response, target_str, eval_mode),
        best_loss=best_loss, injection=best_injection,
        target_response=best_response, n_queries=n_queries,
        n_forward_passes=n_forward, n_steps=step + 1,
        time_seconds=time.time() - start_time, goal=goal,
        target_str=target_str, trace=trace)


_COMMON_DOC = """
    Args:
        goal:              Attacker objective — substituted into each seed template.
        target_str:        The desired agent output. Its tokens are the CE-loss target
                           and the success check runs against it.
        target_llm:        Must have ``backend == "local"`` — the loss needs the weights.
        eval_target_fn:    ``callable(injection) -> response`` used for the success
                           check. ``AutoDANAttacker`` passes ``harness.make_target_fn``;
                           without one the victim is called directly, which skips the
                           IPI carrier.
        eval_mode:         ``check_ipi_success`` mode. Normally resolved from the
                           instance by ``AutoDANAttacker``, not chosen here.
        num_steps:         Generations. Default 100.
        batch_size:        Population size. Default 64 (upstream uses 256).
        num_elites_frac:   Fraction carried over unchanged. Default 0.05.
        crossover_prob:    Probability of recombining a parent pair. Default 0.5.
        num_points:        Crossover points per candidate. Default 5.
        mutation_rate:     Probability the LLM mutator rewrites an offspring. Default 0.01.
        extra_seeds:       Extra templates to seed generation 0 with.
        seed_variant:      "ipi" (41 IPI templates, default) or "original" (the 128
                           upstream AutoDAN seeds, for the paper row).
        llm_mutator:       ``callable(str) -> str``. Upstream's ``if_api`` branch. With
                           None the word-level operator carries the mutation instead.
        prompt_builder:    ``callable(injection) -> messages`` — the prompt the fitness
                           is computed on. Pass ``harness.build_optimization_messages``
                           bound to the instance, as ``run_scenario`` does; without it
                           the loss is computed on a bare ``[system][user]`` pair that
                           the victim is never given.
        score_batch_size:  Candidates per scoring forward pass. None means the whole
                           population in one batch, which OOMs on a large one.
        budget_seconds:    Wall-clock budget. None = unlimited.
        seed:              RNG seed.
        verbose:           Log each generation.

    Returns:
        AutoDANResult.
"""


def run_autodan_ga(
    goal: str,
    target_str: str,
    target_llm: Victim,
    eval_target_fn: Optional[Callable[[str], str]] = None,
    eval_mode: str = "contains",
    num_steps: int = 100,
    batch_size: int = 64,
    num_elites_frac: float = 0.05,
    crossover_prob: float = 0.5,
    num_points: int = 5,
    mutation_rate: float = 0.01,
    extra_seeds: Optional[list[str]] = None,
    seed_variant: str = "ipi",
    llm_mutator: Optional[Callable[[str], str]] = None,
    prompt_builder: Optional[Callable[[str], list[dict]]] = None,
    score_batch_size: Optional[int] = 32,
    budget_seconds: Optional[float] = None,
    seed: int = 20,
    verbose: bool = False,
) -> AutoDANResult:
    """
    AutoDAN-GA — sentence crossover and mutation, no word-level operator.
    """
    return _run_autodan(
        goal, target_str, target_llm, variant="ga",
        eval_target_fn=eval_target_fn, eval_mode=eval_mode, num_steps=num_steps,
        batch_size=batch_size, num_elites_frac=num_elites_frac,
        crossover_prob=crossover_prob, num_points=num_points,
        mutation_rate=mutation_rate, hga_period=1, extra_seeds=extra_seeds,
        seed_variant=seed_variant, llm_mutator=llm_mutator,
        prompt_builder=prompt_builder, score_batch_size=score_batch_size,
        budget_seconds=budget_seconds, seed=seed, verbose=verbose)


def run_autodan_hga(
    goal: str,
    target_str: str,
    target_llm: Victim,
    eval_target_fn: Optional[Callable[[str], str]] = None,
    eval_mode: str = "contains",
    num_steps: int = 100,
    batch_size: int = 64,
    num_elites_frac: float = 0.05,
    crossover_prob: float = 0.5,
    num_points: int = 5,
    mutation_rate: float = 0.01,
    hga_period: int = 5,
    extra_seeds: Optional[list[str]] = None,
    seed_variant: str = "ipi",
    llm_mutator: Optional[Callable[[str], str]] = None,
    prompt_builder: Optional[Callable[[str], list[dict]]] = None,
    score_batch_size: Optional[int] = 32,
    budget_seconds: Optional[float] = None,
    seed: int = 20,
    verbose: bool = False,
) -> AutoDANResult:
    """
    AutoDAN-HGA — a GA step every ``hga_period`` generations, word-level replacement
    driven by the momentum dictionary in between.
    """
    return _run_autodan(
        goal, target_str, target_llm, variant="hga",
        eval_target_fn=eval_target_fn, eval_mode=eval_mode, num_steps=num_steps,
        batch_size=batch_size, num_elites_frac=num_elites_frac,
        crossover_prob=crossover_prob, num_points=num_points,
        mutation_rate=mutation_rate, hga_period=hga_period, extra_seeds=extra_seeds,
        seed_variant=seed_variant, llm_mutator=llm_mutator,
        prompt_builder=prompt_builder, score_batch_size=score_batch_size,
        budget_seconds=budget_seconds, seed=seed, verbose=verbose)


run_autodan_ga.__doc__ += _COMMON_DOC
run_autodan_hga.__doc__ += _COMMON_DOC
run_autodan_hga.__doc__ += """
        hga_period:        Generations between full GA steps. Default 5.
"""


# ---------------------------------------------------------------------------
# AutoDANAttacker
# ---------------------------------------------------------------------------

class AutoDANAttacker(AdaptiveAttacker):
    """
    AutoDAN attacker — a genetic search over readable injection templates.

    Requires a local target: fitness is a forward pass through the victim's own weights.

    Args:
        judge:            Guidance Evaluator (owned by this attacker). Annotative —
                          success comes from ``check_ipi_success``.
        variant:          "hga" (default) or "ga".
        num_steps:        Generations. Default 100.
        batch_size:       Population size. Default 64.
        num_elites_frac:  Fraction kept as elites. Default 0.05.
        crossover_prob:   Probability of recombining a parent pair. Default 0.5.
        num_points:       Crossover points. Default 5.
        mutation_rate:    Probability the LLM mutator rewrites an offspring. Default 0.01.
        hga_period:       Generations between GA steps in the HGA variant. Default 5.
        eval_mode:        IPI success check mode. Default None → read the instance's own
                          ``attack_eval_mode``.
        llm_mutator:      ``callable(str) -> str`` for LLM-driven mutation.
        extra_seeds:      Extra templates for generation 0.
        seed_variant:     "ipi" (default) or "original".
        score_batch_size: Candidates per scoring forward pass. Default 32.
        budget_seconds:   Wall-clock budget per scenario.
        seed:             RNG seed. Default 20.
    """

    def __init__(
        self,
        judge: Optional[Evaluator] = None,
        variant: str = "hga",
        num_steps: int = 100,
        batch_size: int = 64,
        num_elites_frac: float = 0.05,
        crossover_prob: float = 0.5,
        num_points: int = 5,
        mutation_rate: float = 0.01,
        hga_period: int = 5,
        eval_mode: Optional[str] = None,
        llm_mutator: Optional[Callable[[str], str]] = None,
        extra_seeds: Optional[list[str]] = None,
        seed_variant: str = "ipi",
        score_batch_size: Optional[int] = 32,
        budget_seconds: Optional[float] = None,
        seed: int = 20,
    ):
        super().__init__(judge)
        if variant not in ("ga", "hga"):
            raise ValueError(f"variant must be 'ga' or 'hga', got {variant!r}")
        if seed_variant not in ("ipi", "original"):
            raise ValueError(
                f"seed_variant must be 'ipi' or 'original', got {seed_variant!r}")
        self.variant          = variant
        self.num_steps        = num_steps
        self.batch_size       = batch_size
        self.num_elites_frac  = num_elites_frac
        self.crossover_prob   = crossover_prob
        self.num_points       = num_points
        self.mutation_rate    = mutation_rate
        self.hga_period       = hga_period
        self.eval_mode        = eval_mode
        self.llm_mutator      = llm_mutator
        self.extra_seeds      = extra_seeds
        self.seed_variant     = seed_variant
        self.score_batch_size = score_batch_size
        self.budget_seconds   = budget_seconds
        self.seed             = seed

    @classmethod
    def requires_local_target(cls) -> bool:
        return True

    def run_scenario(self, target: Victim, instance: Instance, verbose: bool = False):
        from ..harness import build_optimization_messages, make_target_fn
        from ..metrics import ScenarioResult, resolve_attack_target
        target_fn = make_target_fn(instance, target)
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)

        # The fitness must be computed on the same prompt the success check sends: the
        # IPI carrier, plus whatever the defense does to it.
        def prompt_builder(injection: str) -> list[dict]:
            return build_optimization_messages(instance, target, injection)

        runner = run_autodan_ga if self.variant == "ga" else run_autodan_hga
        kwargs = dict(
            goal=instance.query,
            target_str=target_str,
            target_llm=target,
            eval_target_fn=target_fn,
            eval_mode=eval_mode,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            num_elites_frac=self.num_elites_frac,
            crossover_prob=self.crossover_prob,
            num_points=self.num_points,
            mutation_rate=self.mutation_rate,
            extra_seeds=self.extra_seeds,
            seed_variant=self.seed_variant,
            llm_mutator=self.llm_mutator,
            prompt_builder=prompt_builder,
            score_batch_size=self.score_batch_size,
            budget_seconds=self.budget_seconds,
            seed=self.seed,
            verbose=verbose,
        )
        if self.variant == "hga":
            kwargs["hga_period"] = self.hga_period
        r = runner(**kwargs)

        return ScenarioResult(
            scenario_id=instance.id,
            goal=instance.query,
            success=r.success,
            score=max(0, int(10 - r.best_loss)),   # rough 0-10 scale from the loss
            injection=r.injection,
            target_response=r.target_response,
            n_queries=r.n_queries,
            attack=f"autodan_{self.variant}",
            extra={
                "best_loss": r.best_loss,
                "n_steps": r.n_steps,
                "n_forward_passes": r.n_forward_passes,
                "time_seconds": r.time_seconds,
            },
        )

    def __repr__(self) -> str:
        return (
            f"AutoDANAttacker(variant={self.variant!r}, "
            f"num_steps={self.num_steps}, batch_size={self.batch_size})"
        )
