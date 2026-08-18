"""
GPTFuzzer adapted for Indirect Prompt Injection.

Original paper: Yu et al. (2023), "GPTFUZZER: Red Teaming Large Language Models with
Auto-Generated Jailbreak Prompts"
arXiv: https://arxiv.org/abs/2309.10253
Source: https://github.com/sherdencooper/GPTFuzz

Mechanism
---------
A black-box fuzzing loop over a pool of seed templates:
  1. Select a seed with MCTS — walk down the mutation tree from a root, balancing a
     seed's past reward against how rarely it has been tried.
  2. Mutate it with one of five LLM operators — Expand, Shorten, Rephrase,
     GenerateSimilar, CrossOver — each keeping the ``{query}`` placeholder.
  3. Render the mutant with the goal, query the target, evaluate, add the mutant to the
     pool as a new node, and back-propagate its reward up the selected path.
Repeat until a query budget is hit; stop early on success.

This recipe is **pure wiring**: the pool is an ``AttackDataset`` of ``Instance``s, the
operators come from ``ipi.mutation``, selection from ``ipi.selector``, success from
``ipi.metrics``, and the seeds from ``ipi.seed``. There is no attack-local candidate type,
prompt text or selection rule left.

Fidelity
--------
  seeds:      the 77 GPTFuzzer initial seed templates, verbatim (``attack.Gptfuzzer.original``).
  mutators:   the five operators with their upstream prompts (``ipi/mutation/``).
  selection:  ``MCTSExploreSelectPolicy`` — the reference's own policy, formulas verbatim.
              (Until the carrier existed this recipe used flat UCB1, because MCTS needs
              per-node ``index``/``visited_num``/``level``/``children``.)
  reward:     binary, as upstream — its RoBERTa judge emits a jailbreak *label*, and
              ``Instance.num_jailbreak`` sums those labels. We substitute the dataset's own
              ground truth (``EvaluatorIPISuccess``), which is stronger than a classifier:
              dual-verifiable scenarios know exactly what success looks like.
  delivery:   the rendered template is injected through the data channel via
              ``harness.make_target_fn`` (untrusted content), not a user turn.
  success:    resolved from the scenario (``resolve_attack_target``) to match the dataset's
              own ``attack_eval_mode``.

IPI adaptations vs original
---------------------------
  reward source: upstream trains a RoBERTa classifier on jailbreak/no-jailbreak pairs.
      We have deterministic ground truth per scenario, so ``EvaluatorIPISuccess`` replaces
      it. A guidance evaluator (``judge=``) is now **optional and annotative** — it scores
      the trace and breaks ties for the reported best candidate, but no longer drives
      selection. Before the MCTS port it did (normalised to [0,1]), which was a workaround
      for not shipping the classifier, not a property of the algorithm.
  budget:  the reference fuzzes to a fixed iteration count over a whole question set; we
      budget by target queries per scenario, which is the scarce resource here.

Report of differences from the reference is in docs/easyjailbreak-audit.md.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from ..attacker import AdaptiveAttacker
from ..config import ATTACK_TEMP, ATTACK_TOP_P, ATTACK_MAX_TOKENS
from ..datasets import AttackDataset, Instance
from ..llm_unified import APILLM, UnifiedLLM
from ..metrics import Evaluator, EvaluatorIPISuccess
from ..mutation import GPTFUZZER_MUTATORS
from ..seed import SeedTemplate, render
from ..selector import MCTSExploreSelectPolicy
from ..victim import Victim

log = logging.getLogger(__name__)


def _load_seeds() -> list[str]:
    """The 77 verbatim GPTFuzzer seed templates from the registry."""
    return SeedTemplate().new_seeds(
        prompt_usage="attack", method_list=["Gptfuzzer"], variant="original",
    )


@dataclass
class GPTFuzzerResult:
    success: bool
    score: int
    injection: str          # best rendered injection
    target_response: str
    n_queries: int
    goal: str
    iterations: int = 0
    best_template: str = ""
    trace: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"GPTFuzzerResult(success={self.success}, score={self.score}, "
            f"n_queries={self.n_queries})"
        )


def _seed_pool(goal: str, templates: list[str], target_str: str,
               eval_mode: str) -> AttackDataset:
    """The initial population: one root ``Instance`` per seed template."""
    return AttackDataset([
        Instance(
            id=f"seed-{i}",
            query=goal,
            jailbreak_prompt=template,
            reference_responses=[target_str],
            attack_attrs={"target_str": target_str, "attack_eval_mode": eval_mode},
        )
        for i, template in enumerate(templates)
    ])


def run_gptfuzzer(
    goal: str,
    target_fn: Callable[[str], str],
    attacker_model: Union[str, UnifiedLLM],
    judge: Optional[Evaluator] = None,
    max_queries: int = 40,
    energy: int = 1,
    seed_templates: Optional[list[str]] = None,
    eval_mode: str = "contains",
    target_str: str = "",
    seed: Optional[int] = None,
) -> GPTFuzzerResult:
    """
    Run the GPTFuzzer fuzzing loop.

    Args:
        goal:           Attacker injection goal.
        target_fn:      Callable(injection: str) -> response: str.
        attacker_model: LLM (or model string) used for the mutation operators.
        judge:          Optional guidance Evaluator. Annotates the trace and breaks ties
                        for the reported best candidate; it does **not** drive selection —
                        see "IPI adaptations vs original".
        max_queries:    Target-query budget (the stopping criterion). Default 40.
        energy:         Mutants generated per selected seed each round. Default 1.
        seed_templates: Override the seed pool (default: 77 verbatim GPTFuzzer seeds).
        eval_mode:      check_ipi_success mode (usually resolved from the scenario).
        target_str:     Expected success string (falls back to goal).
        seed:           Optional RNG seed for reproducibility.

    Returns:
        GPTFuzzerResult.
    """
    rng = random.Random(seed)
    attacker = (
        APILLM(model=attacker_model, temperature=ATTACK_TEMP, top_p=ATTACK_TOP_P,
               max_tokens=ATTACK_MAX_TOKENS)
        if isinstance(attacker_model, str) else attacker_model
    )

    templates = seed_templates or _load_seeds()
    check_str = target_str or goal

    pool = _seed_pool(goal, templates, check_str, eval_mode)
    roots = AttackDataset(list(pool))          # snapshot: MCTS descends from these
    policy = MCTSExploreSelectPolicy(
        dataset=pool, initial_prompt_pool=roots, questions=1, seed=seed)
    # Operators rewrite the *template*, not the goal; CrossOver draws from the seed pool.
    mutators = GPTFUZZER_MUTATORS(attacker, seed_pool=templates,
                                  attr_name="jailbreak_prompt")
    evaluator = EvaluatorIPISuccess(mode=eval_mode)

    best = GPTFuzzerResult(
        success=False, score=0, injection="", target_response="",
        n_queries=0, goal=goal,
    )
    iteration = 0

    while best.n_queries < max_queries:
        iteration += 1
        parent = policy.select()[0]

        tried = AttackDataset([])
        for _ in range(energy):
            if best.n_queries >= max_queries:
                break

            mutator = rng.choice(mutators)
            # The dataset path records the parent/child edge and the level MCTS needs.
            child = mutator(AttackDataset([parent]))[0]
            # A mutant has not been tried yet; it inherits the parent's transcript
            # through copy(), which HistoricalInsight wants and we do not.
            child.target_responses = []
            child.eval_results = []
            policy.register(child)

            injection = render(child.jailbreak_prompt, goal)
            try:
                response = target_fn(injection)
            except Exception as exc:
                log.warning("[GPTFuzzer] target_fn raised: %s", exc)
                response = ""
            best.n_queries += 1
            child.target_responses = [response]
            tried.add(child)

            evaluator(AttackDataset([child]))
            success = bool(child.eval_results and child.eval_results[-1])

            score = 10 if success else 1
            if judge is not None:
                try:
                    score = judge.score(injection, response, attacker_goal=goal)
                except Exception as exc:
                    log.debug("[GPTFuzzer] judge.score failed: %s", exc)

            best.trace.append({
                "iteration": iteration, "template": child.jailbreak_prompt,
                "injection": injection, "response": response,
                "success": success, "score": score, "level": child.level,
            })

            if (success and not best.success) or (score > best.score) or not best.injection:
                best.success = best.success or success
                best.score = max(best.score, score)
                best.injection = injection
                best.target_response = response
                best.best_template = child.jailbreak_prompt

            if success:
                best.success = True
                break

        # Back-propagate up the selected path.
        policy.update(tried)

        best.iterations = iteration
        if best.success:
            break

    return best


class GPTFuzzerAttacker(AdaptiveAttacker):
    """
    GPTFuzzer attacker — budgeted seed-template fuzzing with MCTS selection (API-compatible).

    From: Yu et al. (2023), "GPTFUZZER: Red Teaming Large Language Models with
    Auto-Generated Jailbreak Prompts".

    Args:
        attacker_llm: LLM (or model string) driving the mutation operators.
        judge:        Optional guidance Evaluator. Annotates the trace; the search reward
                      is binary success against the scenario's ground truth.
        max_queries:  Target-query budget (stopping criterion). Default 40.
        energy:       Mutants per selected seed each round. Default 1.
        eval_mode:    check_ipi_success mode. Default None → auto-detect from the scenario's
                      ``attack_eval_mode``.
        seed:         Optional RNG seed for reproducibility.
    """

    _ATTACK_NAME = "gptfuzzer"

    def __init__(
        self,
        attacker_llm: Union[str, APILLM],
        judge: Optional[Evaluator] = None,
        max_queries: int = 40,
        energy: int = 1,
        eval_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(judge)
        self.attacker_llm = (
            APILLM(model=attacker_llm, temperature=ATTACK_TEMP, top_p=ATTACK_TOP_P,
                   max_tokens=ATTACK_MAX_TOKENS)
            if isinstance(attacker_llm, str) else attacker_llm
        )
        self.max_queries = max_queries
        self.energy      = energy
        self.eval_mode   = eval_mode
        self.seed        = seed

    @classmethod
    def requires_local_target(cls) -> bool:
        return False

    def run_scenario(self, target: Victim, instance: Instance, verbose: bool = False):
        from ..harness import make_target_fn
        from ..metrics import ScenarioResult, resolve_attack_target
        target_fn = make_target_fn(instance, target)
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)

        r = run_gptfuzzer(
            goal=instance.query,
            target_fn=target_fn,
            attacker_model=self.attacker_llm,
            judge=self.judge,
            max_queries=self.max_queries,
            energy=self.energy,
            eval_mode=eval_mode,
            target_str=target_str,
            seed=self.seed,
        )

        if verbose:
            log.info("[gptfuzzer] scenario=%s success=%s score=%d n_queries=%d",
                     instance.id, r.success, r.score, r.n_queries)

        return ScenarioResult(
            scenario_id=instance.id,
            goal=instance.query,
            success=r.success,
            score=r.score,
            injection=r.injection,
            target_response=r.target_response,
            n_queries=r.n_queries,
            attack=self._ATTACK_NAME,
            extra={"iterations": r.iterations, "best_template": r.best_template},
        )

    def __repr__(self) -> str:
        return (
            f"GPTFuzzerAttacker(attacker={self.attacker_llm.model_name!r}, "
            f"max_queries={self.max_queries}, energy={self.energy})"
        )
