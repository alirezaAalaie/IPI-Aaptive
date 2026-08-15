"""
GPTFuzzer adapted for Indirect Prompt Injection.

Original paper: Yu et al. (2023), "GPTFUZZER: Red Teaming Large Language Models with
Auto-Generated Jailbreak Prompts"
arXiv: https://arxiv.org/abs/2309.10253
Source: https://github.com/sherdencooper/GPTFuzz

Mechanism
---------
A black-box fuzzing loop over a pool of seed templates:
  1. Select a seed from the pool (tree selection — balance exploiting good seeds against
     exploring rarely-tried ones).
  2. Mutate it with one of five LLM operators — Expand, Shorten, Rephrase,
     GenerateSimilar, CrossOver — shared verbatim-prompt objects in
     ``mutations.GPTFUZZER_MUTATORS``, each keeping the ``{query}`` placeholder.
  3. Render the mutant with the goal, query the target, score the response, add the
     mutant back to the pool, and back-propagate its reward.
Repeat until a query / success budget is hit; stop early on success.

Fidelity
--------
  seeds:      the 77 GPTFuzzer initial seed templates, loaded verbatim from the seed
              registry (``original/Gptfuzzer``).
  mutators:   the five operators with their upstream prompts (``mutations.py``).
  selection:  the reference uses an MCTS-explore policy; we use UCB1 over the seed nodes,
              which is the same exploit/explore idea with a lighter bookkeeping. (Noted as
              a deviation in docs/easyjailbreak-audit.md.)
  reward:     the reference scores with a fine-tuned RoBERTa judge. We use a Judge score
              in [1,10] when provided (normalized), else binary ``check_ipi_success``.
  delivery:   the rendered template is injected through the data channel via
              ``make_scenario_target_fn`` (untrusted content), not a user turn.
  success:    resolved from the scenario (``resolve_attack_target``) to match the dataset's
              own ``attack_eval_mode``.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from ..attacker import AdaptiveAttacker
from ..config import ATTACK_TEMP, ATTACK_TOP_P, ATTACK_MAX_TOKENS
from ..evaluator import check_ipi_success
from ..judges import Judge
from ..llm_unified import APILLM, UnifiedLLM
from ..victim import Victim
from .mutations import GPTFUZZER_MUTATORS
from .seeds import SeedTemplate, render


log = logging.getLogger(__name__)


def _load_seeds() -> list[str]:
    """The 77 verbatim GPTFuzzer seed templates from the registry."""
    return SeedTemplate().new_seeds(method_list=["Gptfuzzer"], prompt_usage="original")


# ---------------------------------------------------------------------------
# Seed node (a template + its UCB statistics)
# ---------------------------------------------------------------------------

@dataclass
class _SeedNode:
    template: str
    visits: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0


def _ucb_select(nodes: list[_SeedNode], total_visits: int, rng: random.Random,
                c: float = 1.41) -> _SeedNode:
    """UCB1 selection. Unvisited nodes are picked first (ties broken randomly)."""
    unvisited = [n for n in nodes if n.visits == 0]
    if unvisited:
        return rng.choice(unvisited)
    log_total = math.log(total_visits + 1)
    best_node, best_val = nodes[0], -math.inf
    for node in nodes:
        ucb = node.mean_reward + c * math.sqrt(log_total / node.visits)
        if ucb > best_val:
            best_val, best_node = ucb, node
    return best_node


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


def run_gptfuzzer(
    goal: str,
    target_fn: Callable[[str], str],
    attacker_model: Union[str, UnifiedLLM],
    judge: Optional[Judge] = None,
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
        attacker_model: LLM (or model string) used for mutation operators.
        judge:          Optional Judge; its score drives the reward. Without it the reward
                        is binary success.
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
    llm_call = lambda text: attacker(text)  # noqa: E731  (str) -> str

    templates = seed_templates or _load_seeds()
    mutators = GPTFUZZER_MUTATORS(seed_pool=templates)   # CrossOver draws from the seed pool
    pool = [_SeedNode(template=t) for t in templates]
    check_str = target_str or goal

    best = GPTFuzzerResult(
        success=False, score=0, injection="", target_response="",
        n_queries=0, goal=goal,
    )
    total_visits = 0
    iteration = 0

    while best.n_queries < max_queries:
        iteration += 1
        parent = _ucb_select(pool, total_visits, rng)

        round_reward = 0.0
        for _ in range(energy):
            if best.n_queries >= max_queries:
                break
            mutator = rng.choice(mutators)
            mutant_tmpl = mutator(llm_call, parent.template)
            injection = render(mutant_tmpl, goal)

            try:
                response = target_fn(injection)
            except Exception as exc:
                log.warning("[GPTFuzzer] target_fn raised: %s", exc)
                response = ""
            best.n_queries += 1

            success = check_ipi_success(response, check_str, eval_mode)
            if judge is not None:
                try:
                    score = judge.score(injection, response, attacker_goal=goal)
                except Exception as exc:
                    log.debug("[GPTFuzzer] judge.score failed: %s", exc)
                    score = 10 if success else 1
                reward = score / 10.0
            else:
                score = 10 if success else 1
                reward = 1.0 if success else 0.0

            round_reward = max(round_reward, reward)

            # Add the mutant to the pool as a new explorable seed.
            pool.append(_SeedNode(template=mutant_tmpl, visits=1, reward_sum=reward))

            best.trace.append({
                "iteration": iteration, "template": mutant_tmpl,
                "injection": injection, "response": response,
                "success": success, "score": score,
            })

            if (success and not best.success) or (score > best.score) or not best.injection:
                best.success = best.success or success
                best.score = max(best.score, score)
                best.injection = injection
                best.target_response = response
                best.best_template = mutant_tmpl

            if success:
                best.success = True
                break

        # Back-propagate to the selected parent.
        parent.visits += 1
        parent.reward_sum += round_reward
        total_visits += 1

        best.iterations = iteration
        if best.success:
            break

    return best


class GPTFuzzerAttacker(AdaptiveAttacker):
    """
    GPTFuzzer attacker — budgeted seed-template fuzzing with UCB selection (API-compatible).

    From: Yu et al. (2023), "GPTFUZZER: Red Teaming Large Language Models with
    Auto-Generated Jailbreak Prompts".

    Args:
        attacker_llm: LLM (or model string) driving the mutation operators.
        judge:        Optional Judge; its score drives the reward signal. Without it the
                      reward is binary (success / failure).
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
        judge: Optional[Judge] = None,
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

    def run_scenario(self, target: Victim, scenario, verbose: bool = False):
        from ..evaluator import ScenarioResult, make_scenario_target_fn, resolve_attack_target
        target_fn = make_scenario_target_fn(scenario, target)
        target_str, eval_mode = resolve_attack_target(scenario, self.eval_mode)

        r = run_gptfuzzer(
            goal=scenario.injection_goal,
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
                     scenario.id, r.success, r.score, r.n_queries)

        return ScenarioResult(
            scenario_id=scenario.id,
            goal=scenario.injection_goal,
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
