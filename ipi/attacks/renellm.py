"""
ReNeLLM adapted for Indirect Prompt Injection.

Original paper: Ding et al. (2023), "A Wolf in Sheep's Clothing: Generalized Nested
Jailbreak Prompts can Fool Large Language Models Easily"
arXiv: https://arxiv.org/abs/2311.08268
Source: https://github.com/NJUNLP/ReNeLLM

Mechanism (faithful to the reference ``ReNeLLM`` recipe)
-------------------------------------------------------
For each attempt (up to ``evo_max``):
  1. Prompt rewriting — pick a random number ``n`` of the six rewrite operators, apply a
     random ordering of them to the payload. Operators are shared, verbatim-prompt
     objects in ``mutations.RENELLM_MUTATORS`` (AlterSentenceStructure, ChangeStyle,
     Rephrase, InsertMeaninglessCharacters, MisspellSensitiveWords, Translation).
  2. Scenario nesting — select one of the three container scenarios (code completion,
     story continuation, LaTeX table) and substitute the rewritten payload for
     ``{query}``. Scenarios are loaded verbatim from the seed registry
     (``original/ReNeLLM``).
  3. Query the target and check success; stop early on success.

This mirrors the reference's ``single_attack`` (``n = randint(1, len(Mutations));
mutators = random.sample(...); random.shuffle(...)``) and its random scenario selection.

IPI adaptation vs original
--------------------------
  payload:    the attacker's injection goal (not a harmful QA prompt).
  nesting:    the three containers map onto IPI directly — an injected *document* that is a
              code file / story / table with the instruction hidden inside.
  constraint: the reference runs a ``DeleteHarmLess`` LLM harmfulness filter between
              rewrites and an LLM harmfulness judge for success. We drop the harmfulness
              filter (our payloads are injection instructions, not harmful content) and
              use ``check_ipi_success`` resolved from the scenario instead. An optional
              Judge still scores for the trace.
  delivery:   the nested prompt is injected through the data channel via
              ``make_scenario_target_fn``.

Report of differences is in docs/easyjailbreak-audit.md.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from ..attacker import AdaptiveAttacker
from ..config import ATTACK_TEMP, ATTACK_TOP_P, ATTACK_MAX_TOKENS
from ..evaluator import check_ipi_success
from ..judges import Judge
from ..llm_unified import APILLM, UnifiedLLM
from ..victim import Victim
from .mutations import RENELLM_MUTATORS
from .seeds import SeedTemplate, render

log = logging.getLogger(__name__)


def _load_scenarios() -> list[str]:
    """The three verbatim ReNeLLM nesting containers from the registry."""
    return SeedTemplate().new_seeds(method_list=["ReNeLLM"], prompt_usage="original")


@dataclass
class ReNeLLMResult:
    success: bool
    score: int
    injection: str          # the final nested prompt
    target_response: str
    n_queries: int
    goal: str
    iterations: int = 0
    scenario_used: str = ""
    trace: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ReNeLLMResult(success={self.success}, score={self.score}, "
            f"iterations={self.iterations})"
        )


def run_renellm(
    goal: str,
    target_fn: Callable[[str], str],
    attacker_model: Union[str, UnifiedLLM],
    judge: Optional[Judge] = None,
    evo_max: int = 20,
    eval_mode: str = "contains",
    target_str: str = "",
    scenarios: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> ReNeLLMResult:
    """
    Run the ReNeLLM nested-jailbreak attack.

    Args:
        goal:           Attacker injection goal.
        target_fn:      Callable(injection: str) -> response: str.
        attacker_model: LLM (or model string) used for the rewrite operators.
        judge:          Optional Judge for scoring the trace.
        evo_max:        Max rewrite→nest→query iterations. Default 20.
        eval_mode:      check_ipi_success mode (usually resolved from the scenario).
        target_str:     Expected success string (falls back to goal).
        scenarios:      Override the nesting containers (default: verbatim registry).
        seed:           Optional RNG seed for reproducibility.

    Returns:
        ReNeLLMResult.
    """
    rng = random.Random(seed)
    attacker = (
        APILLM(model=attacker_model, temperature=ATTACK_TEMP, top_p=ATTACK_TOP_P,
               max_tokens=ATTACK_MAX_TOKENS)
        if isinstance(attacker_model, str) else attacker_model
    )
    # A plain (str) -> str callable for the mutation operators.
    llm_call = lambda text: attacker(text)  # noqa: E731  UnifiedLLM.__call__(str) wraps as a user msg
    containers = scenarios or _load_scenarios()
    check_str = target_str or goal

    best = ReNeLLMResult(
        success=False, score=0, injection="", target_response="",
        n_queries=0, goal=goal,
    )

    for it in range(evo_max):
        # --- Stage 1: rewrite the payload (random subset, random order) -------
        payload = goal
        n = rng.randint(1, len(RENELLM_MUTATORS))
        mutators = rng.sample(RENELLM_MUTATORS, n)
        rng.shuffle(mutators)
        for mutator in mutators:
            payload = mutator(llm_call, payload)

        # --- Stage 2: nest into a scenario container --------------------------
        scenario_tmpl = rng.choice(containers)
        injection = render(scenario_tmpl, payload)

        # --- Query + evaluate -------------------------------------------------
        try:
            response = target_fn(injection)
        except Exception as exc:
            log.warning("[ReNeLLM] target_fn raised: %s", exc)
            response = ""
        best.n_queries += 1

        success = check_ipi_success(response, check_str, eval_mode)
        score = 0
        if judge is not None:
            try:
                score = judge.score(injection, response, attacker_goal=goal)
            except Exception as exc:
                log.debug("[ReNeLLM] judge.score failed: %s", exc)

        best.trace.append({
            "iteration": it + 1, "rewritten_payload": payload,
            "injection": injection, "response": response,
            "success": success, "score": score,
        })

        if (success and not best.success) or (score > best.score) or not best.injection:
            best.success = best.success or success
            best.score = max(best.score, score)
            best.injection = injection
            best.target_response = response
            best.scenario_used = scenario_tmpl[:40]

        best.iterations = it + 1
        if success:
            best.success = True
            break

    return best


class ReNeLLMAttacker(AdaptiveAttacker):
    """
    ReNeLLM attacker — iterative rewrite + scenario-nesting (API-compatible).

    From: Ding et al. (2023), "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak
    Prompts can Fool Large Language Models Easily".

    Args:
        attacker_llm: LLM (or model string) driving the rewrite operators.
        judge:        Optional Judge for scoring the trace (not required for success).
        evo_max:      Max iterations. Default 20 (paper).
        eval_mode:    check_ipi_success mode. Default None → auto-detect from the
                      scenario's ``attack_eval_mode``.
        seed:         Optional RNG seed for reproducibility.
    """

    _ATTACK_NAME = "renellm"

    def __init__(
        self,
        attacker_llm: Union[str, APILLM],
        judge: Optional[Judge] = None,
        evo_max: int = 20,
        eval_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(judge)
        self.attacker_llm = (
            APILLM(model=attacker_llm, temperature=ATTACK_TEMP, top_p=ATTACK_TOP_P,
                   max_tokens=ATTACK_MAX_TOKENS)
            if isinstance(attacker_llm, str) else attacker_llm
        )
        self.evo_max   = evo_max
        self.eval_mode = eval_mode
        self.seed      = seed

    @classmethod
    def requires_local_target(cls) -> bool:
        return False

    def run_scenario(self, target: Victim, scenario, verbose: bool = False):
        from ..evaluator import ScenarioResult, make_scenario_target_fn, resolve_attack_target
        target_fn = make_scenario_target_fn(scenario, target)
        target_str, eval_mode = resolve_attack_target(scenario, self.eval_mode)

        r = run_renellm(
            goal=scenario.injection_goal,
            target_fn=target_fn,
            attacker_model=self.attacker_llm,
            judge=self.judge,
            evo_max=self.evo_max,
            eval_mode=eval_mode,
            target_str=target_str,
            seed=self.seed,
        )

        if verbose:
            log.info("[renellm] scenario=%s success=%s score=%d iters=%d",
                     scenario.id, r.success, r.score, r.iterations)

        return ScenarioResult(
            scenario_id=scenario.id,
            goal=scenario.injection_goal,
            success=r.success,
            score=r.score,
            injection=r.injection,
            target_response=r.target_response,
            n_queries=r.n_queries,
            attack=self._ATTACK_NAME,
            extra={"iterations": r.iterations, "scenario_used": r.scenario_used},
        )

    def __repr__(self) -> str:
        return (
            f"ReNeLLMAttacker(attacker={self.attacker_llm.model_name!r}, "
            f"evo_max={self.evo_max})"
        )
