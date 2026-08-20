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
     objects in ``mutation.RENELLM_MUTATORS`` (AlterSentenceStructure, ChangeStyle,
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
              rewrites and an LLM harmfulness judge for success. We drop the filter — our
              payloads are injection instructions, not harmful content, so it would reject
              every candidate and starve the search — and use ``EvaluatorIPISuccess``
              against the scenario's ground truth instead. The filter is ported and
              available as ``constraint.DeleteHarmLess`` for a paper-faithful row. An
              optional guidance evaluator still scores the reported candidate.
  delivery:   the nested prompt is injected through the data channel via
              ``harness.VictimQuery`` (which wraps ``harness.make_target_fn``).

Composition
-----------
One attempt is::

    candidate = mutator(AttackDataset([candidate]))   # x n, from RENELLM_MUTATORS
    candidate.jailbreak_prompt = render(container, candidate.query)
    batch = query(AttackDataset([candidate]))         # harness.VictimQuery
    evaluator(batch)                                  # EvaluatorIPISuccess

The payload rides on an ``Instance.query`` and each rewrite operator runs on the dataset
path, so a candidate's ``level`` counts how many operators touched it and the rewrite
chain is inspectable. The containers and the operator prompts come from ``ipi.seed`` and
``ipi.mutation``.

Behaviour change on moving to the ``single_attack`` seam
---------------------------------------------------------
The ``ReNeLLMResult`` dataclass, its ``trace`` and the ``run_renellm`` free function are
gone; the attack is ``single_attack`` over an ``AttackDataset`` and the best candidate is
picked by ``BaseAttacker.keep_best``. One consequence, the same one GPTFuzzer's
conversion has: **the guidance evaluator is now called once per scenario**, on the
candidate that ends up being reported, instead of once per attempt. It never decided
success — but it used to decide *which failed attempt* got reported, and that now falls
to ``keep_best``, which reports the first (all failures score alike). With ``judge=None``
— the default, and how every published row was run — the reported candidate is
unchanged. Query counts are unchanged: one victim query per attempt, as before.

Report of differences is in docs/easyjailbreak-audit.md.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Union

from ..attacker import AdaptiveAttacker, as_attacker_llm
from ..datasets import AttackDataset, Instance
from ..harness import VictimQuery
from ..llm_unified import APILLM
from ..metrics import Evaluator, EvaluatorIPISuccess, resolve_attack_target
from ..mutation import RENELLM_MUTATORS
from ..seed import SeedTemplate, render
from ..victim import Victim

log = logging.getLogger(__name__)

__all__ = ["ReNeLLMAttacker"]


def _load_scenarios() -> list[str]:
    """The three verbatim ReNeLLM nesting containers from the registry."""
    return SeedTemplate().new_seeds(
        prompt_usage="attack", method_list=["ReNeLLM"], variant="original",
    )


class ReNeLLMAttacker(AdaptiveAttacker):
    """
    ReNeLLM attacker — iterative rewrite + scenario nesting.

    From: Ding et al. (2023), "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak
    Prompts can Fool Large Language Models Easily".

    Args:
        attacker_llm: LLM (or model string) driving the six rewrite operators.
        judge:        Optional guidance Evaluator. Grades the reported candidate so the
                      row carries a 1-10 score; it never decides success, which comes
                      from the scenario's ground truth.
        evo_max:      Max rewrite→nest→query attempts. Default 20 (paper). One victim
                      query each, so this is also the query budget.
        eval_mode:    ``check_ipi_success`` mode. Default None → read the scenario's own
                      ``attack_eval_mode``. Pass a string only to force a deliberate
                      deviation.
        scenarios:    Override the nesting containers (default: the three upstream ones).
        seed:         RNG seed, for reproducibility.
    """

    _ATTACK_NAME = "renellm"

    def __init__(
        self,
        attacker_llm: Union[str, APILLM],
        judge: Optional[Evaluator] = None,
        evo_max: int = 20,
        eval_mode: Optional[str] = None,
        scenarios: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(judge)
        self.attacker_llm = as_attacker_llm(attacker_llm)
        self.evo_max      = evo_max
        self.eval_mode    = eval_mode
        self.seed         = seed
        self.containers   = scenarios or _load_scenarios()
        # The six rewrite operators, with the attacker LLM bound. They rewrite `query`,
        # which is where the payload lives on the carrier.
        self.mutators = RENELLM_MUTATORS(self.attacker_llm)

    # ------------------------------------------------------------------
    # The attack
    # ------------------------------------------------------------------

    def single_attack(self, target: Victim, instance: Instance,
                      verbose: bool = False) -> AttackDataset:
        """
        Rewrite, nest, query — up to ``evo_max`` times, stopping on the first success.

        Every attempt starts again from the untouched payload (the reference does the
        same: its rewrites are a fresh random subset each round, not a chain across
        rounds), so ``root`` is the parent of every candidate and a candidate's
        ``level`` is the number of operators that touched it.
        """
        goal = instance.query or ""
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)
        rng = random.Random(self.seed)
        evaluator = EvaluatorIPISuccess(mode=eval_mode)
        query = VictimQuery(instance, target)

        root = Instance(
            id="renellm-root",
            query=goal,
            reference_responses=[target_str],
            attack_attrs={"target_str": target_str, "attack_eval_mode": eval_mode},
        )

        best: Optional[Instance] = None
        succeeded = False
        iterations = 0

        for attempt in range(self.evo_max):
            iterations = attempt + 1

            # --- Stage 1: rewrite the payload (random subset, random order) ---
            n = rng.randint(1, len(self.mutators))
            chain = rng.sample(self.mutators, n)
            rng.shuffle(chain)
            candidate = root
            for mutator in chain:
                candidate = mutator(AttackDataset([candidate]))[0]

            # --- Stage 2: nest the rewritten payload in a scenario container ---
            container = rng.choice(self.containers)
            candidate.jailbreak_prompt = render(container, candidate.query)
            candidate.attack_attrs["nesting_scenario"] = container[:40]

            # --- Query + evaluate -------------------------------------------
            batch = query(AttackDataset([candidate]))
            if not len(batch):
                break
            evaluator(batch)
            best = self.keep_best(best, batch)

            if verbose:
                log.info("[ReNeLLM] attempt=%d/%d  operators=%d  level=%d  success=%s",
                         iterations, self.evo_max, n, candidate.level,
                         bool(candidate.eval_results and candidate.eval_results[-1]))

            if candidate.eval_results and candidate.eval_results[-1]:
                succeeded = True
                break

        return self.report(
            best, query,
            success=succeeded,
            score=self._grade(best, goal),
            iterations=iterations,
            scenario_used=(best.attack_attrs.get("nesting_scenario", "")
                           if best is not None else ""),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grade(self, best: Optional[Instance], goal: str) -> Optional[int]:
        """
        The reported 1-10 score: the judge's if there is one, else the binary verdict.

        ``None`` lets ``build_result`` fall back to ``eval_results[-1]`` — the ground
        truth mapped through ``Evaluator.as_score`` (10 or 1).
        """
        if best is None or self.judge is None:
            return None
        try:
            return self.judge.score(
                best.attack_attrs.get("injection", "") or best.jailbreak_prompt or "",
                best.target_responses[-1] if best.target_responses else "",
                attacker_goal=goal,
            )
        except Exception as exc:
            log.debug("[ReNeLLM] judge.score failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return (f"ReNeLLMAttacker(attacker={self.attacker_llm.model_name!r}, "
                f"evo_max={self.evo_max})")
