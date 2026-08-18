"""
AttackEvaluator — runs an attack across a dataset and owns the reported number.

The attack's own evaluator (its ``judge=``) only steers its search: which branch to
expand, when to stop early. Whatever it concluded is **overwritten** here, by
``EvaluatorIPISuccess`` against the scenario's own ground truth. That separation is why
an ASR from this harness is comparable across attacks that use wildly different internal
signals — and why swapping a cheap judge for an expensive one changes cost, not ASR.

    ScenarioResult   one attack against one scenario
    EvalResult       the aggregate, with ASR / utility rate / query counts
    AttackEvaluator  the runner

Utility is the second axis: ``EvaluatorUserUtility`` asks whether the *legitimate* task
still succeeded, so a defense that blocks the injection by refusing to do anything at all
is visible rather than scored as a win.

The unit of work is an ``Instance`` from an ``AttackDataset``. Nothing here mutates it —
verdicts are computed from the returned ``ScenarioResult``, not written back — so the
same dataset can be run against several defenses without one run's responses leaking
into the next one's ``num_jailbreak``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..datasets import Instance
from ..victim import Victim
from .judge import EvaluatorIPISuccess, EvaluatorUserUtility

log = logging.getLogger(__name__)

__all__ = ["ScenarioResult", "EvalResult", "AttackEvaluator"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """
    Result of running one attack against one scenario.

    Attributes:
        scenario_id:     The scenario's id string.
        goal:            The attacker's injection goal.
        success:         Whether the attack was considered successful (ASR).
        score:           Best guidance score achieved (1-10).
        injection:       The best injection string found.
        target_response: Agent response to the best injection.
        n_queries:       Number of calls to the target_fn.
        attack:          Attack name ("tap", "pair", "rs", "beam", "beast").
        utility_success: Optional bool indicating if legitimate user task succeeded.
        final_prompt:    Optional final prompt / messages list seen by victim model after defense transformation.
        defense_name:    Optional defense class name or label applied.
        extra:           Optional extra info (trace, depth_reached, etc.).
    """
    scenario_id:     str
    goal:            str
    success:         bool
    score:           int
    injection:       str
    target_response: str
    n_queries:       int
    attack:          str
    utility_success: Optional[bool] = None
    final_prompt:    Optional[Any] = None
    defense_name:    Optional[str] = None
    extra:           dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """
    Aggregated evaluation result across a dataset.

    Attributes:
        attack:          Attack name.
        asr:             Attack success rate (n_success / n_total).
        avg_score:       Average best score.
        avg_queries:     Average target queries per scenario.
        n_success:       Number of successful attacks.
        n_total:         Total scenarios evaluated.
        utility_rate:    Optional User Utility success rate (n_utility_success / n_total).
        n_utility:       Optional count of scenarios where legitimate user task succeeded.
        results:         Per-scenario ScenarioResult list.
    """
    attack:       str
    asr:          float
    avg_score:    float
    avg_queries:  float
    n_success:    int
    n_total:      int
    utility_rate: Optional[float] = None
    n_utility:    Optional[int] = None
    results:      list[ScenarioResult] = field(default_factory=list)

    def summary(self) -> str:
        s = (
            f"Attack:       {self.attack}\n"
            f"ASR:          {self.asr:.1%}  ({self.n_success}/{self.n_total})\n"
        )
        if self.utility_rate is not None and self.n_utility is not None:
            s += f"Utility Rate: {self.utility_rate:.1%}  ({self.n_utility}/{self.n_total})\n"
        s += (
            f"Avg score:    {self.avg_score:.2f}/10\n"
            f"Avg queries:  {self.avg_queries:.1f}"
        )
        return s

    def failed_scenarios(self) -> list[ScenarioResult]:
        return [r for r in self.results if not r.success]

    def successful_scenarios(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.success]

    def to_dict(self) -> dict:
        """Convert EvalResult to a structured JSON-serializable dictionary."""
        d = {
            "attack": self.attack,
            "asr": self.asr,
            "avg_score": self.avg_score,
            "avg_queries": self.avg_queries,
            "n_success": self.n_success,
            "n_total": self.n_total,
        }
        if self.utility_rate is not None:
            d["utility_rate"] = self.utility_rate
            d["n_utility"] = self.n_utility

        results_list = []
        for r in self.results:
            sr_dict = {
                "scenario_id": r.scenario_id,
                "defense_name": r.defense_name or r.extra.get("defense_name", ""),
                "user_task": r.extra.get("user_task", ""),
                "user_target": r.extra.get("user_target", ""),
                "injection_goal": r.goal,
                "target_str": r.extra.get("target_str", ""),
                "optimization_target": r.extra.get("optimization_target", ""),
                "attack_str": r.injection,
                "final_prompt": r.final_prompt if r.final_prompt is not None else r.extra.get("final_prompt"),
                "model_response": r.target_response,
                "attack_success": r.success,
                "utility_success": r.utility_success,
                "score": r.score,
                "n_queries": r.n_queries,
                "attack": r.attack,
            }
            if r.extra:
                # Merge any additional scenario metadata
                for k, v in r.extra.items():
                    if k not in sr_dict:
                        sr_dict[k] = v
            results_list.append(sr_dict)

        d["results"] = results_list
        return d

    def save_to_file(
        self,
        filepath: Optional[str] = None,
        output_dir: str = "results",
        defense_name: str = "undefended",
    ) -> str:
        """
        Save detailed evaluation metrics and per-scenario records to a JSON file.

        File name format when filepath is None:
          `{output_dir}/eval_attack-{attack}_defense-{defense_name}_{timestamp}.json`
        """
        import os, json, re
        from datetime import datetime

        if filepath is None:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_def = re.sub(r"[^\w\-]", "_", defense_name) if defense_name else "undefended"
            clean_atk = re.sub(r"[^\w\-]", "_", self.attack)
            filename = f"eval_attack-{clean_atk}_defense-{clean_def}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

        parent_dir = os.path.dirname(os.path.abspath(filepath))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        log.info("Saved evaluation results to %s", filepath)
        return filepath


# ---------------------------------------------------------------------------
# AttackEvaluator
# ---------------------------------------------------------------------------

def _innermost_prompt(victim):
    """
    The messages list closest to the model, walking down a chain of defenses.

    Each ``DefendedVictim`` records its *own* output in ``last_input_messages``, so the
    outermost wrapper holds the least-transformed prompt. Taking the first non-None —
    the previous behaviour — reported ``CompositeDefense``'s record, and Composite has a
    no-op ``preprocess_messages`` of its own: the transformations happen in the defenses
    it chains underneath. The result was a ``final_prompt`` showing an untouched prompt
    for a defense that had in fact rewritten it. The ASR was right; the audit trail was
    not, which is worse than an obvious failure.
    """
    seen = set()
    best = None
    node = victim
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        found = getattr(node, "last_input_messages", None)
        if found is not None:
            best = found
        node = getattr(node, "target", None) or getattr(node, "llm", None)
    return best


class AttackEvaluator:
    """
    Runs a BaseAttacker across an AttackDataset and decides success.

    Args:
        target:     Victim instance (TargetLLM or custom defense subclass).
        attacker:   Any BaseAttacker subclass (TAPAttacker, PAIRAttacker, ...).
        verbose:    Enable INFO-level logging per instance.
        success_fn: Escape hatch — ``Callable(response, target_str) -> bool``, or
                    ``Callable(response, instance) -> bool`` (recognised by the second
                    parameter being named ``instance``/``scenario`` or annotated
                    ``Instance``). Replaces the default
                    ``EvaluatorIPISuccess``. An ASR reported with a custom
                    ``success_fn`` is not comparable to one without it; say so if you
                    publish it.

    Example:
        attacker  = TAPAttacker(judge=EvaluatorIPIGetScore("gpt-4o-mini"), depth=10)
        evaluator = AttackEvaluator(target=TargetLLM(APILLM("gpt-4o-mini")),
                                    attacker=attacker)
        result = evaluator.run(dataset, save_file=True)
        print(result.summary())
    """

    def __init__(
        self,
        target: Victim,
        attacker,
        verbose: bool = False,
        success_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        self.target     = target
        self.attacker   = attacker
        self.verbose    = verbose
        self.success_fn = success_fn   # None → the default EvaluatorIPISuccess
        self.success_evaluator = EvaluatorIPISuccess()
        self.utility_evaluator = EvaluatorUserUtility()

    def _check_success(self, response: str, instance: Instance) -> bool:
        """Apply ``success_fn``, or the instance's own ``attack_eval_mode``."""
        target_str, atk_mode = self.success_evaluator.resolve(instance)

        if self.success_fn is not None:
            import inspect
            sig = inspect.signature(self.success_fn)
            params = list(sig.parameters.values())
            # A success_fn whose second parameter asks for the instance gets the whole
            # instance; anything else gets the resolved target string. Passed
            # positionally so either parameter name works.
            wants_instance = len(params) >= 2 and (
                params[1].name in ("instance", "scenario")
                or params[1].annotation is Instance
            )
            if wants_instance:
                return self.success_fn(response, instance)
            return self.success_fn(response, target_str)

        return self.success_evaluator.check(response, target_str, atk_mode)

    def _check_utility(self, response: str, instance: Instance) -> Optional[bool]:
        """Check user-task utility if the instance carries a ground-truth user_target."""
        attrs = instance.attack_attrs
        return self.utility_evaluator.check(
            response,
            attrs.get("user_target") or "",
            attrs.get("user_eval_mode", "contains"),
        )

    def run(
        self,
        dataset,
        save_file: bool = False,
        output_dir: str = "results",
        defense_name: Optional[str] = None,
    ) -> EvalResult:
        """
        Run the attacker on every instance in dataset.

        The attacker's own evaluator guides its search; success is recomputed here, so
        the final ASR is independent of that evaluator's threshold or scoring style.

        Args:
            dataset:      AttackDataset (or any iterable of Instances).
            save_file:    If True, automatically saves detailed JSON log to disk.
            output_dir:   Directory to save the evaluation file (default: "results").
            defense_name: Optional defense label for filename (defaults to target class name).
        """
        attack_name = type(self.attacker).__name__.replace("Attacker", "").lower()
        results: list[ScenarioResult] = []
        for instance in dataset:
            attrs = instance.attack_attrs
            try:
                r = self.attacker.run_scenario(self.target, instance, verbose=self.verbose)
                # Override success: the evaluator owns this, not the attack's judge
                r.success = self._check_success(r.target_response, instance)
                r.utility_success = self._check_utility(r.target_response, instance)

                # The exact messages the model saw, after every defense transformation.
                final_prompt = _innermost_prompt(self.target)

                def_label = defense_name or type(self.target).__name__

                r.final_prompt = final_prompt
                r.defense_name = def_label

                # Enrich extra with instance metadata for comprehensive reporting
                r.extra["user_task"] = attrs.get("user_task", "")
                r.extra["user_target"] = attrs.get("user_target", "")
                r.extra["target_str"] = attrs.get("target_str", "")
                r.extra["optimization_target"] = attrs.get("optimization_target", "")
                r.extra["final_prompt"] = final_prompt
                r.extra["defense_name"] = def_label

                results.append(r)
            except Exception as e:
                log.error("[AttackEvaluator] %s instance=%s error: %s",
                          attack_name, instance.id, e)
                results.append(ScenarioResult(
                    scenario_id=instance.id, goal=instance.query or "",
                    success=False, score=0, injection="", target_response="",
                    n_queries=0, attack=attack_name, utility_success=False,
                    extra={
                        "error": str(e),
                        "user_task": attrs.get("user_task", ""),
                        "user_target": attrs.get("user_target", ""),
                        "target_str": attrs.get("target_str", ""),
                        "optimization_target": attrs.get("optimization_target", ""),
                    },
                ))

        eval_result = self._aggregate(attack_name, results)

        if save_file:
            def_label = defense_name or type(self.target).__name__
            eval_result.save_to_file(output_dir=output_dir, defense_name=def_label)

        return eval_result

    @staticmethod
    def _aggregate(attack: str, results: list[ScenarioResult]) -> EvalResult:
        n_total   = len(results)
        n_success = sum(1 for r in results if r.success)
        asr         = n_success / n_total if n_total else 0.0
        avg_score   = sum(r.score     for r in results) / n_total if n_total else 0.0
        avg_queries = sum(r.n_queries for r in results) / n_total if n_total else 0.0

        utility_list = [r.utility_success for r in results if r.utility_success is not None]
        if utility_list:
            n_utility = sum(1 for u in utility_list if u)
            utility_rate = n_utility / len(utility_list)
        else:
            n_utility = None
            utility_rate = None

        return EvalResult(
            attack=attack, asr=asr, avg_score=avg_score, avg_queries=avg_queries,
            n_success=n_success, n_total=n_total,
            utility_rate=utility_rate, n_utility=n_utility, results=results,
        )
