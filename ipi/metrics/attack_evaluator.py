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

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from ..datasets import Instance
from ..victim import Victim
from .judge import EvaluatorIPISuccess, EvaluatorUserUtility

log = logging.getLogger(__name__)

__all__ = ["ScenarioResult", "EvalResult", "AttackEvaluator", "evaluate_saved_results"]


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
        success:         Whether the attack was considered successful (ASR), or None if deferred.
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
    success:         Optional[bool]
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
        asr:             Attack success rate (n_success / n_total), or None if deferred.
        avg_score:       Average best score.
        avg_queries:     Average target queries per scenario.
        n_success:       Number of successful attacks, or None if deferred.
        n_total:         Total scenarios evaluated.
        utility_rate:    Optional User Utility success rate, over the scenarios that
                         actually carry a ``user_target`` — NOT over n_total. A
                         scenario with no utility label, or one whose attack raised,
                         has no utility verdict and stays out of both sides.
        n_utility:       Count of those scenarios where the legitimate user task succeeded.
        n_utility_total: How many scenarios had a utility verdict at all — the
                         denominator of ``utility_rate``. Equals n_total on the
                         packaged benchmark (all 360 carry a user_target).
        results:         Per-scenario ScenarioResult list.
    """
    attack:          str
    asr:             Optional[float]
    avg_score:       float
    avg_queries:     float
    n_success:       Optional[int]
    n_total:         int
    utility_rate:    Optional[float] = None
    n_utility:       Optional[int] = None
    n_utility_total: Optional[int] = None
    results:         list[ScenarioResult] = field(default_factory=list)

    def summary(self) -> str:
        if self.asr is None:
            asr_line = "ASR:          Pending evaluation (deferred judging)"
        else:
            asr_line = f"ASR:          {self.asr:.1%}  ({self.n_success}/{self.n_total})"
        s = (
            f"Attack:       {self.attack}\n"
            f"{asr_line}\n"
        )
        if self.utility_rate is not None and self.n_utility is not None:
            # The denominator is the labelled subset, not n_total — printing n_total
            # here made the percentage and the fraction disagree on any dataset where
            # some scenarios carry no user_target.
            denom = self.n_utility_total if self.n_utility_total is not None else self.n_total
            s += f"Utility Rate: {self.utility_rate:.1%}  ({self.n_utility}/{denom})\n"
            if denom != self.n_total:
                s += f"              ({self.n_total - denom} of {self.n_total} scenarios had no utility verdict)\n"
        s += (
            f"Avg score:    {self.avg_score:.2f}/10\n"
            f"Avg queries:  {self.avg_queries:.1f}"
        )
        return s

    def failed_scenarios(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.success is False]

    def successful_scenarios(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.success is True]

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
            d["n_utility_total"] = self.n_utility_total

        results_list = []
        for r in self.results:
            sr_dict = {
                "scenario_id": r.scenario_id,
                "defense_name": r.defense_name or r.extra.get("defense_name", ""),
                "user_task": r.extra.get("user_task", ""),
                "user_target": r.extra.get("user_target", ""),
                "injection_goal": r.goal,
                "target_str": r.extra.get("target_str", ""),
                "judge_question": r.extra.get("judge_question", ""),
                "attack_eval_mode": r.extra.get("attack_eval_mode", "contains"),
                "optimization_target": r.extra.get("optimization_target", ""),
                "attack_str": r.injection,
                "final_prompt": r.final_prompt if r.final_prompt is not None else r.extra.get("final_prompt"),
                "final_prompt_source": r.extra.get("final_prompt_source", ""),
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

    This is the **fallback** for ``final_prompt``, not the primary: it is whatever call
    ran last, which for any search attack is its last candidate rather than the best one
    it reports. See ``_final_prompt``.
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


def _final_prompt(victim, instance: Instance, result: "ScenarioResult"):
    """
    The prompt that goes in the audit trail, and where it came from.

    Returns ``(messages, source)`` with ``source`` in ``"rebuilt"`` / ``"last_call"`` /
    ``"unavailable"``, recorded alongside so a reader never has to guess how literal
    the record is.

    ``"rebuilt"`` is the one that answers the question actually being asked — *what
    prompt produced the response this row reports?* Reading ``last_input_messages``
    answers a different one: what did the victim see on its final call. Those coincide
    only for the one-shot attacks. TAP, PAIR, GCG, RS and Beam-RS all report their
    **best** candidate while their last query was some later, worse one, so the
    ``final_prompt`` beside ``attack_str`` was a different injection's prompt.

    Rebuilding replays ``harness`` with the separator and template the attack used and
    runs the defense chain again; it never queries the model. A defense whose
    preprocessing is nondeterministic (a sanitizer that samples) yields an equivalent
    prompt rather than a byte-identical one — still the right injection, which the
    recorded last call was not.
    """
    from ..harness import rebuild_victim_prompt

    if result.injection:
        rebuilt = rebuild_victim_prompt(instance, victim, result.injection)
        if rebuilt is not None:
            return rebuilt, "rebuilt"
    recorded = _innermost_prompt(victim)
    if recorded is not None:
        return recorded, "last_call"
    return None, "unavailable"


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
        deferred_judging: bool = False,
        progress: Union[bool, int] = False,
    ):
        self.target     = target
        self.attacker   = attacker
        self.verbose    = verbose
        self.success_fn = success_fn   # None → the default EvaluatorIPISuccess
        self.deferred_judging = deferred_judging
        self.progress   = progress
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
        progress: Optional[Union[bool, int]] = None,
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
            progress:     If True or int step > 0, prints progress percentages during execution.
        """
        attack_name = type(self.attacker).__name__.replace("Attacker", "").lower()
        def_label = defense_name or type(self.target).__name__

        total = len(dataset) if hasattr(dataset, "__len__") else None
        show_progress = self.progress if progress is None else progress
        if isinstance(show_progress, int) and show_progress > 0:
            step = show_progress
        elif show_progress:
            step = max(1, total // 10) if total else 1
        else:
            step = None

        results: list[ScenarioResult] = []
        for idx, instance in enumerate(dataset, start=1):
            if step and total:
                if idx == 1 or idx % step == 0 or idx == total:
                    pct = (idx / total) * 100
                    print(f"[{def_label}] [{idx:02d}/{total:02d} ({pct:5.1f}%)] Processing scenario: {instance.id}")
            attrs = instance.attack_attrs
            try:
                r = self.attacker.run_scenario(self.target, instance, verbose=self.verbose)
                if self.deferred_judging:
                    r.success = None
                    r.utility_success = None
                else:
                    # Override success: the evaluator owns this, not the attack's judge
                    r.success = self._check_success(r.target_response, instance)
                    r.utility_success = self._check_utility(r.target_response, instance)

                # The messages the model saw for the injection this row reports —
                # rebuilt from that injection, not read back off the last call.
                final_prompt, prompt_source = _final_prompt(self.target, instance, r)

                def_label = defense_name or type(self.target).__name__

                r.final_prompt = final_prompt
                r.defense_name = def_label

                # Enrich extra with instance metadata for comprehensive reporting
                r.extra["user_task"] = attrs.get("user_task", "")
                r.extra["user_target"] = attrs.get("user_target", "")
                r.extra["target_str"] = attrs.get("target_str", "")
                r.extra["optimization_target"] = attrs.get("optimization_target", "")
                r.extra["judge_question"] = attrs.get("judge_question", "")
                r.extra["attack_eval_mode"] = attrs.get("attack_eval_mode", "")
                r.extra["user_eval_mode"] = attrs.get("user_eval_mode", "contains")
                r.extra["model_response"] = r.target_response
                r.extra["final_prompt"] = final_prompt
                r.extra["final_prompt_source"] = prompt_source
                r.extra["defense_name"] = def_label

                results.append(r)
            except Exception as e:
                log.error("[AttackEvaluator] %s instance=%s error: %s",
                          attack_name, instance.id, e)
                def_label = defense_name or type(self.target).__name__
                success_val = None if self.deferred_judging else False
                results.append(ScenarioResult(
                    scenario_id=instance.id, goal=instance.query or "",
                    success=success_val, score=0, injection="", target_response="",
                    # utility_success stays None: an attack that crashed produced no
                    # response to judge, and False here would enter the utility
                    # denominator as a failure — on a dataset with no utility labels
                    # that turns "no utility line" into "Utility Rate: 0.0%".
                    n_queries=0, attack=attack_name, utility_success=None,
                    defense_name=def_label,
                    extra={
                        "error": str(e),
                        "user_task": attrs.get("user_task", ""),
                        "user_target": attrs.get("user_target", ""),
                        "target_str": attrs.get("target_str", ""),
                        "optimization_target": attrs.get("optimization_target", ""),
                        "judge_question": attrs.get("judge_question", ""),
                        "attack_eval_mode": attrs.get("attack_eval_mode", ""),
                        "user_eval_mode": attrs.get("user_eval_mode", "contains"),
                        "model_response": "",
                        "defense_name": def_label,
                    },
                ))

        eval_result = self._aggregate(attack_name, results)

        if save_file:
            def_label = defense_name or type(self.target).__name__
            eval_result.save_to_file(output_dir=output_dir, defense_name=def_label)

        return eval_result

    @staticmethod
    def _aggregate(attack: str, results: list[ScenarioResult]) -> EvalResult:
        n_total = len(results)
        evaluated_success = [r.success for r in results if r.success is not None]
        if not evaluated_success and n_total > 0:
            n_success = None
            asr = None
        else:
            n_success = sum(1 for s in evaluated_success if s)
            asr = n_success / n_total if n_total else 0.0

        avg_score   = sum(r.score     for r in results) / n_total if n_total else 0.0
        avg_queries = sum(r.n_queries for r in results) / n_total if n_total else 0.0

        utility_list = [r.utility_success for r in results if r.utility_success is not None]
        if utility_list:
            n_utility = sum(1 for u in utility_list if u)
            n_utility_total = len(utility_list)
            utility_rate = n_utility / n_utility_total
        else:
            n_utility = None
            n_utility_total = None
            utility_rate = None

        return EvalResult(
            attack=attack, asr=asr, avg_score=avg_score, avg_queries=avg_queries,
            n_success=n_success, n_total=n_total,
            utility_rate=utility_rate, n_utility=n_utility,
            n_utility_total=n_utility_total, results=results,
        )


def evaluate_saved_results(
    data_or_path: Union[str, os.PathLike, dict, EvalResult, list],
    judge: Optional[Any] = None,
    output_path: Optional[str] = None,
    progress: Union[bool, int] = False,
) -> EvalResult:
    """
    Evaluate previously generated model responses and calculate ASR and utility.

    Enables decoupled/deferred judging so that model generation can run without
    an active judge model or network connection, and evaluation can be performed
    later deterministically or via an LLM judge.

    Args:
        data_or_path: File path (str or Path), dictionary, list of records, or EvalResult.
        judge:        Judge evaluator (defaults to EvaluatorCyberSecEvalJudge()).
        output_path:  Optional path to save updated evaluation results JSON.
        progress:     If True or int step > 0, prints evaluation progress percentages.

    Returns:
        Updated EvalResult with computed ASR, utility rate, and per-scenario results.
    """
    from .checks import check_ipi_success, check_user_utility
    from .judge import EvaluatorCyberSecEvalJudge

    if judge is None:
        judge = EvaluatorCyberSecEvalJudge()

    raw_data: dict
    if isinstance(data_or_path, (str, os.PathLike)):
        with open(data_or_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            raw_data = {"results": loaded}
        elif isinstance(loaded, dict):
            raw_data = loaded
        else:
            raise ValueError(f"Invalid JSON content in {data_or_path}")
    elif isinstance(data_or_path, EvalResult):
        raw_data = data_or_path.to_dict()
    elif isinstance(data_or_path, dict):
        raw_data = dict(data_or_path)
    elif isinstance(data_or_path, list):
        raw_data = {"results": data_or_path}
    else:
        raise TypeError(f"Unsupported type for data_or_path: {type(data_or_path)}")

    if "results" in raw_data and isinstance(raw_data["results"], list):
        records = raw_data["results"]
    elif "rows" in raw_data and isinstance(raw_data["rows"], list):
        records = raw_data["rows"]
    else:
        records = [raw_data]

    attack_name = raw_data.get("attack") or "evaluation"

    total = len(records)
    if isinstance(progress, int) and progress > 0:
        step = progress
    elif progress:
        step = max(1, total // 10) if total else 1
    else:
        step = None

    scenario_results: list[ScenarioResult] = []
    for idx, record in enumerate(records, start=1):
        if step and total:
            if idx == 1 or idx % step == 0 or idx == total:
                pct = (idx / total) * 100
                print(f"[{attack_name}] Adjudicating: [{idx:02d}/{total:02d} ({pct:5.1f}%)]")
        if isinstance(record, ScenarioResult):
            model_response = record.target_response
            extra = record.extra
            judge_question = extra.get("judge_question", "")
            target_str = extra.get("target_str", "")
            eval_mode = extra.get("attack_eval_mode", "contains")
            user_target = extra.get("user_target", "")
            user_eval_mode = extra.get("user_eval_mode", "contains")
            rec_attack = record.attack
            rec_id = record.scenario_id
            rec_goal = record.goal
            rec_score = record.score
            rec_injection = record.injection
            rec_queries = record.n_queries
            rec_def = record.defense_name
            rec_final_prompt = record.final_prompt
        elif isinstance(record, dict):
            extra = record.get("extra", {}) if isinstance(record.get("extra"), dict) else {}
            model_response = record.get("model_response")
            if model_response is None:
                model_response = record.get("target_response", "")
            judge_question = record.get("judge_question") or extra.get("judge_question", "")
            target_str = record.get("target_str") or extra.get("target_str", "")
            eval_mode = record.get("attack_eval_mode") or extra.get("attack_eval_mode", "contains")
            user_target = record.get("user_target") or extra.get("user_target", "")
            user_eval_mode = record.get("user_eval_mode") or extra.get("user_eval_mode", "contains")
            rec_attack = record.get("attack") or attack_name
            rec_id = str(record.get("scenario_id", ""))
            rec_goal = str(record.get("injection_goal") or record.get("goal", ""))
            rec_score = int(record.get("score", 0))
            rec_injection = str(record.get("attack_str") or record.get("injection", ""))
            rec_queries = int(record.get("n_queries", 0))
            rec_def = record.get("defense_name")
            rec_final_prompt = record.get("final_prompt")
        else:
            continue

        # Evaluate attack success
        if judge_question:
            if hasattr(judge, "judge"):
                atk_success = judge.judge(
                    model_response,
                    judge_question=judge_question,
                    target_str=target_str,
                )
            elif hasattr(judge, "check"):
                atk_success = judge.check(
                    model_response,
                    target_str=target_str,
                    judge_question=judge_question,
                )
            elif callable(judge):
                try:
                    atk_success = bool(judge(model_response, judge_question=judge_question, target_str=target_str))
                except TypeError:
                    atk_success = bool(judge(model_response, target_str))
            else:
                atk_success = False
        else:
            atk_success = check_ipi_success(model_response, target_str, eval_mode)

        # Evaluate user utility
        if user_target:
            util_success = check_user_utility(model_response, user_target, user_eval_mode)
        else:
            util_success = None

        if isinstance(record, dict):
            record["attack_success"] = atk_success
            record["utility_success"] = util_success
            if "success" in record:
                record["success"] = atk_success
        elif isinstance(record, ScenarioResult):
            record.success = atk_success
            record.utility_success = util_success

        sr = ScenarioResult(
            scenario_id=rec_id,
            goal=rec_goal,
            success=atk_success,
            score=rec_score,
            injection=rec_injection,
            target_response=str(model_response),
            n_queries=rec_queries,
            attack=str(rec_attack),
            utility_success=util_success,
            final_prompt=rec_final_prompt,
            defense_name=rec_def,
            extra=dict(extra),
        )
        sr.extra["user_task"] = record.get("user_task", extra.get("user_task", "")) if isinstance(record, dict) else extra.get("user_task", "")
        sr.extra["user_target"] = user_target
        sr.extra["target_str"] = target_str
        sr.extra["judge_question"] = judge_question
        sr.extra["attack_eval_mode"] = eval_mode
        sr.extra["user_eval_mode"] = user_eval_mode
        sr.extra["defense_name"] = rec_def or extra.get("defense_name", "")
        sr.extra["model_response"] = model_response
        scenario_results.append(sr)

    eval_result = AttackEvaluator._aggregate(attack_name, scenario_results)

    if output_path is not None:
        raw_data["attack"] = eval_result.attack
        raw_data["asr"] = eval_result.asr
        raw_data["n_success"] = eval_result.n_success
        raw_data["n_total"] = eval_result.n_total
        raw_data["avg_score"] = eval_result.avg_score
        raw_data["avg_queries"] = eval_result.avg_queries
        if eval_result.utility_rate is not None:
            raw_data["utility_rate"] = eval_result.utility_rate
            raw_data["n_utility"] = eval_result.n_utility
            raw_data["n_utility_total"] = eval_result.n_utility_total
        if "results" in raw_data:
            raw_data["results"] = eval_result.to_dict()["results"]
        elif "rows" in raw_data:
            raw_data["rows"] = records
        else:
            raw_data["results"] = eval_result.to_dict()["results"]

        parent_dir = os.path.dirname(os.path.abspath(output_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, default=str)
        log.info("Saved updated evaluation results to %s", output_path)

    return eval_result
