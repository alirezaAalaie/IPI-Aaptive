"""
ipi.metrics — everything that decides "did it work?" and "how close was it?".

Mirrors EasyJailbreak's ``easyjailbreak/metrics/``, which deliberately makes no
Scorer/Evaluator split: both jobs are ``Evaluator`` subclasses over an
``AttackDataset``, told apart by what they write onto each ``Instance``.

    guidance (int 1-10)              success (bool)
    ------------------------------   ----------------------------------------
    EvaluatorGenerativeGetScore      EvaluatorIPISuccess     <- the ASR authority
    EvaluatorIPIGetScore             EvaluatorUserUtility    <- the other axis
    EvaluatorEditDistanceGetScore    EvaluatorMatch
                                     EvaluatorPrefixExactMatch
                                     EvaluatorPatternJudge
                                     EvaluatorKeywordJudge

plus ``AttackEvaluator`` (the runner) and the primitives in ``checks.py`` that the
success evaluators wrap.

This replaces ``ipi/scoring.py`` and ``ipi/judges.py``. The old ``Judge`` name is gone:
it meant "guidance" here and "success" everywhere else, which is exactly the confusion
upstream's naming avoids.

    old                            new
    ---                            ---
    judges.Judge                   metrics.Evaluator
    judges.IPILLMJudge             metrics.EvaluatorIPIGetScore
    judges.GPTJudge                metrics.EvaluatorGenerativeGetScore
    judges.EditDistanceJudge       metrics.EvaluatorEditDistanceGetScore
    judges.KeywordJudge            metrics.EvaluatorKeywordJudge
    evaluator.check_ipi_success    metrics.check_ipi_success  (moved, unchanged)
    evaluator.AttackEvaluator      metrics.AttackEvaluator    (moved)

Not ported: upstream's ``metrics/Metric/`` (``AttackSuccessRate``, perplexity).
``EvalResult`` already aggregates ASR, utility and query counts over our
``ScenarioResult`` records, and a second aggregation path would be one more place for
the reported number to diverge.
"""
from .base import Evaluator, instance_from_scalars, get_attr
from .checks import (
    check_function_name,
    check_exact_function_call,
    check_ipi_success,
    check_user_utility,
    resolve_attack_target,
    REFUSAL_PHRASES,
)
from .get_score import (
    EvaluatorGenerativeGetScore,
    EvaluatorIPIGetScore,
    EvaluatorEditDistanceGetScore,
)
from .judge import (
    EvaluatorIPISuccess,
    EvaluatorUserUtility,
    EvaluatorMatch,
    EvaluatorPrefixExactMatch,
    EvaluatorPatternJudge,
    EvaluatorKeywordJudge,
)
from .attack_evaluator import ScenarioResult, EvalResult, AttackEvaluator

__all__ = [
    # Base
    "Evaluator", "instance_from_scalars", "get_attr",
    # Primitives
    "check_function_name", "check_exact_function_call", "check_ipi_success",
    "check_user_utility", "resolve_attack_target", "REFUSAL_PHRASES",
    # Guidance
    "EvaluatorGenerativeGetScore", "EvaluatorIPIGetScore",
    "EvaluatorEditDistanceGetScore",
    # Success
    "EvaluatorIPISuccess", "EvaluatorUserUtility", "EvaluatorMatch",
    "EvaluatorPrefixExactMatch", "EvaluatorPatternJudge", "EvaluatorKeywordJudge",
    # Runner
    "ScenarioResult", "EvalResult", "AttackEvaluator",
]
