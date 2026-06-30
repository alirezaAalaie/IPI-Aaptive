"""
IPI Attack Research Package
============================

Clean implementations of five adversarial attacks adapted for
Indirect Prompt Injection (IPI) research.

File layout
-----------
  llm_unified.py  — UnifiedLLM (abstract), APILLM (API), LocalLLM (local HF)
  victim.py       — Victim ABC  (interface any defense must implement)
  attacker.py     — BaseAttacker ABC
  prompts.py      — attacker and judge system prompts
  judges.py       — EditDistanceJudge, IPILLMJudge, GPTJudge, KeywordJudge
  tap.py          — TAP + TAPAttacker
  pair.py         — PAIR + PAIRAttacker
  adaptive.py     — RS + Beam-RS + RSAttacker + BeamRSAttacker
  beast.py        — BEAST + BEASTAttacker  (local-only)
  dataset.py      — IPIScenario, IPIDataset, ManualIPIDataset, AgentDojoDataset
  evaluator.py    — IPI success checks + AttackEvaluator
  runner.py       — run_attack / run_experiment (simple scenario-level API)
  target.py       — TargetLLM (Victim wrapping UnifiedLLM) + make_target factory

Quick start
-----------
    from ipi.llm_unified import APILLM
    from ipi.target import TargetLLM, make_target
    from ipi.tap import TAPAttacker
    from ipi.judges import IPILLMJudge
    from ipi.evaluator import AttackEvaluator

    target   = TargetLLM(APILLM("gpt-4o-mini", system_prompt="You are an email agent..."))
    judge    = IPILLMJudge(model="gpt-4o-mini")
    attacker = TAPAttacker(judge=judge, attacker_llm=APILLM("gpt-4o"), depth=10)

    evaluator = AttackEvaluator(target=target, attacker=attacker)
    result    = evaluator.run(dataset)
    print(result.summary())

Custom defense
--------------
    from ipi.victim import Victim

    class MyDefense(Victim):
        backend = "local"

        def __init__(self, model_path):
            self._model = MyDefendedModel(model_path)
            self.system_prompt = AGENT_SYSTEM_PROMPT
            self.model_name = model_path

        def generate(self, messages, max_tokens=200, temperature=0.0):
            return self._model.run_with_defense(messages)

        def get_first_token_logprobs(self, messages, n_top=20):
            return self._model.get_logprobs(messages, n=n_top)

    evaluator = AttackEvaluator(target=MyDefense("path/to/model"), attacker=attacker)
"""

# ---- Victim interface (implement this for custom defenses) ----
from .victim import Victim

# ---- Core LLM interface ----
from .llm_unified import (
    APILLM,
    LocalLLM,
    UnifiedLLM,
    KNOWN_MODELS,
    ModelSpec,
    parse_json_response,
    LogprobNotSupportedError,
    LocalOnlyError,
    METIS_BASE_IR,
    METIS_BASE_GLOBAL,
)

# ---- Target wrapper (Victim wrapping UnifiedLLM) ----
from .target import TargetLLM, make_target

# ---- Judges ----
from .judges import EditDistanceJudge, GPTJudge, IPILLMJudge, KeywordJudge

# ---- Abstract attacker base ----
from .attacker import BaseAttacker

# ---- Attack classes + low-level functions ----
from .tap import TAPAttacker, TAPResult, run_tap
from .pair import PAIRAttacker, PAIRResult, run_pair
from .adaptive import (
    RSAttacker, BeamRSAttacker, AdaptiveResult,
    run_adaptive_rs, run_adaptive_beam,
)
from .beast import BEASTAttacker, BEASTResult, run_beast

# ---- Dataset ----
from .dataset import IPIScenario, IPIDataset, ManualIPIDataset, AgentDojoDataset

# ---- Evaluator ----
from .evaluator import (
    check_function_name,
    check_exact_function_call,
    check_ipi_success,
    get_target_token,
    ipi_early_stopping_condition,
    make_scenario_target_fn,
    ScenarioResult,
    EvalResult,
    AttackEvaluator,
)

# ---- Simple high-level API ----
from .runner import ExperimentResult, run_attack, run_experiment

__all__ = [
    # Victim interface
    "Victim",
    # LLM hierarchy
    "UnifiedLLM", "APILLM", "LocalLLM",
    "KNOWN_MODELS", "ModelSpec", "parse_json_response",
    "LogprobNotSupportedError", "LocalOnlyError",
    "METIS_BASE_IR", "METIS_BASE_GLOBAL",
    # Target (Victim wrapping UnifiedLLM)
    "TargetLLM", "make_target",
    # Judges
    "EditDistanceJudge", "IPILLMJudge", "GPTJudge", "KeywordJudge",
    # Attacker base
    "BaseAttacker",
    # Attack classes
    "TAPAttacker", "PAIRAttacker", "RSAttacker", "BeamRSAttacker", "BEASTAttacker",
    # Low-level attack functions
    "run_tap", "TAPResult",
    "run_pair", "PAIRResult",
    "run_adaptive_rs", "run_adaptive_beam", "AdaptiveResult",
    "run_beast", "BEASTResult",
    # Dataset
    "IPIScenario", "IPIDataset", "ManualIPIDataset", "AgentDojoDataset",
    # Evaluator
    "check_function_name", "check_exact_function_call", "check_ipi_success",
    "get_target_token", "ipi_early_stopping_condition",
    "make_scenario_target_fn", "ScenarioResult", "EvalResult", "AttackEvaluator",
    # Simple API
    "run_attack", "run_experiment", "ExperimentResult",
]
