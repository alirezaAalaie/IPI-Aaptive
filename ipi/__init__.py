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
  seed/           — SeedTemplate: the registry of every fixed string an attack uses
                    (injection payloads, attacker/judge/constraint system prompts,
                    in-context demonstrations)
  metrics/        — the Evaluator family: *GetScore guidance (1-10) beside
                    *Judge/*Match success (bool), plus AttackEvaluator, which owns
                    the reported ASR. Replaces judges.py / scoring.py
  mutation/       — MutationBase operators: generation (LLM-driven) · rule
                    (deterministic encodings, scrambles, crossover). Replaces
                    attacks/mutations.py
  selector/       — SelectPolicy: which candidate gets the next victim query
                    (Random · RoundRobin · UCB · EXP3 · MCTSExplore · SelectBasedOnScores
                    · ReferenceLossSelector)
  constraint/     — ConstraintBase filters: DeleteOffTopic · DeleteHarmLess
                    · PerplexityConstraint
  attacks/        — attack implementations (subpackage)
    tap.py        — TAP + TAPAttacker
    pair.py       — PAIR + PAIRAttacker
    adaptive.py   — RS + Beam-RS + RSAttacker + BeamRSAttacker
    beast.py      — BEAST + BEASTAttacker  (local-only)
    autodan.py    — AutoDAN GA/HGA + AutoDANAttacker  (local-only)
    gcg.py        — GCG + GCGAttacker  (local-only)
    static_injection.py — Naive, Escape, Ignore, FakeCompletion, Combined (OPI, API-compatible)
  datasets/       — Instance (the carrier every component speaks) · AttackDataset
                    · DualVerifiableDataset
  dataset.py      — LEGACY IPIScenario / IPIDataset; deleted once recipes migrate
  evaluator.py    — harness helpers (make_scenario_target_fn, RS early stop)
  runner.py       — run_attack / run_experiment (simple scenario-level API)
  target.py       — TargetLLM (Victim wrapping UnifiedLLM) + make_target factory

Quick start
-----------
    from ipi.llm_unified import APILLM
    from ipi.target import TargetLLM, make_target
    from ipi.attacks.tap import TAPAttacker
    from ipi.metrics import EvaluatorIPIGetScore, AttackEvaluator

    target   = TargetLLM(APILLM("gpt-4o-mini", system_prompt="You are an email agent..."))
    judge    = EvaluatorIPIGetScore(model="gpt-4o-mini")
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

# ---- Abstract Attacker Base ----
from .attacker import BaseAttacker, StaticAttacker, AdaptiveAttacker, JudgeGuidedAttacker

# ---- Mutation operators ----
from .mutation import MutationBase, GPTFUZZER_MUTATORS, RENELLM_MUTATORS

# ---- Selection policies ----
from .selector import SelectPolicy, MCTSExploreSelectPolicy, SelectBasedOnScores

# ---- Constraints ----
from .constraint import ConstraintBase, DeleteOffTopic, PerplexityConstraint

# ---- Seed / prompt registry ----
from .seed import (
    SeedBase, SeedTemplate, PLACEHOLDER, TARGET_PLACEHOLDER,
    load_seed_templates, render, sample_population, ica_demos,
)

# ---- Attack classes + low-level functions ----
from .attacks.tap import TAPAttacker, TAPResult, run_tap
from .attacks.pair import PAIRAttacker, PAIRResult, run_pair
from .attacks.adaptive import (
    RSAttacker, BeamRSAttacker, AdaptiveResult,
    run_adaptive_rs, run_adaptive_beam,
)
from .attacks import (
    BEASTAttacker, BEASTResult, run_beast,
    AutoDANAttacker, AutoDANResult, run_autodan_ga, run_autodan_hga,
    GCGAttacker, GCGResult, run_gcg,
)
from .attacks.static_injection import (
    NaiveAttacker, EscapeAttacker, IgnoreAttacker,
    FakeCompletionAttacker, CombinedAttacker,
    StaticInjectionResult, run_static_injection, create_static_attacker,
    build_naive_injection, build_escape_injection, build_ignore_injection,
    build_fake_completion_injection, build_combined_injection,
)
from .attacks import (
    DeepInceptionAttacker, DeepInceptionResult, run_deepinception,
    ICAAttacker, ICAResult, run_ica,
    MultilingualAttacker, MultilingualResult, run_multilingual,
    ReNeLLMAttacker, ReNeLLMResult, run_renellm,
    GPTFuzzerAttacker, GPTFuzzerResult, run_gptfuzzer,
)

# ---- Dataset ----
# The carrier (Instance / AttackDataset) is the new seam; ipi.dataset's IPIScenario
# is legacy and disappears once the recipes migrate. Both are exported during the
# transition, under distinct names so nothing is ambiguous.
from .datasets import Instance, AttackDataset
from .datasets import DualVerifiableDataset, load_dual_verifiable
from .dataset import IPIScenario, IPIDataset
from .dataset import DualVerifiableDataset as LegacyDualVerifiableDataset

# ---- Metrics (success + guidance) ----
from .metrics import (
    Evaluator,
    check_function_name,
    check_exact_function_call,
    check_ipi_success,
    check_user_utility,
    resolve_attack_target,
    EvaluatorGenerativeGetScore,
    EvaluatorIPIGetScore,
    EvaluatorEditDistanceGetScore,
    EvaluatorIPISuccess,
    EvaluatorUserUtility,
    EvaluatorMatch,
    EvaluatorPrefixExactMatch,
    EvaluatorPatternJudge,
    EvaluatorKeywordJudge,
    ScenarioResult,
    EvalResult,
    AttackEvaluator,
)

# ---- Harness helpers ----
from .evaluator import (
    get_target_token,
    ipi_early_stopping_condition,
    make_scenario_target_fn,
)

# ---- Defenses ----
from .defenses import (
    DefendedVictim,
    InstructionalDefense,
    ReminderDefense,
    SandwichDefense,
    SpotlightDefense,
    CompositeDefense,
    SecAlignDefense,
    StruQDefense,
)

# ---- Simple high-level API ----
from .runner import ExperimentResult, run_attack, run_experiment

# ---- Subpackage namespaces ----
from . import (datasets, defenses, attacks, seed, mutation, selector,
               constraint, metrics)

__all__ = [
    # Subpackage namespaces
    "datasets", "defenses", "attacks", "seed", "mutation", "selector",
    "constraint", "metrics",
    # Victim interface
    "Victim",
    # LLM hierarchy
    "UnifiedLLM", "APILLM", "LocalLLM",
    "KNOWN_MODELS", "ModelSpec", "parse_json_response",
    "LogprobNotSupportedError", "LocalOnlyError",
    "METIS_BASE_IR", "METIS_BASE_GLOBAL",
    # Target (Victim wrapping UnifiedLLM)
    "TargetLLM", "make_target",
    # Defenses
    "DefendedVictim",
    "InstructionalDefense", "ReminderDefense", "SandwichDefense",
    "SpotlightDefense", "CompositeDefense",
    "SecAlignDefense", "StruQDefense",

    # Attacker Base
    "BaseAttacker", "StaticAttacker", "AdaptiveAttacker", "JudgeGuidedAttacker",
    # Mutation operators
    "MutationBase", "GPTFUZZER_MUTATORS", "RENELLM_MUTATORS",
    # Selection policies
    "SelectPolicy", "MCTSExploreSelectPolicy", "SelectBasedOnScores",
    # Constraints
    "ConstraintBase", "DeleteOffTopic", "PerplexityConstraint",
    # Seed / prompt registry
    "SeedBase", "SeedTemplate", "PLACEHOLDER", "TARGET_PLACEHOLDER",
    "load_seed_templates", "render", "sample_population", "ica_demos",
    # Attack classes
    "TAPAttacker", "PAIRAttacker", "RSAttacker", "BeamRSAttacker", "BEASTAttacker",
    "AutoDANAttacker", "GCGAttacker",
    # Static injection (OPI)
    "NaiveAttacker", "EscapeAttacker", "IgnoreAttacker",
    "FakeCompletionAttacker", "CombinedAttacker",
    # Template / iterative attacks (EasyJailbreak ports)
    "DeepInceptionAttacker", "ICAAttacker", "MultilingualAttacker",
    "ReNeLLMAttacker", "GPTFuzzerAttacker",
    # Low-level attack functions
    "run_tap", "TAPResult",
    "run_pair", "PAIRResult",
    "run_adaptive_rs", "run_adaptive_beam", "AdaptiveResult",
    "run_beast", "BEASTResult",
    "run_autodan_ga", "run_autodan_hga", "AutoDANResult",
    "run_gcg", "GCGResult",
    # Static injection (OPI) low-level
    "run_static_injection", "StaticInjectionResult", "create_static_attacker",
    "build_naive_injection", "build_escape_injection", "build_ignore_injection",
    "build_fake_completion_injection", "build_combined_injection",
    # Dataset — carrier
    "Instance", "AttackDataset", "DualVerifiableDataset", "load_dual_verifiable",
    # Dataset — legacy (removed once recipes migrate)
    "IPIScenario", "IPIDataset", "LegacyDualVerifiableDataset",
    # Metrics — primitives
    "Evaluator",
    "check_function_name", "check_exact_function_call", "check_ipi_success",
    "check_user_utility", "resolve_attack_target",
    # Metrics — guidance (1-10) and success (bool)
    "EvaluatorGenerativeGetScore", "EvaluatorIPIGetScore",
    "EvaluatorEditDistanceGetScore",
    "EvaluatorIPISuccess", "EvaluatorUserUtility", "EvaluatorMatch",
    "EvaluatorPrefixExactMatch", "EvaluatorPatternJudge", "EvaluatorKeywordJudge",
    # Metrics — runner
    "ScenarioResult", "EvalResult", "AttackEvaluator",
    # Harness helpers
    "get_target_token", "ipi_early_stopping_condition", "make_scenario_target_fn",
    # Simple API
    "run_attack", "run_experiment", "ExperimentResult",
]


