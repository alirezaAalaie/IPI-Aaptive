"""
IPI Attack Research Package
============================

Clean implementations of five adversarial attacks adapted for
Indirect Prompt Injection (IPI) research.

File layout
-----------
  llm_unified.py  — UnifiedLLM (abstract), APILLM (API), LocalLLM (local HF),
                    KaggleLLM (kaggle_benchmarks) · make_llm dispatches on the model string
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
    deepinception.py · ica.py · multilingual.py · renellm.py · gptfuzzer.py
                  — EasyJailbreak ports
  datasets/       — Instance (the carrier every component speaks) · AttackDataset
                    · DualVerifiableDataset
  channels.py     — ChanneledPrompt: the trusted instruction / untrusted data split,
                    carried with the prompt (StruQ's model) instead of parsed out of it
  harness.py      — the Instance-to-Victim plumbing: make_target_fn · attack_context
                    · resolve_optimization_target · build_channeled_prompt
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
    result    = evaluator.run(DualVerifiableDataset())
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

# ---- Instruction / data channels ----
from .channels import (
    ChanneledPrompt, ChanneledMessages, channels_of,
    PreRenderedMessages, PreRenderedPromptError,
)

# ---- Core LLM interface ----
from .llm_unified import (
    APILLM,
    LocalLLM,
    KaggleLLM,
    UnifiedLLM,
    make_llm,
    KNOWN_MODELS,
    KAGGLE_MODELS,
    KAGGLE_PREFIX,
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
from .attacks.tap import TAPAttacker
from .attacks.pair import PAIRAttacker
from .attacks.adaptive import RSAttacker, BeamRSAttacker
from .attacks import (
    BEASTAttacker, BEASTResult, run_beast,
    AutoDANAttacker, AutoDANResult, run_autodan_ga, run_autodan_hga,
    GCGAttacker, GCGResult, run_gcg,
)
from .attacks.static_injection import (
    NaiveAttacker, EscapeAttacker, IgnoreAttacker,
    FakeCompletionAttacker, CombinedAttacker, create_static_attacker,
    build_naive_injection, build_escape_injection, build_ignore_injection,
    build_fake_completion_injection, build_combined_injection,
)
from .attacks import (
    DeepInceptionAttacker, ICAAttacker, MultilingualAttacker,
    ReNeLLMAttacker, GPTFuzzerAttacker,
)

# ---- Dataset (the carrier) ----
from .datasets import Instance, AttackDataset
from .datasets import DualVerifiableDataset, load_dual_verifiable

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

# ---- Harness helpers (Instance -> Victim plumbing) ----
from .harness import (
    make_target_fn, attack_context, resolve_optimization_target,
    build_channeled_prompt, build_victim_messages,
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
    "UnifiedLLM", "APILLM", "LocalLLM", "KaggleLLM", "make_llm",
    "KNOWN_MODELS", "KAGGLE_MODELS", "KAGGLE_PREFIX", "ModelSpec", "parse_json_response",
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
    # Low-level attack functions. Only the white-box three keep one: they are a
    # documented standalone (non-scenario) entry point, and the structural swap that
    # would remove them is deferred in docs/next-session.md 4a.
    "run_beast", "BEASTResult",
    "run_autodan_ga", "run_autodan_hga", "AutoDANResult",
    "run_gcg", "GCGResult",
    # Static injection (OPI) low-level
    "create_static_attacker",
    "build_naive_injection", "build_escape_injection", "build_ignore_injection",
    "build_fake_completion_injection", "build_combined_injection",
    # Dataset — the carrier
    "Instance", "AttackDataset", "DualVerifiableDataset", "load_dual_verifiable",
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
    # Channels
    "ChanneledPrompt", "ChanneledMessages", "channels_of",
    "PreRenderedMessages", "PreRenderedPromptError",
    # Harness helpers
    "make_target_fn", "attack_context", "resolve_optimization_target",
    "build_channeled_prompt", "build_victim_messages",
    # Simple API
]


