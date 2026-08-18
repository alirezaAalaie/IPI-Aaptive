"""
ipi.attacks — attack implementations.

Class Hierarchy
---------------
    BaseAttacker (ABC)
    ├── StaticAttacker (ABC, 1-query deterministic template attacks)
    │   ├── NaiveAttacker           (static_injection.py)
    │   ├── EscapeAttacker          (static_injection.py)
    │   ├── IgnoreAttacker          (static_injection.py)
    │   ├── FakeCompletionAttacker  (static_injection.py)
    │   └── CombinedAttacker        (static_injection.py)
    │
    └── AdaptiveAttacker (ABC, iterative search/optimization attacks)
        ├── JudgeGuidedAttacker (requires Judge)
        │   ├── TAPAttacker         (tap.py)
        │   └── PAIRAttacker        (pair.py)
        │
        └── Logprob / Gradient / Search Attacks (Judge optional)
            ├── RSAttacker          (adaptive.py)
            ├── BeamRSAttacker      (adaptive.py)
            ├── BEASTAttacker       (beast.py — local)
            ├── AutoDANAttacker     (autodan.py — local)
            └── GCGAttacker         (gcg.py — local)
"""
from ..attacker import BaseAttacker, StaticAttacker, AdaptiveAttacker, JudgeGuidedAttacker
from .tap      import TAPAttacker,    TAPResult,      run_tap
from .pair     import PAIRAttacker,   PAIRResult,     run_pair
from .adaptive import RSAttacker, BeamRSAttacker, AdaptiveResult, run_adaptive_rs, run_adaptive_beam
from .static_injection import (
    NaiveAttacker, EscapeAttacker, IgnoreAttacker,
    FakeCompletionAttacker, CombinedAttacker,
    StaticInjectionResult, run_static_injection, create_static_attacker,
    build_naive_injection, build_escape_injection, build_ignore_injection,
    build_fake_completion_injection, build_combined_injection,
)
# Shared building blocks are their own component families now — import the seed/prompt
# registry from ``ipi.seed`` and the mutation operators from ``ipi.mutation``, not here.
# Template / static attacks ported from EasyJailbreak (API-compatible, no torch)
from .deepinception import DeepInceptionAttacker, DeepInceptionResult, run_deepinception
from .ica          import ICAAttacker,          ICAResult,          run_ica
from .multilingual import MultilingualAttacker, MultilingualResult, run_multilingual
# Iterative attacks ported from EasyJailbreak (require an attacker LLM, no torch)
from .renellm      import ReNeLLMAttacker,   ReNeLLMResult,   run_renellm
from .gptfuzzer    import GPTFuzzerAttacker, GPTFuzzerResult, run_gptfuzzer

# Optional local white-box attacks (require torch)
try:
    from .beast import BEASTAttacker, BEASTResult, run_beast
except ImportError:
    BEASTAttacker = BEASTResult = run_beast = None

try:
    from .autodan import AutoDANAttacker, AutoDANResult, run_autodan_ga, run_autodan_hga
except ImportError:
    AutoDANAttacker = AutoDANResult = run_autodan_ga = run_autodan_hga = None

try:
    from .gcg import GCGAttacker, GCGResult, run_gcg
except ImportError:
    GCGAttacker = GCGResult = run_gcg = None


__all__ = [
    # Base hierarchy
    "BaseAttacker", "StaticAttacker", "AdaptiveAttacker", "JudgeGuidedAttacker",
    # Judge-guided adaptive attacks
    "TAPAttacker",    "TAPResult",    "run_tap",
    "PAIRAttacker",   "PAIRResult",   "run_pair",
    # Non-judge adaptive attacks
    "RSAttacker",     "BeamRSAttacker", "AdaptiveResult",
    "run_adaptive_rs", "run_adaptive_beam",
    "BEASTAttacker",  "BEASTResult",  "run_beast",
    "AutoDANAttacker", "AutoDANResult", "run_autodan_ga", "run_autodan_hga",
    "GCGAttacker",    "GCGResult",    "run_gcg",
    # Static attacks (OPI)
    "NaiveAttacker", "EscapeAttacker", "IgnoreAttacker",
    "FakeCompletionAttacker", "CombinedAttacker",
    "StaticInjectionResult", "run_static_injection", "create_static_attacker",
    "build_naive_injection", "build_escape_injection", "build_ignore_injection",
    "build_fake_completion_injection", "build_combined_injection",
    # Template / static attacks (EasyJailbreak ports)
    "DeepInceptionAttacker", "DeepInceptionResult", "run_deepinception",
    "ICAAttacker", "ICAResult", "run_ica",
    "MultilingualAttacker", "MultilingualResult", "run_multilingual",
    # Iterative attacks (EasyJailbreak ports)
    "ReNeLLMAttacker", "ReNeLLMResult", "run_renellm",
    "GPTFuzzerAttacker", "GPTFuzzerResult", "run_gptfuzzer",
]
