"""
ipi.attacks — attack implementations.

    TAPAttacker      tap.py       Tree of Attacks with Pruning
    PAIRAttacker     pair.py      Parallel Iterative Refinement
    RSAttacker       adaptive.py  Random Search
    BeamRSAttacker   adaptive.py  Beam Random Search
    BEASTAttacker    beast.py     Beam Search Adversarial Suffix Tokens (local)
    AutoDANAttacker  autodan.py          AutoDAN GA/HGA genetic-algorithm attack (local)
    GCGAttacker      gcg.py              GCG greedy coordinate gradient attack (local)
    NaiveAttacker        static_injection.py  Naive direct injection (static)
    EscapeAttacker       static_injection.py  Escape-character injection (static)
    IgnoreAttacker       static_injection.py  Context-ignoring injection (static)
    FakeCompletionAttacker static_injection.py Fake-completion injection (static)
    CombinedAttacker     static_injection.py  Combined injection (static)
"""
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
    "TAPAttacker",    "TAPResult",    "run_tap",
    "PAIRAttacker",   "PAIRResult",   "run_pair",
    "RSAttacker",     "BeamRSAttacker", "AdaptiveResult",
    "run_adaptive_rs", "run_adaptive_beam",
    "BEASTAttacker",  "BEASTResult",  "run_beast",
    "AutoDANAttacker", "AutoDANResult", "run_autodan_ga", "run_autodan_hga",
    "GCGAttacker",    "GCGResult",    "run_gcg",
    # Static injection (OPI)
    "NaiveAttacker", "EscapeAttacker", "IgnoreAttacker",
    "FakeCompletionAttacker", "CombinedAttacker",
    "StaticInjectionResult", "run_static_injection", "create_static_attacker",
    "build_naive_injection", "build_escape_injection", "build_ignore_injection",
    "build_fake_completion_injection", "build_combined_injection",
]
