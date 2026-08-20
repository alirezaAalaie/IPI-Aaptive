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
from .tap      import TAPAttacker
from .pair     import PAIRAttacker
from .adaptive import RSAttacker, BeamRSAttacker
from .static_injection import (
    NaiveAttacker, EscapeAttacker, IgnoreAttacker,
    FakeCompletionAttacker, CombinedAttacker, create_static_attacker,
    build_naive_injection, build_escape_injection, build_ignore_injection,
    build_fake_completion_injection, build_combined_injection,
)
# Shared building blocks are their own component families now — import the seed/prompt
# registry from ``ipi.seed`` and the mutation operators from ``ipi.mutation``, not here.
# Template / static attacks ported from EasyJailbreak (API-compatible, no torch)
from .deepinception import DeepInceptionAttacker
from .ica          import ICAAttacker
from .multilingual import MultilingualAttacker
# Iterative attacks ported from EasyJailbreak (require an attacker LLM, no torch)
from .renellm      import ReNeLLMAttacker
from .gptfuzzer    import GPTFuzzerAttacker

# Optional local white-box attacks. BEAST and GCG import ``mutation.gradient``, which
# imports torch, so without it these names are None and a notebook finds out at import
# rather than mid-run.
try:
    from .beast import BEASTAttacker, BEASTResult, run_beast
except ImportError:
    BEASTAttacker = BEASTResult = run_beast = None

# AutoDAN survives a torch-free import since the swap onto the component families: its
# only torch is inside ``ReferenceLossSelector``, which imports lazily. It still needs a
# local victim — ``requires_local_target()`` is True and ``run_autodan_*`` raises on a
# non-local backend — so the guard stays, it just no longer fires here.
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
    "TAPAttacker",    "PAIRAttacker",
    # Non-judge adaptive attacks
    "RSAttacker",     "BeamRSAttacker",
    "BEASTAttacker",  "BEASTResult",  "run_beast",
    "AutoDANAttacker", "AutoDANResult", "run_autodan_ga", "run_autodan_hga",
    "GCGAttacker",    "GCGResult",    "run_gcg",
    # Static attacks (OPI)
    "NaiveAttacker", "EscapeAttacker", "IgnoreAttacker",
    "FakeCompletionAttacker", "CombinedAttacker",
    "create_static_attacker",
    "build_naive_injection", "build_escape_injection", "build_ignore_injection",
    "build_fake_completion_injection", "build_combined_injection",
    # Template / static attacks (EasyJailbreak ports)
    "DeepInceptionAttacker", "ICAAttacker", "MultilingualAttacker",
    # Iterative attacks (EasyJailbreak ports)
    "ReNeLLMAttacker",
    "GPTFuzzerAttacker",
]
