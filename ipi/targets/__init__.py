"""
ipi.targets — Victim models, LLM wrappers, and target builders.
"""
from ..victim import Victim
from ..target import TargetLLM, make_target
from ..llm_unified import UnifiedLLM, APILLM, LocalLLM

__all__ = [
    "Victim",
    "TargetLLM",
    "make_target",
    "UnifiedLLM",
    "APILLM",
    "LocalLLM",
]
