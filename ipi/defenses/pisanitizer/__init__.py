"""
PISanitizer Defense Package.

"PISanitizer: Preventing Prompt Injection to Long-Context LLMs via Prompt
Sanitization" (Geng, Wang, Yin, Cheng, Chen, Jia — arXiv:2511.10720).
Upstream: ``code/defense/PISanitizer-main/``.

Attention-guided *prevention*: anchor the model into instruction-following
mode, read which context tokens the first generated token attends to, and
delete the strongest span. Requires scipy — ``pip install ipi-adaptive[pisanitizer]``.
"""
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_SANITIZER_MODEL,
    LLAMA3_DELIMITERS,
    MAX_ITERATIONS,
    SANITIZATION_INSTRUCTIONS,
    SIGNAL_MODES,
    resolve_config,
)
from .defense import PISanitizerDefense
from .peaks import group_peaks
from .sanitizer import PISanitizer, RemovedSpan, SanitizationTrace

__all__ = [
    "PISanitizer",
    "PISanitizerDefense",
    "SanitizationTrace",
    "RemovedSpan",
    "group_peaks",
    "resolve_config",
    "DEFAULT_CONFIG",
    "DEFAULT_SANITIZER_MODEL",
    "LLAMA3_DELIMITERS",
    "MAX_ITERATIONS",
    "SANITIZATION_INSTRUCTIONS",
    "SIGNAL_MODES",
]
