"""
DefensiveToken Defense Package.

"Defending Against Prompt Injection With a Few DefensiveTokens"
(Chen, Wang, Carlini, Sitawarin, Wagner — AISec 2025).
Upstream: ``code/defense/DefensiveToken-main/``.
"""
from .build import apply_defensive_tokens, build_defensive_token_model
from .config import (
    CHAT_TEMPLATES,
    DEFENSIVE_TOKEN_NAMES,
    NUM_DEFENSIVE_TOKENS,
    SUPPORTED_MODELS,
    load_defensive_tokens,
    resolve_defensive_tokens_path,
    resolve_model_key,
)
from .defense import DefensiveTokenDefense

__all__ = [
    "DefensiveTokenDefense",
    "apply_defensive_tokens",
    "build_defensive_token_model",
    "load_defensive_tokens",
    "resolve_defensive_tokens_path",
    "resolve_model_key",
    "CHAT_TEMPLATES",
    "DEFENSIVE_TOKEN_NAMES",
    "NUM_DEFENSIVE_TOKENS",
    "SUPPORTED_MODELS",
]
