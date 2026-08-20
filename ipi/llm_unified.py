"""
Unified LLM interface for IPI attack research.

UnifiedLLM — abstract base class. Holds shared config and concrete helpers
             (__call__, get_logprob, get_response, chat, __repr__).
             Subclass with APILLM or LocalLLM.

APILLM     — API-backed model (litellm / OpenAI-compat / Metis proxy / OpenRouter).
             Supports: openai, anthropic, google/gemini, deepseek, openrouter,
                       metis_openai, metis_deepseek, metis_gemini,
                       and any raw litellm model string.
             Logprob support: openai, metis_openai, deepseek, metis_deepseek, openrouter.
             Use: APILLM("gpt-4o-mini", system_prompt="...")

LocalLLM   — Local HuggingFace model. Full logits/logprob access.
             Required for: BEAST, logprob-based RS on any model.
             Use: LocalLLM("lmsys/vicuna-7b-v1.5")

KaggleLLM  — Model served by the Kaggle Benchmarks runtime (``kbench.llms[...]``).
             Only importable inside a Kaggle Benchmarks notebook, and only usable
             from inside a running ``@kbench.task``. No logprobs, no white-box access.
             Use: KaggleLLM("kaggle/google/gemini-2.5-flash")

make_llm   — The dispatcher. A ``kaggle/`` prefix picks KaggleLLM, backend="local"
             picks LocalLLM, everything else APILLM; a non-str passes through. Every
             seam that takes a model *string* (attacker LLM, *GetScore judge, TAP's
             on-topic model, Multilingual's translator, make_target) goes through it,
             so "kaggle/<id>" works in all of them.

Usage
-----
    from ipi.llm_unified import APILLM, LocalLLM

    # API target / attacker
    llm = APILLM("gpt-4o-mini", system_prompt="You are an email agent.")
    response = llm("Summarize my inbox.")          # target_fn mode (str in)
    response = llm([{"role": "user", ...}])        # attacker mode (list in)

    # API logprobs (RS attack)
    logprobs = llm.get_first_token_logprobs([{"role": "user", "content": msg}])

    # Local target (BEAST / RS)
    local = LocalLLM("lmsys/vicuna-7b-v1.5")
    logits, tokens = local.generate_n_tokens_batch([prompt_ids], max_gen_len=10)

    # Kaggle Benchmarks target / judge (inside a @kbench.task)
    from ipi.llm_unified import make_llm
    llm = make_llm("kaggle/google/gemini-2.5-flash", system_prompt="You are an email agent.")

Model registry
--------------
    APILLM.supported_models()   → dict[str, ModelSpec] of all known API model IDs
    LocalLLM.supported_models()  → str describing accepted format
    KaggleLLM.supported_models() → live kbench.llms keys (KAGGLE_MODELS off-Kaggle)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import inspect
import itertools
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

METIS_BASE_IR     = "https://api.metisai.ir"      # access from inside Iran
METIS_BASE_GLOBAL = "https://api.tapsage.com"     # access from Colab, Kaggle, abroad

_METIS_OPENAI_PATH   = "/openai/v1"
_METIS_DEEPSEEK_PATH = "/deepseek/v1"

_LITELLM_PREFIXES: dict[str, str] = {
    "google":   "gemini/",
    "deepseek": "deepseek/",
    # openai, anthropic, litellm → no prefix
}


@dataclass(frozen=True)
class ModelSpec:
    """Provider + model-ID pair. provider is one of: openai | anthropic | google |
    deepseek | metis_openai | metis_deepseek | metis_gemini | openrouter | litellm | local"""
    provider: str
    model_id: str


KNOWN_MODELS: dict[str, ModelSpec] = {
    # ---- Google (Gemini API) ----
    "gemini-2.5-flash-lite":  ModelSpec("google",   "gemini-2.5-flash-lite"),
    "gemma-3-27b-it":         ModelSpec("google",   "gemma-3-27b-it"),
    "gemini-2.0-flash":       ModelSpec("google",   "gemini-2.0-flash"),
    "gemini-2.5-pro":         ModelSpec("google",   "gemini-2.5-pro"),
    # ---- DeepSeek (direct) ----
    "deepseek-v4-flash":      ModelSpec("deepseek", "deepseek-v4-flash"),
    "deepseek-chat":          ModelSpec("deepseek", "deepseek-chat"),
    "deepseek-reasoner":      ModelSpec("deepseek", "deepseek-reasoner"),
    # ---- OpenAI (direct) ----
    "gpt-5-nano":             ModelSpec("openai",   "gpt-5-nano"),
    "gpt-4o-mini":            ModelSpec("openai",   "gpt-4o-mini"),
    "gpt-4.1-nano":           ModelSpec("openai",   "gpt-4.1-nano"),
    "gpt-4o":                 ModelSpec("openai",   "gpt-4o"),
    "gpt-4.1":                ModelSpec("openai",   "gpt-4.1"),
    # ---- Anthropic (direct) ----
    "claude-sonnet-4-6":      ModelSpec("anthropic", "claude-sonnet-4-6"),
    "claude-haiku-4-5":       ModelSpec("anthropic", "claude-haiku-4-5-20251001"),
    "claude-opus-4-6":        ModelSpec("anthropic", "claude-opus-4-6"),
    # ---- Metis → OpenAI ----
    "metis/gpt-4o":           ModelSpec("metis_openai", "gpt-4o"),
    "metis/gpt-4o-mini":      ModelSpec("metis_openai", "gpt-4o-mini"),
    "metis/gpt-4.1-nano":     ModelSpec("metis_openai", "gpt-4.1-nano"),
    "metis/gpt-5-nano":       ModelSpec("metis_openai", "gpt-5-nano"),
    # ---- Metis → DeepSeek ----
    "metis/deepseek-chat":        ModelSpec("metis_deepseek", "deepseek-chat"),
    "metis/deepseek-v4-flash":    ModelSpec("metis_deepseek", "deepseek-v4-flash"),
    "metis/deepseek-reasoner":    ModelSpec("metis_deepseek", "deepseek-reasoner"),
    # ---- Metis → Gemini ----
    "metis/gemini-2.5-pro":        ModelSpec("metis_gemini", "gemini-2.5-pro"),
    "metis/gemini-2.0-flash":      ModelSpec("metis_gemini", "gemini-2.0-flash"),
    "metis/gemini-2.5-flash-lite": ModelSpec("metis_gemini", "gemini-2.5-flash-lite"),
    # ---- OpenRouter (Direct & Free Models) ----
    "google/gemma-4-26b-a4b-it:free":            ModelSpec("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "openai/gpt-oss-20b:free":                   ModelSpec("openrouter", "openai/gpt-oss-20b:free"),
    "google/gemma-4-31b-it:free":                ModelSpec("openrouter", "google/gemma-4-31b-it:free"),
    "openrouter/google/gemma-4-26b-a4b-it:free": ModelSpec("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "openrouter/openai/gpt-oss-20b:free":        ModelSpec("openrouter", "openai/gpt-oss-20b:free"),
    "openrouter/google/gemma-4-31b-it:free":     ModelSpec("openrouter", "google/gemma-4-31b-it:free"),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LogprobNotSupportedError(RuntimeError):
    """Raised when logprob retrieval is attempted against a provider without logprob API."""


class LocalOnlyError(RuntimeError):
    """Raised when a local-only method is called on APILLM."""


_LOGPROB_API_PROVIDERS = {"openai", "metis_openai", "deepseek", "metis_deepseek", "openrouter"}


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class UnifiedLLM(ABC):
    """
    Abstract base for all LLM wrappers in this package.

    Holds shared configuration (model_name, system_prompt, temperature, etc.)
    and provides concrete helpers that delegate to the abstract
    ``generate`` and ``get_first_token_logprobs`` methods.

    Subclass with:
      APILLM   — for API-backed models (OpenAI, Anthropic, Google, DeepSeek, Metis).
      LocalLLM — for locally loaded HuggingFace models (BEAST, logprob-based RS).
    """

    backend: ClassVar[str] = "api"   # overridden by APILLM ("api") and LocalLLM ("local")

    _ENV_VARS: ClassVar[dict[str, str]] = {
        "openai":         "OPENAI_API_KEY",
        "anthropic":      "ANTHROPIC_API_KEY",
        "google":         "GOOGLE_API_KEY",
        "deepseek":       "DEEPSEEK_API_KEY",
        "metis_openai":   "METIS_API_KEY",
        "metis_deepseek": "METIS_API_KEY",
        "metis_gemini":   "METIS_API_KEY",
        "openrouter":     "OPENROUTER_API_KEY",
    }

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        api_key: str = "",
        metis_location: str = "ir",
        extra_messages: Optional[list[dict]] = None,
        max_bs: int = 50,
        enable_reasoning: bool = False,
    ):
        self.model_name       = model
        self.system_prompt    = system_prompt
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self.top_p            = top_p
        self.top_k            = top_k
        self.max_bs           = max_bs
        self.extra_messages   = extra_messages or []
        self.enable_reasoning = enable_reasoning

        # Resolve spec (uses self.backend class variable)
        self._spec = KNOWN_MODELS.get(model)
        if self._spec is None:
            if model.startswith("openrouter/"):
                self._spec = ModelSpec("openrouter", model.removeprefix("openrouter/"))
            else:
                self._spec = ModelSpec(
                    "litellm" if self.backend == "api" else "local",
                    model,
                )
        self._metis_base = METIS_BASE_IR if metis_location == "ir" else METIS_BASE_GLOBAL
        self._api_key    = api_key or self._resolve_api_key(self._spec.provider)

        # Token usage counters
        self.n_input_tokens  = 0
        self.n_output_tokens = 0
        self.n_input_chars   = 0
        self.n_output_chars  = 0

    def _resolve_api_key(self, provider: str) -> str:
        env_var = self._ENV_VARS.get(provider, "")
        return os.environ.get(env_var, "") if env_var else ""

    # ------------------------------------------------------------------
    # Abstract interface  (implement in APILLM / LocalLLM)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def supported_models(cls):
        """
        Report which models this class supports.

        APILLM:   returns dict[str, ModelSpec] — the full KNOWN_MODELS registry.
        LocalLLM: returns str — description of accepted model ID format.
        """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate a response from a messages list."""

    @abstractmethod
    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        """
        Return log-probabilities for the top-n most likely first generated tokens.

        Raises LogprobNotSupportedError for providers without logprob APIs
        (anthropic, google, metis_gemini when using APILLM).
        """

    # ------------------------------------------------------------------
    # Concrete helpers  (shared by all subclasses)
    # ------------------------------------------------------------------

    def __call__(self, messages_or_injection) -> str:
        """
        Dual-mode callable:

        • str  input → target_fn mode: wraps injection as a user message.
          Compatible with TAP/PAIR ``target_fn: Callable[[str], str]``.

        • list input → attacker/judge mode: messages list forwarded to generate().
          Compatible with the attacker/judge call signature.
        """
        if isinstance(messages_or_injection, str):
            messages: list[dict] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(self.extra_messages)
            messages.append({"role": "user", "content": messages_or_injection})
            return self.generate(messages)
        if isinstance(messages_or_injection, list):
            return self.generate(messages_or_injection)
        raise TypeError(
            f"Expected str (injection) or list[dict] (messages), "
            f"got {type(messages_or_injection).__name__}"
        )

    def chat(self, messages: list[dict]) -> str:
        """Alias for generate(). Backward-compat."""
        return self.generate(messages)

    def get_logprob(self, messages: list[dict], target_token: str) -> float:
        """
        Log-probability of ``target_token`` as the first generated token.
        Checks both 'Token' and ' Token' forms. Returns -inf if not in top-k.
        """
        return _extract_logprob(self.get_first_token_logprobs(messages), target_token)

    def get_response(
        self,
        prompts: list[str],
        max_n_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> list[dict]:
        """
        Compatible with the original adaptive attacks ``TargetLM.get_response()``.

        Args:
            prompts:      List of raw text strings (each wrapped as a user message).
            max_n_tokens: Max response tokens. Default: self.max_tokens.
            temperature:  Sampling temperature. Default: self.temperature.
            top_p:        Nucleus sampling p. Default: self.top_p.

        Returns:
            List of dicts (one per prompt):
              { 'text': str, 'logprobs': [dict], 'n_input_tokens': int, 'n_output_tokens': int }
            'logprobs' is [{}] when logprobs are unavailable for the provider.
        """
        mt  = max_n_tokens if max_n_tokens is not None else self.max_tokens
        tmp = temperature  if temperature  is not None else self.temperature
        tp  = top_p        if top_p        is not None else self.top_p

        results = []
        for prompt in prompts:
            messages: list[dict] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Best-effort logprobs (first token, cheap)
            logprobs_dict: dict[str, float] = {}
            try:
                logprobs_dict = self.get_first_token_logprobs(messages)
            except LogprobNotSupportedError:
                pass   # expected for anthropic/google APILLM
            except Exception as e:
                log.debug("get_response: logprob fetch failed: %s", e)

            text = self.generate(messages, max_tokens=mt, temperature=tmp, top_p=tp)
            n_in  = sum(len(m.get("content", "")) for m in messages) // 4
            n_out = len(text) // 4

            self.n_input_tokens  += n_in
            self.n_output_tokens += n_out
            self.n_input_chars   += len(prompt)
            self.n_output_chars  += len(text)

            results.append({
                "text":            text,
                "logprobs":        [logprobs_dict],
                "n_input_tokens":  n_in,
                "n_output_tokens": n_out,
            })
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={self.model_name!r}, "
            f"provider={self._spec.provider!r}, temperature={self.temperature})"
        )


# ---------------------------------------------------------------------------
# APILLM — API-backed model
# ---------------------------------------------------------------------------

class APILLM(UnifiedLLM):
    """
    API-backed LLM using litellm or direct OpenAI-compatible SDK.

    Supports all providers in KNOWN_MODELS plus any raw litellm model string.
    Logprob access: openai, metis_openai, deepseek, metis_deepseek.
    For anthropic / google / metis_gemini, use LocalLLM for logprob-based attacks.
    """

    backend: ClassVar[str] = "api"

    @classmethod
    def supported_models(cls) -> dict[str, ModelSpec]:
        """Return the full registry of known API models."""
        return KNOWN_MODELS

    # --- Abstract method implementations ---

    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        mt  = max_tokens  if max_tokens  is not None else self.max_tokens
        tmp = temperature if temperature is not None else self.temperature
        tp  = top_p       if top_p       is not None else self.top_p
        return self._api_generate(messages, mt, tmp, tp)

    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        return self._api_first_token_logprobs(messages, n_top)

    # --- Private: litellm / OpenAI-compat generation ---

    def _litellm_model_str(self) -> str:
        provider = self._spec.provider
        model    = self._spec.model_id
        prefix   = _LITELLM_PREFIXES.get(provider, "")
        if prefix and not model.startswith(prefix):
            return f"{prefix}{model}"
        return model

    def _api_generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        provider = self._spec.provider
        if provider in ("metis_openai", "metis_deepseek", "openrouter"):
            return self._openai_compat_generate(messages, max_tokens, temperature, top_p)
        if provider == "metis_gemini":
            return self._metis_gemini_generate(messages, max_tokens, temperature)

        import litellm
        kwargs: dict = dict(
            model=self._litellm_model_str(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        if self._api_key:
            kwargs["api_key"] = self._api_key
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content.strip()

    def _openai_compat_generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        from openai import OpenAI
        provider = self._spec.provider
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "metis_openai":
            base_url = self._metis_base + _METIS_OPENAI_PATH
        else:
            base_url = self._metis_base + _METIS_DEEPSEEK_PATH

        client = OpenAI(api_key=self._api_key or "dummy", base_url=base_url)
        kwargs: dict = dict(
            model=self._spec.model_id,
            messages=messages,   # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        if provider == "openrouter" and self.enable_reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def _metis_gemini_generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for metis_gemini. "
                "Install: pip install google-generativeai"
            ) from exc
        endpoint = self._metis_base.removeprefix("https://").removeprefix("http://")
        genai.configure(api_key=self._api_key, client_options={"api_endpoint": endpoint})
        system_text = ""
        conv: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                role = "user" if m["role"] == "user" else "model"
                conv.append({"role": role, "parts": [m["content"]]})
        model_kwargs: dict = {"model_name": self._spec.model_id}
        if system_text:
            model_kwargs["system_instruction"] = system_text
        model = genai.GenerativeModel(**model_kwargs)
        resp = model.generate_content(
            conv,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return resp.text.strip()

    # --- Private: API logprobs ---

    def _api_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int,
    ) -> dict[str, float]:
        provider = self._spec.provider
        if provider not in _LOGPROB_API_PROVIDERS:
            raise LogprobNotSupportedError(
                f"Provider '{provider}' does not expose logprob APIs. "
                f"Use LocalLLM or one of: {sorted(_LOGPROB_API_PROVIDERS)}."
            )

        from openai import OpenAI
        if provider == "metis_openai":
            base_url = self._metis_base + _METIS_OPENAI_PATH
        elif provider == "metis_deepseek":
            base_url = self._metis_base + _METIS_DEEPSEEK_PATH
        elif provider == "deepseek":
            base_url = "https://api.deepseek.com/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.openai.com/v1"

        client = OpenAI(api_key=self._api_key or "dummy", base_url=base_url)
        resp = client.chat.completions.create(
            model=self._spec.model_id,
            messages=messages,   # type: ignore[arg-type]
            max_tokens=1,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=min(n_top, 20),
        )
        if (
            resp.choices[0].logprobs
            and resp.choices[0].logprobs.content
        ):
            return {
                item.token: item.logprob
                for item in resp.choices[0].logprobs.content[0].top_logprobs
            }
        return {}


# ---------------------------------------------------------------------------
# LocalLLM — local HuggingFace model
# ---------------------------------------------------------------------------

class LocalLLM(UnifiedLLM):
    """
    Local HuggingFace model (AutoModelForCausalLM).

    Provides full logit / logprob access. Required for BEAST and recommended
    for logprob-based RS experiments on any model.

    Args:
        model:       HuggingFace model ID or absolute path to weights.
        device_map:  HuggingFace device_map. Default "auto".
        torch_dtype: HuggingFace torch_dtype. Default float16 on CUDA.
        max_bs:      Max batch size for BEAST batched generation. Default 50.
        (all other args inherited from UnifiedLLM)
    """

    backend: ClassVar[str] = "local"

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        api_key: str = "",
        metis_location: str = "ir",
        extra_messages: Optional[list[dict]] = None,
        max_bs: int = 50,
        device_map: str = "auto",
        torch_dtype=None,
        adapter_path: Optional[str] = None,
    ):
        self.adapter_path = adapter_path
        self.adapter_loaded = False   # set True in _init_local iff a PEFT adapter actually attached
        super().__init__(
            model, system_prompt, temperature, max_tokens, top_p, top_k,
            api_key, metis_location, extra_messages, max_bs,
        )
        self._tokenizer_obj = None
        self._hf_model_obj  = None
        self._device        = None
        self._init_local(device_map, torch_dtype)


    @classmethod
    def supported_models(cls) -> str:
        return (
            "Any HuggingFace model ID or local path. "
            "Examples: 'lmsys/vicuna-7b-v1.5', 'meta-llama/Llama-2-7b-chat-hf', "
            "'/path/to/local/weights'."
        )

    # --- Abstract method implementations ---

    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        mt  = max_tokens  if max_tokens  is not None else self.max_tokens
        tmp = temperature if temperature is not None else self.temperature
        tp  = top_p       if top_p       is not None else self.top_p
        return self._local_generate(messages, mt, tmp, tp)

    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        return self._local_first_token_logprobs(messages, n_top)

    # --- Properties ---

    @property
    def tokenizer(self):
        if self._tokenizer_obj is None:
            raise LocalOnlyError("tokenizer is only available after local model init.")
        return self._tokenizer_obj

    @property
    def hf_model(self):
        if self._hf_model_obj is None:
            raise LocalOnlyError("hf_model is only available after local model init.")
        return self._hf_model_obj

    # --- Local-only utilities ---

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize text to token IDs."""
        return self._tokenizer_obj.encode(text, add_special_tokens=add_special_tokens)

    def detokenize(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer_obj.decode(token_ids, skip_special_tokens=True)

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ):
        """Apply the model's chat template. Returns token-id list if tokenize=True."""
        return self._tokenizer_obj.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        )

    # --- BEAST interface ---

    def generate_n_tokens_batch(
        self,
        prompt_tokens,
        max_gen_len: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        """
        Batched autoregressive generation. Returns (logits, tokens).
        Mirrors ``arutils.AutoRegressor.generate_n_tokens_batch()``.
        """
        import torch
        if temperature is None: temperature = self.temperature
        if top_p is None:       top_p = self.top_p
        if top_k is None:       top_k = self.top_k

        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.tensor(prompt_tokens)
        prompt_tokens = prompt_tokens.to(self._device)

        if max_gen_len == 0:
            return None, prompt_tokens

        gen_cfg = copy.deepcopy(self._hf_model_obj.generation_config)
        if temperature is not None:
            self._hf_model_obj.generation_config.temperature = temperature
            self._hf_model_obj.generation_config.do_sample = temperature > 0
        if top_p is not None:
            self._hf_model_obj.generation_config.top_p = top_p
        if top_k is not None:
            self._hf_model_obj.generation_config.top_k = top_k

        prompt_len = prompt_tokens.shape[1]
        out = self._hf_model_obj.generate(
            prompt_tokens,
            max_length=max_gen_len + prompt_len,
            min_length=max_gen_len + prompt_len,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=self._hf_model_obj.generation_config,
        )
        self._hf_model_obj.generation_config = gen_cfg

        tokens = out.sequences
        logits = torch.stack(out.scores).permute(1, 0, 2)  # (batch, steps, vocab)
        return logits, tokens

    def attack_objective_targeted(
        self,
        tokens,
        target_str: str,
    ) -> np.ndarray:
        """
        BEAST targeted attack objective: ``-perplexity(target_str | tokens)``.
        Higher (less negative) = target string more likely given tokens.
        """
        import torch
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        tokens = tokens.to(self._device)

        scores = np.zeros(len(tokens))
        target_ids = self._tokenizer_obj.encode(
            target_str, return_tensors="pt", add_special_tokens=False
        ).to(self._device)

        rows = [
            torch.cat([tokens[ii:ii+1], target_ids], dim=1)
            for ii in range(len(tokens))
        ]
        tokens_cat = torch.cat(rows, dim=0).to(tokens.dtype)

        if tokens.shape == tokens_cat.shape:
            bos_id = self._tokenizer_obj.encode(
                self._tokenizer_obj.bos_token or "", add_special_tokens=False
            )
            if bos_id:
                bos = torch.tensor([[bos_id[0]]] * len(tokens_cat)).to(self._device)
                tokens_cat = torch.cat([bos, tokens_cat], dim=1).to(tokens_cat.dtype)
            scores += -(
                self.perplexity(tokens_cat[:, :1], tokens_cat)
                .detach().cpu().numpy()
            )
        else:
            scores += -(
                self.perplexity(tokens, tokens_cat)
                .detach().cpu().numpy()
            )

        return scores

    def perplexity(self, x1, x2):
        """
        Compute sequence perplexity of x2[len(x1):] given x1.
        Both 2D int tensors (batch, seq_len). Returns float tensor (batch,).
        """
        import torch
        import torch.nn.functional as F
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2)
        x2 = x2.to(self._device)

        with torch.no_grad():
            output = self._hf_model_obj(
                input_ids=x2, use_cache=False, past_key_values=None, return_dict=True,
            )

        logs = None
        for curr_pos in range(len(x1[0]), len(x2[0])):
            log_val = -torch.log(
                torch.softmax(output.logits, dim=-1)[
                    torch.arange(len(output.logits)), curr_pos - 1, x2[:, curr_pos]
                ]
            )
            logs = log_val if logs is None else logs + log_val

        return torch.exp(logs / (len(x2[0]) - len(x1[0])))

    # --- Config fork (share GPU weights, change settings) ---

    def with_config(
        self,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> "LocalLLM":
        """
        Return a shallow copy of this LocalLLM with updated config values.

        The copy **shares** the underlying HuggingFace model and tokenizer objects
        (same GPU tensors, no extra VRAM), so this is the correct way to use one
        loaded model in multiple roles (target / attacker / judge) with different
        system prompts or sampling settings.

        Example::

            base = LocalLLM('lmsys/vicuna-7b-v1.5')          # loads once

            target  = base.with_config(system_prompt=AGENT_SYSTEM_PROMPT,
                                       temperature=0.0, max_tokens=500)
            attacker = base.with_config(temperature=1.0, max_tokens=1024)
            judge    = base.with_config(temperature=0.0, max_tokens=20)
        """
        clone = copy.copy(self)                          # shallow: shares _hf_model_obj
        if system_prompt is not None:
            clone.system_prompt = system_prompt
        if temperature is not None:
            clone.temperature = temperature
        if max_tokens is not None:
            clone.max_tokens = max_tokens
        if top_p is not None:
            clone.top_p = top_p
        if top_k is not None:
            clone.top_k = top_k
        return clone

    # --- Private: init ---

    def _init_local(self, device_map: str, torch_dtype) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        base_model_path = self.model_name
        adapter_dir = self.adapter_path

        # If model_name is a directory containing adapter_config.json, auto-detect base model & adapter
        if os.path.isdir(self.model_name) and os.path.exists(os.path.join(self.model_name, "adapter_config.json")):
            adapter_dir = self.model_name
            try:
                with open(os.path.join(self.model_name, "adapter_config.json")) as f:
                    adapter_cfg = json.load(f)
                    base_model_path = adapter_cfg.get("base_model_name_or_path", self.model_name)
            except Exception as e:
                log.warning("[LocalLLM] Could not parse adapter_config.json: %s", e)

        log.info("[LocalLLM] Loading tokenizer from: %s", base_model_path)
        tokenizer_path = adapter_dir if (adapter_dir and os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json"))) else base_model_path
        self._tokenizer_obj = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False, token=os.getenv("HF_TOKEN"), trust_remote_code=True,
        )
        if self._tokenizer_obj.pad_token is None:
            self._tokenizer_obj.pad_token = (
                self._tokenizer_obj.unk_token or self._tokenizer_obj.eos_token or "[PAD]"
            )
        self._tokenizer_obj.padding_side = "left"

        log.info("[LocalLLM] Loading base model: %s", base_model_path)
        model_obj = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            use_cache=False,
            low_cpu_mem_usage=True,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )

        # Resize token embeddings if tokenizer vocabulary was expanded
        if len(self._tokenizer_obj) != model_obj.config.vocab_size:
            try:
                model_obj.resize_token_embeddings(len(self._tokenizer_obj), mean_resizing=False)
            except TypeError:
                model_obj.resize_token_embeddings(len(self._tokenizer_obj))

        # Attach PEFT adapter if adapter_dir is provided or auto-detected
        if adapter_dir and os.path.exists(adapter_dir):
            try:
                from peft import PeftModel
                log.info("[LocalLLM] Attaching PEFT adapter from: %s", adapter_dir)
                model_obj = PeftModel.from_pretrained(model_obj, adapter_dir)
                self.adapter_loaded = True
            except Exception as e:
                # Non-fatal by design (some callers deliberately fall back to the base
                # model), but this MUST NOT fail silently: a swallowed adapter-load
                # error here means every downstream "defended" eval is actually
                # running the undefended base model while reporting success. Print
                # (not just log.warning — notebooks often don't surface Python
                # logging) and leave self.adapter_loaded False so callers can check
                # `victim._hf_model_obj` / `LocalLLM.adapter_loaded` before trusting
                # results.
                msg = (
                    f"[LocalLLM] ⚠️  PEFT ADAPTER FAILED TO LOAD from {adapter_dir!r}: {e}\n"
                    f"[LocalLLM] ⚠️  Continuing with the BASE model ({base_model_path!r}) "
                    f"UNMODIFIED — any defense/fine-tuning this adapter provides is "
                    f"NOT applied. Check `.adapter_loaded` before trusting results."
                )
                log.warning(msg)
                print(msg)

        self._hf_model_obj = model_obj.eval()
        self._device = next(self._hf_model_obj.parameters()).device
        log.info("[LocalLLM] Loaded on device: %s", self._device)


    # --- Private: local generation ---

    def _build_local_prompt_ids(self, messages) -> list[int]:
        # Accept a plain string — wrap it so callers can do generate("hi")
        if isinstance(messages, str):
            msgs: list[dict] = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": messages})
            messages = msgs

        # Rendering lives in ``render_messages`` so the white-box attacks, which have
        # to tokenize this same prompt themselves, cannot drift from it. It handles
        # all three shapes: the lone {"role": "raw"} turn of the structured-query
        # defenses (StruQ / SecAlign, whose models are fine-tuned on bare Alpaca
        # strings and must NOT be wrapped in a chat template — see
        # ipi/defenses/channels.py:RAW_ROLE), the chat template, and the plain
        # USER/ASSISTANT fallback for tokenizers with no template at all.
        #
        # tokenize=False + a separate encode() is deliberate: tokenize=True returns
        # str on some transformers versions and list[int] on others, which surfaces
        # as a torch.tensor() failure much further downstream.
        text, add_special = render_messages(self._tokenizer_obj, messages)
        return self._tokenizer_obj.encode(text, add_special_tokens=add_special)

    def _local_generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        import torch
        prompt_ids = self._build_local_prompt_ids(messages)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(self._device)

        with torch.no_grad():
            out = self._hf_model_obj.generate(
                prompt_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        new_ids = out[0][len(prompt_ids):]
        return self._tokenizer_obj.decode(new_ids, skip_special_tokens=True).strip()

    def _local_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int,
    ) -> dict[str, float]:
        import torch
        import torch.nn.functional as F

        prompt_ids = self._build_local_prompt_ids(messages)
        x = torch.tensor([prompt_ids], dtype=torch.long).to(self._device)

        with torch.no_grad():
            out = self._hf_model_obj(input_ids=x, return_dict=True)

        logits = out.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(log_probs, k=min(n_top, log_probs.size(0)))
        return {
            self._tokenizer_obj.decode([idx.item()]): val.item()
            for idx, val in zip(top_indices, top_values)
        }


# ---------------------------------------------------------------------------
# KaggleLLM — a model served by the Kaggle Benchmarks runtime
# ---------------------------------------------------------------------------

#: Prefix that routes a model string to ``KaggleLLM`` in ``make_llm`` / ``make_target``.
#: ``"kaggle/google/gemini-2.5-flash"`` → ``kbench.llms["google/gemini-2.5-flash"]``.
KAGGLE_PREFIX = "kaggle/"

#: The ids ``kbench.llms`` exposed when this adapter was written. It is a convenience
#: list only — nothing validates against it. ``KaggleLLM`` resolves against the *live*
#: ``kbench.llms`` keys, so a model added (or retired) by Kaggle needs no edit here.
KAGGLE_MODELS: tuple[str, ...] = (
    "anthropic/claude-haiku-4-5@20251001",
    "anthropic/claude-opus-4-1@20250805",
    "anthropic/claude-opus-4-5@20251101",
    "anthropic/claude-opus-4-6@default",
    "anthropic/claude-opus-4-7@default",
    "anthropic/claude-opus-4-8@default",
    "anthropic/claude-opus-5@default",
    "anthropic/claude-sonnet-4-5@20250929",
    "anthropic/claude-sonnet-4-6@default",
    "anthropic/claude-sonnet-4@20250514",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.1",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.6-flash",
    "google/gemini-3.7-flash",
    "google/gemma-4-26b-a4b",
    "google/gemma-4-31b",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.4-mini-2026-03-17",
    "openai/gpt-5.4-nano-2026-03-17",
    "openai/gpt-5.5-2026-04-23",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-next-80b-a3b-thinking",
    "xai/grok-4.20-0309-non-reasoning",
    "xai/grok-4.20-0309-reasoning",
    "zai/glm-5",
)

_KAGGLE_IMPORT_HINT = (
    "kaggle_benchmarks is only available inside a Kaggle Benchmarks notebook "
    "(open one at https://www.kaggle.com/benchmarks/tasks/new — the package and its "
    "credentials are pre-installed there and cannot be pip-installed elsewhere)."
)

_KAGGLE_CONTEXT_HINT = (
    "kbench.llms[...].prompt() needs an active Chat, which exists only while a "
    "@kbench.task is running. Drive the whole evaluation from inside one task, e.g.\n"
    "    @kbench.task(name='ipi_asr')\n"
    "    def run_ipi(llm):\n"
    "        target = make_target('kaggle/google/gemini-2.5-flash')\n"
    "        ...\n"
    "    run_ipi.run(llm=kbench.llm)"
)


def _kbench_module():
    """Import ``kaggle_benchmarks`` lazily, with a message that says where it lives."""
    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:                            # pragma: no cover - Kaggle-only
        raise ImportError(_KAGGLE_IMPORT_HINT) from exc
    return kbench


#: A trailing ``-YYYY-MM-DD`` release date on a Kaggle model id.
_KAGGLE_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _strip_kaggle_version(key: str) -> str:
    """``openai/gpt-5.4-mini-2026-03-17`` / ``anthropic/claude-opus-5@default`` → base id."""
    return _KAGGLE_DATE_SUFFIX.sub("", key.split("@", 1)[0])


def resolve_kaggle_model_id(name: str, available) -> str:
    """
    Map a model string onto a live ``kbench.llms`` key.

    Kaggle versions its ids three ways: an ``@`` suffix
    (``anthropic/claude-opus-5@default``), a trailing date
    (``openai/gpt-5.4-mini-2026-03-17``), or nothing at all
    (``google/gemini-2.5-flash``). Accepting the un-versioned form means a notebook does
    not have to re-pin every model string when Kaggle rolls a version. An exact key
    always wins; otherwise both version forms are stripped before matching, and if
    several versions share a base, ``@default`` wins, else the highest-sorting one (the
    latest date). Only the version suffix is ever stripped — nothing is prefix-matched,
    so ``openai/gpt-5.4`` can never silently resolve to ``openai/gpt-5.4-mini-…``.
    """
    want = name.removeprefix(KAGGLE_PREFIX)
    keys = list(available)
    if want in keys:
        return want
    by_base: dict[str, list[str]] = {}
    for key in keys:
        by_base.setdefault(_strip_kaggle_version(key), []).append(key)
    hits = by_base.get(_strip_kaggle_version(want), [])
    if len(hits) == 1:
        return hits[0]
    if hits:
        for hit in hits:
            if hit.endswith("@default"):
                return hit
        return sorted(hits)[-1]
    raise ValueError(
        f"Kaggle model {want!r} is not in kbench.llms. Available: {sorted(keys)}"
    )


class KaggleLLM(UnifiedLLM):
    """
    A model served by ``kaggle_benchmarks`` (``kbench.llms[...]``), as a ``UnifiedLLM``.

    Use it anywhere ``APILLM`` goes — target, attacker, judge, translator — by prefixing
    the model id with ``kaggle/``::

        target   = make_target("kaggle/google/gemini-2.5-flash", system_prompt=AGENT_PROMPT)
        judge    = EvaluatorIPIGetScore(model="kaggle/google/gemini-2.5-flash")
        attacker = TAPAttacker(judge=judge, attacker_llm="kaggle/openai/gpt-5.4-mini")

    Or hand it a live object, which is what ``kbench.llm`` (the notebook's own model
    under test) is::

        KaggleLLM(llm=kbench.llm, system_prompt=AGENT_PROMPT)

    Three things it does NOT give you, all of them structural rather than fixable here:

    * **No logprobs.** ``get_first_token_logprobs`` raises ``LogprobNotSupportedError``,
      so RS / Beam-RS cannot run against a Kaggle victim. Attacks that only call
      ``generate`` (TAP, PAIR, the static one-shots, ReNeLLM, GPTFuzzer, …) are fine.
    * **No white-box access.** ``backend`` is ``"api"``; GCG / BEAST / AutoDAN gate it out.
    * **No native system channel** unless ``kbench``'s ``prompt()`` grows one. See
      ``system_mode`` — the default folds the system prompt into the first user turn and
      warns once, because silently dropping it would make a defended victim look
      undefended.

    Every call runs in its own ``kbench.chats.new(...)`` context. That isolation is
    load-bearing: a ``Chat`` accumulates turns, so sharing one across an attack's
    iterations would feed candidate *n* every earlier candidate's transcript.

    Args:
        model:       ``"kaggle/<id>"`` or a bare ``kbench.llms`` id. Optional if ``llm``
                     is given.
        llm:         A live ``kbench.llms[...]`` / ``kbench.llm`` object to wrap directly.
        system_mode: ``"auto"`` (default) uses a native system argument if ``prompt()``
                     accepts one and folds it into the user turn otherwise;
                     ``"native"`` requires the native argument and raises if absent;
                     ``"fold"`` always folds.
        chat_prefix: Name prefix for the per-call chat, so runs are legible in the
                     recorded ``Run``.
        (all other args inherited from ``UnifiedLLM``)
    """

    backend: ClassVar[str] = "api"

    #: ``prompt()`` kwarg names we know how to fill, most-preferred first. Only names the
    #: live signature actually declares are passed — an unknown kwarg is dropped, never
    #: guessed, because ``prompt()`` is Kaggle's API and may change under us.
    _PROMPT_ARG_ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "system":      ("system", "system_prompt", "system_instruction"),
        "temperature": ("temperature",),
        "max_tokens":  ("max_tokens", "max_output_tokens", "max_new_tokens"),
        "top_p":       ("top_p",),
    }

    _chat_counter: ClassVar[itertools.count] = itertools.count()

    def __init__(
        self,
        model: str = "",
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        extra_messages: Optional[list[dict]] = None,
        max_bs: int = 50,
        llm=None,
        system_mode: str = "auto",
        chat_prefix: str = "ipi",
    ):
        if not model and llm is None:
            raise ValueError("KaggleLLM needs either a model id or a live llm= object.")
        if system_mode not in ("auto", "native", "fold"):
            raise ValueError(
                f"system_mode must be 'auto', 'native' or 'fold', got {system_mode!r}")

        name = model or getattr(llm, "name", None) or getattr(llm, "model", None) or str(llm)
        super().__init__(
            model=name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            extra_messages=extra_messages,
            max_bs=max_bs,
        )
        # Resolution against the live registry is deferred to first use so that
        # constructing a KaggleLLM off-Kaggle (docs, smoke checks) does not import.
        self._spec        = ModelSpec("kaggle", name.removeprefix(KAGGLE_PREFIX))
        self._llm_obj     = llm
        self.system_mode  = system_mode
        self.chat_prefix  = chat_prefix
        self._prompt_args: Optional[set[str]] = None
        self._warned: set[str] = set()

    @classmethod
    def supported_models(cls):
        """Live ``kbench.llms`` keys when the runtime is importable, else ``KAGGLE_MODELS``."""
        try:
            return tuple(sorted(_kbench_module().llms.keys()))
        except ImportError:
            return KAGGLE_MODELS

    # --- The wrapped kbench object ---

    @property
    def kaggle_llm(self):
        """The ``kbench.llms`` object, resolved and cached on first use."""
        if self._llm_obj is None:
            kbench = _kbench_module()
            model_id = resolve_kaggle_model_id(self._spec.model_id, kbench.llms.keys())
            self._spec = ModelSpec("kaggle", model_id)
            self._llm_obj = kbench.llms[model_id]
        return self._llm_obj

    # --- Abstract method implementations ---

    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        mt  = max_tokens  if max_tokens  is not None else self.max_tokens
        tmp = temperature if temperature is not None else self.temperature
        tp  = top_p       if top_p       is not None else self.top_p
        return self._kaggle_generate(messages, mt, tmp, tp)

    def get_first_token_logprobs(
        self,
        messages: list[dict],
        n_top: int = 20,
    ) -> dict[str, float]:
        raise LogprobNotSupportedError(
            "kaggle_benchmarks exposes generated text only, not token logprobs, so "
            "RS / Beam-RS cannot search against a Kaggle model. Use a logprob provider "
            f"({sorted(_LOGPROB_API_PROVIDERS)}) or LocalLLM for those attacks."
        )

    # --- Private ---

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("[KaggleLLM] %s", msg)

    def _supported_prompt_args(self) -> set[str]:
        """Names ``prompt()`` explicitly declares. ``**kwargs`` is deliberately ignored —
        a name absorbed by ``**kwargs`` is silently discarded, which is worse than folding."""
        if self._prompt_args is None:
            prompt = self.kaggle_llm.prompt   # resolution errors must propagate, not
            try:                              # be swallowed into "takes no arguments"
                params = inspect.signature(prompt).parameters
                self._prompt_args = {
                    n for n, p in params.items()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                }
            except (TypeError, ValueError):               # pragma: no cover - exotic callables
                self._prompt_args = set()
        return self._prompt_args

    def _arg_for(self, role: str) -> Optional[str]:
        supported = self._supported_prompt_args()
        for name in self._PROMPT_ARG_ALIASES[role]:
            if name in supported:
                return name
        return None

    def _prompt_kwargs(self, system_text: str, max_tokens: int,
                       temperature: float, top_p: float) -> tuple[dict, bool]:
        """Build ``prompt()`` kwargs. Returns ``(kwargs, system_is_native)``."""
        kwargs: dict = {}
        for role, value in (("temperature", temperature),
                            ("max_tokens", max_tokens),
                            ("top_p", top_p)):
            name = self._arg_for(role)
            if name is not None:
                kwargs[name] = value
            else:
                self._warn_once(
                    role,
                    f"kbench prompt() takes no {role} argument — {role}={value!r} is "
                    f"ignored and the platform default applies.")

        native = False
        if system_text:
            name = self._arg_for("system") if self.system_mode != "fold" else None
            if name is not None:
                kwargs[name] = system_text
                native = True
            elif self.system_mode == "native":
                raise RuntimeError(
                    "system_mode='native' but kbench prompt() declares no system argument "
                    f"(it takes {sorted(self._supported_prompt_args())}). Pass "
                    "system_mode='fold' to accept the system prompt being folded into the "
                    "user turn.")
            else:
                self._warn_once(
                    "system",
                    "kbench prompt() takes no system argument — the system prompt is "
                    "folded into the user turn. For a VICTIM that changes the "
                    "trusted/untrusted structure the defense sees; say so when reporting.")
        return kwargs, native

    def _send_turn(self, kbench, role: str, content: str) -> None:
        """Replay one historical turn into the active chat."""
        if role == "assistant":
            send = getattr(self.kaggle_llm, "send", None)
            if callable(send):
                send(content)
                return
            self._warn_once(
                "assistant",
                "kbench LLM object exposes no send() — prior assistant turns are "
                "replayed as labelled user text, so a multi-turn attacker transcript "
                "is not byte-identical to the same transcript on an API provider.")
            kbench.user.send(f"[ASSISTANT]: {content}")
            return
        kbench.user.send(content)

    def _kaggle_generate(self, messages, max_tokens: int,
                         temperature: float, top_p: float) -> str:
        kbench = _kbench_module()
        system_text, history, final_text = _split_chat_turns(messages)
        kwargs, native_system = self._prompt_kwargs(
            system_text, max_tokens, temperature, top_p)
        prompt_text = final_text
        if system_text and not native_system:
            prompt_text = f"{system_text}\n\n{final_text}" if final_text else system_text

        chat_name = f"{self.chat_prefix}-{next(KaggleLLM._chat_counter)}"
        try:
            with kbench.chats.new(chat_name):
                for role, content in history:
                    self._send_turn(kbench, role, content)
                raw = self.kaggle_llm.prompt(prompt_text, **kwargs)
        except (AttributeError, LookupError, RuntimeError) as exc:
            raise RuntimeError(f"{exc}\n\n{_KAGGLE_CONTEXT_HINT}") from exc

        text = raw if isinstance(raw, str) else str(raw)
        self.n_input_chars  += len(prompt_text)
        self.n_output_chars += len(text)
        self.n_input_tokens  += len(prompt_text) // 4
        self.n_output_tokens += len(text) // 4
        return text.strip()


def _split_chat_turns(messages) -> tuple[str, list[tuple[str, str]], str]:
    """
    Split a messages list into ``(system_text, history, final_text)``.

    ``history`` is every non-system turn but the last, as ``(role, content)``; the last
    one becomes the prompt regardless of its role, so an assistant *prefill* (PAIR's
    ``{"improvement": "", "prompt": "``) is carried rather than dropped.
    """
    if isinstance(messages, str):
        return "", [], messages
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    turns = [(m.get("role", "user"), m.get("content", ""))
             for m in messages if m.get("role") != "system"]
    system_text = "\n\n".join(p for p in system_parts if p)
    if not turns:
        return system_text, [], ""
    return system_text, turns[:-1], turns[-1][1]


def run_in_kaggle_task(fn, *args, name: str = "ipi", llm=None, **kwargs):
    """
    Call ``fn(*args, **kwargs)`` inside a ``@kbench.task`` and return its result.

    A ``KaggleLLM`` can only talk while a task is running, so a notebook cell that would
    normally read::

        result = AttackEvaluator(target=target, attacker=tap).run(subset)

    becomes::

        result = run_in_kaggle_task(AttackEvaluator(target=target, attacker=tap).run, subset)

    and the whole attack — every judge call, every attacker-LLM call — runs inside one
    recorded ``Run``. One task per evaluation, not per victim query: a task is the unit
    Kaggle records, and one per query would write a run file per query.

    The return value comes back through a closure rather than the task's own return
    slot, deliberately. Kaggle serializes what a task returns into the run file, and a
    ``ScenarioResults`` is not JSON — routing it through the task would either fail or
    silently flatten it.

    Args:
        fn:   The callable to run. Usually a bound ``AttackEvaluator.run``.
        name: Task name. Kaggle enforces a length limit and rejects an over-long one at
              decoration time, so this is truncated and suffixed with a counter.
        llm:  The model recorded as "under test" for this run. Defaults to ``kbench.llm``
              — the notebook's own model — which is *not* necessarily the victim; nothing
              in the eval reads it, it is bookkeeping for the leaderboard.
    """
    kbench = _kbench_module()
    box: dict = {}
    task_name = f"{name[:40]}-{next(KaggleLLM._chat_counter)}"

    def _body(llm):                      # first parameter must be the model under test
        box["value"] = fn(*args, **kwargs)

    _body.__name__ = task_name.replace("-", "_")
    # store_task=False keeps /kaggle/working free of a task file per attack; older
    # builds of kbench.task may not take it, hence the fallbacks.
    for decorator_kwargs in ({"name": task_name, "store_task": False},
                             {"name": task_name},
                             {}):
        try:
            task = kbench.task(**decorator_kwargs)(_body)
            break
        except TypeError:
            continue
    else:                                                # pragma: no cover - defensive
        raise RuntimeError("kbench.task rejected every decorator form tried")

    task.run(llm=kbench.llm if llm is None else llm)
    if "value" not in box:                               # pragma: no cover - defensive
        raise RuntimeError(
            f"kbench task {task_name!r} finished without running {fn!r} — check the Run "
            "output in the notebook for the assertion or error it recorded.")
    return box["value"]


# ---------------------------------------------------------------------------
# Factory — one model string, the right subclass
# ---------------------------------------------------------------------------

def make_llm(model, backend: str = "api", **kwargs) -> UnifiedLLM:
    """
    Turn a model string into the right ``UnifiedLLM``, or pass an instance through.

    The one place the ``kaggle/`` prefix is interpreted. Every seam that used to build an
    ``APILLM`` from a bare string — the attacker LLM, the ``*GetScore`` judge, TAP's
    on-topic model, Multilingual's translator, ``make_target`` — goes through here, so a
    Kaggle model is usable in each of those roles by prefix alone.

    Args:
        model:    ``"kaggle/<id>"`` → ``KaggleLLM``; anything else → ``APILLM`` (or
                  ``LocalLLM`` when ``backend="local"``). A non-``str`` is returned
                  unchanged, so a pre-built LLM (or a bare callable) survives the trip.
        backend:  ``"api"`` (default) · ``"local"`` · ``"kaggle"``. The prefix wins over
                  the default, so ``backend`` only has to be set for local models.
        **kwargs: Forwarded to the chosen constructor; keys the constructor does not take
                  (``api_key`` / ``metis_location`` for Kaggle and local models,
                  ``device_map`` for API ones) are dropped rather than raising.
    """
    if not isinstance(model, str):
        return model
    if backend == "kaggle" or model.startswith(KAGGLE_PREFIX):
        return KaggleLLM(model=model, **_only_kwargs_for(KaggleLLM, kwargs))
    if backend == "local":
        return LocalLLM(model=model, **kwargs)
    return APILLM(model=model, **_only_kwargs_for(APILLM, kwargs))


def _only_kwargs_for(cls, kwargs: dict) -> dict:
    """Drop kwargs ``cls.__init__`` does not declare (see ``make_llm``)."""
    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        log.debug("make_llm: %s ignores %s", cls.__name__, dropped)
    return {k: v for k, v in kwargs.items() if k in accepted}


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

#: Marker a white-box attack puts where its adversarial tokens will go, so the
#: victim's *whole* prompt can be rendered once and split around that span. Private
#: Use Area codepoints — no tokenizer template and no dataset text contains them.
ADV_SENTINEL = "\ue100ADVSPAN\ue100"


def render_messages(tokenizer, messages: list[dict],
                    add_generation_prompt: bool = True) -> tuple[str, bool]:
    """
    Render a messages list to the exact prompt text a local model is fed.

    The three branches mirror ``LocalLLM._build_local_prompt_ids`` — which now calls
    this — so a white-box attack that tokenizes the result is optimizing the same
    string the victim will actually be given. Keeping them in one place is the point:
    a second renderer that drifts is how an attack ends up optimizing a prompt shape
    the victim never sees.

    Returns:
        (text, add_special_tokens) — the flag to pass to ``tokenizer.encode``. It is
        ``False`` for the chat-template branch, whose output already carries BOS.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # A lone {"role": "raw"} turn carries a fully pre-rendered prompt (StruQ / SecAlign).
    if len(messages) == 1 and messages[0].get("role") == "raw":
        return messages[0].get("content", ""), True

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
        return text, False
    except ValueError:
        parts: list[str] = []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"USER: {content}")
            elif role == "assistant":
                parts.append(f"ASSISTANT: {content}")
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        log.debug("[render_messages] No chat template — using plain USER/ASSISTANT format")
        return "\n\n".join(parts), True


def split_prompt_around(tokenizer, messages: list[dict],
                        marker: str = ADV_SENTINEL) -> tuple[str, str, bool]:
    """
    Render ``messages`` and split the result on ``marker``.

    ``messages`` must contain the marker exactly once, in one content field. The
    result is the decomposition every token-level attack needs:

        full_ids = encode(head) + <adversarial token ids> + encode(tail) [+ target]

    ``tail`` is what BEAST calls ``end_inst_token`` — the close of the user turn plus
    the generation prompt. Rendering once and splitting is what keeps the adversarial
    span *inside* the user message: building the prompt with
    ``add_generation_prompt=True`` and appending tokens after it puts them in the
    assistant turn instead, which optimizes a continuation of the model's own reply.

    Returns:
        (head_text, tail_text, add_special_tokens)

    Raises:
        ValueError: if the marker is absent or repeated after rendering.
    """
    text, add_special = render_messages(tokenizer, messages)
    count = text.count(marker)
    if count != 1:
        raise ValueError(
            f"adversarial marker appears {count} times in the rendered prompt, expected 1"
        )
    head, _, tail = text.partition(marker)
    return head, tail, add_special


def bare_prompt_split(tokenizer, injection_prefix: str, injection_suffix: str = "",
                      system_prompt: str = "") -> tuple[str, str, bool]:
    """
    ``split_prompt_around`` for a plain ``[system][user]`` prompt, with no IPI carrier.

    What a white-box attack falls back to when it is called as a bare function rather
    than through ``run_scenario``. Against a real scenario use
    ``harness.split_optimization_prompt`` instead — this shape is not the one the
    victim is given.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {"role": "user", "content": f"{injection_prefix}{ADV_SENTINEL}{injection_suffix}"})
    return split_prompt_around(tokenizer, messages, ADV_SENTINEL)


def _extract_logprob(logprob_dict: dict[str, float], target_token: str) -> float:
    """
    Extract log-probability of ``target_token`` from a logprob dict.
    Checks both 'Token' and ' Token' forms. Returns -inf if not found.
    """
    logprobs = []
    if " " + target_token in logprob_dict:
        logprobs.append(logprob_dict[" " + target_token])
    if target_token in logprob_dict:
        logprobs.append(logprob_dict[target_token])
    return max(logprobs) if logprobs else float("-inf")


def parse_json_response(text: str, required_keys: list[str]) -> Optional[dict]:
    """
    Extract the first valid JSON object from text that contains all required keys.
    Tries direct parse first, then regex extraction of the first {...} block.
    Returns None if no valid object is found.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and all(k in obj for k in required_keys):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and all(k in obj for k in required_keys):
                return obj
        except json.JSONDecodeError:
            pass
    log.debug("parse_json_response: no valid JSON found in: %r", text[:200])
    return None
