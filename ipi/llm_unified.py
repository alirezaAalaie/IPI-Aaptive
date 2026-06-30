"""
Unified LLM interface for IPI attack research.

UnifiedLLM — abstract base class. Holds shared config and concrete helpers
             (__call__, get_logprob, get_response, chat, __repr__).
             Subclass with APILLM or LocalLLM.

APILLM     — API-backed model (litellm / OpenAI-compat / Metis proxy).
             Supports: openai, anthropic, google/gemini, deepseek,
                       metis_openai, metis_deepseek, metis_gemini,
                       and any raw litellm model string.
             Logprob support: openai, metis_openai, deepseek, metis_deepseek.
             Use: APILLM("gpt-4o-mini", system_prompt="...")

LocalLLM   — Local HuggingFace model. Full logits/logprob access.
             Required for: BEAST, logprob-based RS on any model.
             Use: LocalLLM("lmsys/vicuna-7b-v1.5")

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

Model registry
--------------
    APILLM.supported_models()  → dict[str, ModelSpec] of all known API model IDs
    LocalLLM.supported_models() → str describing accepted format
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
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
    deepseek | metis_openai | metis_deepseek | metis_gemini | litellm | local"""
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
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LogprobNotSupportedError(RuntimeError):
    """Raised when logprob retrieval is attempted against a provider without logprob API."""


class LocalOnlyError(RuntimeError):
    """Raised when a local-only method is called on APILLM."""


_LOGPROB_API_PROVIDERS = {"openai", "metis_openai", "deepseek", "metis_deepseek"}


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
    ):
        self.model_name     = model
        self.system_prompt  = system_prompt
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.top_p          = top_p
        self.top_k          = top_k
        self.max_bs         = max_bs
        self.extra_messages = extra_messages or []

        # Resolve spec (uses self.backend class variable)
        self._spec = KNOWN_MODELS.get(model) or ModelSpec(
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
        if provider in ("metis_openai", "metis_deepseek"):
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
        path = _METIS_OPENAI_PATH if provider == "metis_openai" else _METIS_DEEPSEEK_PATH
        client = OpenAI(api_key=self._api_key or "dummy", base_url=self._metis_base + path)
        resp = client.chat.completions.create(
            model=self._spec.model_id,
            messages=messages,   # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
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
    ):
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

        log.info("[LocalLLM] Loading: %s", self.model_name)
        self._tokenizer_obj = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, token=os.getenv("HF_TOKEN"),
        )
        if self._tokenizer_obj.pad_token is None:
            self._tokenizer_obj.pad_token = (
                self._tokenizer_obj.unk_token or self._tokenizer_obj.eos_token or "[PAD]"
            )
        self._tokenizer_obj.padding_side = "left"

        self._hf_model_obj = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            use_cache=False,
            low_cpu_mem_usage=True,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        ).eval()
        self._device = next(self._hf_model_obj.parameters()).device
        log.info("[LocalLLM] Loaded on device: %s", self._device)

    # --- Private: local generation ---

    def _build_local_prompt_ids(self, messages: list[dict]) -> list[int]:
        try:
            return self._tokenizer_obj.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        except ValueError:
            # Tokenizer has no chat_template (older models like Vicuna v1.3,
            # base LLaMA, etc.) — fall back to a simple human/assistant format.
            parts: list[str] = []
            for m in messages:
                role    = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append(content)
                elif role == "user":
                    parts.append(f"USER: {content}")
                elif role == "assistant":
                    parts.append(f"ASSISTANT: {content}")
            parts.append("ASSISTANT:")
            text = "\n\n".join(parts)
            log.debug("[LocalLLM] No chat template — using plain USER/ASSISTANT format")
            return self._tokenizer_obj.encode(text, add_special_tokens=True)

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
# Standalone helpers
# ---------------------------------------------------------------------------

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
