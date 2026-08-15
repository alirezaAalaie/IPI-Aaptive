"""
DefensiveToken configuration — token names, chat templates, weight resolution.

Direct port of ``code/defense/DefensiveToken-main/setup.py``.

The defense is five *new* special tokens whose input embeddings were optimised
for prompt-injection robustness (Chen et al., AISec 2025). Everything else about
the model is untouched, which is the paper's selling point: prepend the tokens
when you want security, skip them when you want stock utility.

Load-bearing details
--------------------
* **The chat templates are not the models' stock templates.** They are the
  Meta-SecAlign-style formats the DefensiveTokens were optimised against — e.g.
  Llama-3 renders ``<|start_header_id|>role<|end_header_id|>\\ncontent\\n\\n<|eot_id|>``,
  where stock renders ``<|start_header_id|>role<|end_header_id|>\\n\\ncontent<|eot_id|>``.
  Substituting the stock template silently moves the optimised embeddings out of
  the context they were trained in. They are therefore built here by the same
  concatenation for every model rather than spelled out four times, so a stray
  character cannot drift into one of them (same reasoning as
  ``ipi/defenses/struq/config.py``). ``scripts/check_defensivetoken_fidelity.py``
  asserts byte-equality against the vendored upstream file.
* **The five tokens are emitted *before* ``bos_token``**, not after. That is
  upstream's ordering and the released embeddings assume it.
* **Role convention is inverted relative to Meta-SecAlign**: ``system`` carries
  the *trusted instruction*, ``user`` carries the *untrusted data*. Upstream's
  README calls this out explicitly as the change to make when reproducing with
  the Meta_SecAlign harness.

IPI adaptations vs original
---------------------------
* Upstream ships one script that materialises all four checkpoints to disk.
  Here the same patch is also available in-place against an already-loaded model
  (``apply_defensive_tokens``), because a Kaggle run cannot afford a second
  16 GB copy of the weights just to add five embedding rows.
* Upstream indexes the new embedding rows positionally (``weight[-5 + i]``).
  This resolves them through ``convert_tokens_to_ids`` instead — identical
  whenever ``resize_token_embeddings`` does not pad, and correct when it does.
* ``defensivetokens.json`` is resolved rather than assumed to be in the CWD: an
  explicit path, then ``$IPI_DEFENSIVE_TOKENS_JSON``, then the vendored copy
  under ``code/defense/``, then a cached download from upstream. Kaggle sees
  neither the repo's ``code/`` tree nor the CWD upstream assumes.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Models for which upstream released optimised DefensiveTokens.
SUPPORTED_MODELS: Tuple[str, ...] = (
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

#: Every released checkpoint uses five tokens ("a few DefensiveTokens").
NUM_DEFENSIVE_TOKENS = 5

DEFENSIVE_TOKEN_NAMES: Tuple[str, ...] = tuple(
    f"[DefensiveToken{i}]" for i in range(NUM_DEFENSIVE_TOKENS)
)

#: Directory suffix upstream's setup.py gives the patched checkpoint.
OUTPUT_DIR_SUFFIX = f"-{NUM_DEFENSIVE_TOKENS}DefensiveTokens"

DEFENSIVE_TOKENS_URL = (
    "https://raw.githubusercontent.com/Sizhe-Chen/DefensiveToken/main/defensivetokens.json"
)

DEFENSIVE_TOKENS_FILENAME = "defensivetokens.json"

#: Environment override for the weights file.
DEFENSIVE_TOKENS_ENV = "IPI_DEFENSIVE_TOKENS_JSON"


# ---------------------------------------------------------------------------
# Chat templates (setup.py:chat_templates)
# ---------------------------------------------------------------------------

_TOKEN_LITERAL = "".join(DEFENSIVE_TOKEN_NAMES)

# The odd interior whitespace below is upstream's — its templates are
# triple-quoted literals indented inside a dict, and the surviving newline +
# four spaces are part of the string the released checkpoints carry. `{%-`
# strips it back off at render time, so it is inert; it is reproduced anyway so
# the saved tokenizer_config is byte-identical to upstream's.
_INDENT = "\n\n    "
_BOS_CHUNK = "{{- bos_token }}" + _INDENT


def _build_chat_template(turn: str, generation: str, *, bos: bool) -> str:
    """Assemble one model's DefensiveToken chat template from its two variable
    parts: how a single turn renders, and how the generation prompt renders."""
    return (
        "{%- if add_defensive_tokens %}\n"
        "{{- '" + _TOKEN_LITERAL + "' }}\n"
        "{%- endif %}" + _INDENT
        + (_BOS_CHUNK if bos else "")
        + "{%- for message in messages %}" + "\n\n        "
        + "{{- " + turn + " }}" + _INDENT
        + "{%- endfor %}" + _INDENT
        + "{%- if add_generation_prompt %}\n"
        "{{- " + generation + " }}\n"
        "{%- endif %}\n"
    )


_LLAMA3_TEMPLATE = _build_chat_template(
    turn=(
        "'<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'"
        "+ message['content'] | trim + '\\n\\n' + '<|eot_id|>'"
    ),
    generation="'<|start_header_id|>assistant<|end_header_id|>\\n'",
    bos=True,
)

CHAT_TEMPLATES: Dict[str, str] = {
    "meta-llama/Meta-Llama-3-8B-Instruct": _LLAMA3_TEMPLATE,
    "meta-llama/Llama-3.1-8B-Instruct": _LLAMA3_TEMPLATE,
    "tiiuae/Falcon3-7B-Instruct": _build_chat_template(
        turn="'<|' + message['role'] + '|>\\n' + message['content'] | trim + '\\n\\n'",
        generation="'<|assistant|>\\n'",
        bos=False,
    ),
    "Qwen/Qwen2.5-7B-Instruct": _build_chat_template(
        turn=(
            "'<|im_start|>' + message['role'] + '\\n' "
            "+ message['content'] | trim + '\\n\\n<|im_end|>\\n'"
        ),
        generation="'<|im_start|>assistant\\n'",
        bos=False,
    ),
}


# ---------------------------------------------------------------------------
# Model-key resolution
# ---------------------------------------------------------------------------

def resolve_model_key(model_name: str) -> str:
    """
    Map a model identifier onto one of :data:`SUPPORTED_MODELS`.

    Accepts the exact HF id, a patched-checkpoint directory
    (``...-5DefensiveTokens``), or a local path whose basename matches a
    supported model. Raises rather than guessing — picking the wrong key would
    write another model's embeddings into the vocabulary, which nothing
    downstream would flag.
    """
    if model_name in CHAT_TEMPLATES:
        return model_name

    name = str(model_name).rstrip("/")
    if name.endswith(OUTPUT_DIR_SUFFIX):
        name = name[: -len(OUTPUT_DIR_SUFFIX)]
    if name in CHAT_TEMPLATES:
        return name

    base = Path(name).name.lower()
    matches = [k for k in CHAT_TEMPLATES if Path(k).name.lower() == base]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"No released DefensiveTokens for {model_name!r}. Upstream optimised them "
        f"per-model; they do not transfer. Supported: {list(SUPPORTED_MODELS)}."
    )


# ---------------------------------------------------------------------------
# Weight-file resolution
# ---------------------------------------------------------------------------

def _candidate_paths() -> List[Path]:
    """Local locations to check before hitting the network, in priority order."""
    candidates: List[Path] = []

    env = os.environ.get(DEFENSIVE_TOKENS_ENV)
    if env:
        candidates.append(Path(env).expanduser())

    # Vendored copy: code/defense/DefensiveToken-main/defensivetokens.json,
    # found by walking up from this module (repo checkout, not pip install).
    for parent in Path(__file__).resolve().parents:
        vendored = (
            parent / "code" / "defense" / "DefensiveToken-main" / DEFENSIVE_TOKENS_FILENAME
        )
        if vendored.exists():
            candidates.append(vendored)
            break

    candidates.append(Path.cwd() / DEFENSIVE_TOKENS_FILENAME)
    candidates.append(cache_path())
    return candidates


def cache_path() -> Path:
    """Where a downloaded copy of the weights is kept."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "ipi" / DEFENSIVE_TOKENS_FILENAME


def resolve_defensive_tokens_path(
    path: Optional[str] = None,
    allow_download: bool = True,
) -> Path:
    """
    Locate ``defensivetokens.json``, downloading it to the cache if needed.

    Args:
        path:           Explicit path; checked first and required to exist.
        allow_download: Fall back to fetching from upstream GitHub. Turn this
                        off in offline environments to get a clear failure
                        instead of a network timeout.
    """
    if path is not None:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"defensivetokens.json not found at {p}")
        return p

    for candidate in _candidate_paths():
        if candidate.exists():
            log.info("[DefensiveToken] Using weights at %s", candidate)
            return candidate

    if not allow_download:
        raise FileNotFoundError(
            "defensivetokens.json not found locally and allow_download=False. "
            f"Set ${DEFENSIVE_TOKENS_ENV} or pass tokens_path=..."
        )

    dest = cache_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("[DefensiveToken] Downloading weights from %s", DEFENSIVE_TOKENS_URL)
    import urllib.request

    tmp = dest.with_suffix(".json.part")
    urllib.request.urlretrieve(DEFENSIVE_TOKENS_URL, tmp)
    tmp.replace(dest)
    return dest


def load_defensive_tokens(
    model_name: str,
    tokens_path: Optional[str] = None,
    allow_download: bool = True,
) -> List[Sequence[float]]:
    """
    Return the released DefensiveToken embedding vectors for ``model_name``.

    Shape is ``(NUM_DEFENSIVE_TOKENS, hidden_size)``; the caller is responsible
    for checking ``hidden_size`` against the model it is patching.
    """
    key = resolve_model_key(model_name)
    path = resolve_defensive_tokens_path(tokens_path, allow_download=allow_download)

    with open(path, "r") as f:
        all_tokens = json.load(f)

    if key not in all_tokens:
        raise KeyError(
            f"{path} has no entry for {key!r} (has: {sorted(all_tokens)}). "
            "The file may be stale — delete the cache and retry."
        )

    vectors = all_tokens[key]
    if len(vectors) != NUM_DEFENSIVE_TOKENS:
        raise ValueError(
            f"Expected {NUM_DEFENSIVE_TOKENS} DefensiveTokens for {key}, "
            f"found {len(vectors)} in {path}."
        )
    return vectors
