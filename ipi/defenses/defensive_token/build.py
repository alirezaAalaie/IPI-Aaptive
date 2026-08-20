"""
Adding the released DefensiveTokens to a public model.

Port of ``code/defense/DefensiveToken-main/setup.py``. Two entry points over one
shared core:

  * :func:`apply_defensive_tokens` — patch an already-loaded model in place.
    This is the Kaggle path: no second copy of the weights on disk.
  * :func:`build_defensive_token_model` — upstream's behaviour, materialising a
    ``<model>-5DefensiveTokens`` directory. Use this when a downstream harness
    (e.g. Meta_SecAlign) needs a real checkpoint path.

Both add the five special tokens, resize the embedding matrix, overwrite the new
rows with the optimised vectors, and install the DefensiveToken chat template.

IPI adaptations vs original
---------------------------
* Upstream writes ``weight.data[-5 + i]``, assuming the five new rows land at
  the tail. That holds only when ``resize_token_embeddings`` does not pad, and
  it is silently wrong when it does; rows are resolved via
  ``convert_tokens_to_ids`` here instead.
* The write is read back and compared against the source vector. Writing into a
  ``meta`` / offloaded / quantised embedding table is a no-op that leaves the
  tokens at their mean-initialised warm start — a defense that looks installed
  and does nothing. That failure is now loud.
* ``torch_dtype`` is exposed and defaults to the model's existing dtype rather
  than upstream's implicit float32, so patching a bf16 model in place does not
  silently upcast one matrix.
* **The resize is conditional.** Upstream calls ``resize_token_embeddings``
  unconditionally, which on a *padded* vocabulary shrinks the matrix rather than
  growing it — Qwen2.5 has 152064 embedding rows for 151665 tokens, so the five
  new ids (151665-151669) already have rows and upstream's call would drop the
  matrix to 151670 and reallocate the untied lm_head alongside it. The rows the
  tokens land in are identical either way (they are resolved by id, not by
  position), so the only difference is ~2 GiB of transient allocation and 394
  unused output logits that stock Qwen carries anyway. On a 15 GB card holding a
  7B model in bf16 that allocation is the difference between running and OOM.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from .config import (
    CHAT_TEMPLATES,
    DEFENSIVE_TOKEN_NAMES,
    NUM_DEFENSIVE_TOKENS,
    OUTPUT_DIR_SUFFIX,
    load_defensive_tokens,
    resolve_model_key,
)

log = logging.getLogger(__name__)


def _embedding_matrix(model: Any):
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise TypeError(
            f"{type(model).__name__} exposes no input-embedding weight to patch."
        )
    return emb


def apply_defensive_tokens(
    model: Any,
    tokenizer: Any,
    model_name: Optional[str] = None,
    tokens_path: Optional[str] = None,
    allow_download: bool = True,
    set_chat_template: bool = True,
) -> List[int]:
    """
    Add the released DefensiveTokens to ``model``/``tokenizer`` in place.

    Args:
        model:             A causal-LM whose input embeddings are writable
                           (not on ``meta``, not offloaded, not quantised).
        tokenizer:         Its tokenizer; gains five special tokens and, unless
                           disabled, the DefensiveToken chat template.
        model_name:        Which released token set to use. Defaults to the
                           model's own ``name_or_path``.
        tokens_path:       Explicit ``defensivetokens.json``.
        allow_download:    Permit fetching the weights from upstream GitHub.
        set_chat_template: Install the template that renders the tokens. Leave
                           this on — without it ``add_defensive_tokens=True``
                           is an unused Jinja variable and the defense is inert.

    Returns:
        The five new token ids, in order.

    Idempotent: re-running on an already-patched model rewrites the same rows.
    """
    import torch

    key = resolve_model_key(model_name or getattr(model, "name_or_path", "") or
                            getattr(getattr(model, "config", None), "_name_or_path", ""))
    vectors = load_defensive_tokens(key, tokens_path, allow_download=allow_download)

    emb = _embedding_matrix(model)
    hidden = emb.weight.shape[1]
    if len(vectors[0]) != hidden:
        raise ValueError(
            f"DefensiveTokens for {key} are {len(vectors[0])}-dimensional but the "
            f"model's embeddings are {hidden}-dimensional — wrong checkpoint."
        )

    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(DEFENSIVE_TOKEN_NAMES)}
    )
    rows_before = emb.weight.shape[0]
    if len(tokenizer) > rows_before:
        model.resize_token_embeddings(len(tokenizer))
        emb = _embedding_matrix(model)
        log.info(
            "[DefensiveToken] %s: added %d tokens, embedding rows %d -> %d",
            key, n_added, rows_before, emb.weight.shape[0],
        )
    else:
        # Padded vocabulary — the five new ids already have rows. Resizing here
        # would only *shrink* the matrix to len(tokenizer), which reallocates the
        # embedding table and the lm_head for no benefit; see the module docstring.
        log.info(
            "[DefensiveToken] %s: added %d tokens; embedding matrix already holds "
            "%d rows for %d tokens, no resize needed",
            key, n_added, rows_before, len(tokenizer),
        )

    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in DEFENSIVE_TOKEN_NAMES]
    unknown = tokenizer.convert_tokens_to_ids(tokenizer.unk_token) if tokenizer.unk_token else None
    for name, tid in zip(DEFENSIVE_TOKEN_NAMES, token_ids):
        if tid is None or tid < 0 or tid == unknown or tid >= emb.weight.shape[0]:
            raise RuntimeError(
                f"{name} did not get a usable id after resizing (got {tid}, "
                f"{emb.weight.shape[0]} embedding rows)."
            )

    if emb.weight.device.type == "meta":
        raise RuntimeError(
            "Input embeddings are on the 'meta' device — the model was loaded "
            "with weights offloaded or not materialised, so the DefensiveToken "
            "rows cannot be written. Reload with device_map=None or "
            "device_map={'': 0}."
        )

    with torch.no_grad():
        for i, tid in enumerate(token_ids):
            vec = torch.tensor(
                vectors[i], dtype=emb.weight.dtype, device=emb.weight.device
            )
            emb.weight.data[tid] = vec

            written = emb.weight.data[tid].to(torch.float32).cpu()
            expected = vec.to(torch.float32).cpu()
            if not torch.allclose(written, expected, atol=1e-3, rtol=0):
                raise RuntimeError(
                    f"Writing {DEFENSIVE_TOKEN_NAMES[i]} into row {tid} did not "
                    "take (max |diff| = "
                    f"{(written - expected).abs().max().item():.4g}). The "
                    "embedding table is quantised or offloaded; the defense "
                    "would have been silently inert."
                )

    if set_chat_template:
        tokenizer.chat_template = CHAT_TEMPLATES[key]

    return token_ids


def build_defensive_token_model(
    model_name: str,
    output_dir: Optional[str] = None,
    tokens_path: Optional[str] = None,
    allow_download: bool = True,
    torch_dtype: Any = None,
    device_map: Optional[str] = None,
) -> str:
    """
    Materialise ``<model_name>-5DefensiveTokens`` on disk (upstream setup.py).

    Prefer :func:`apply_defensive_tokens` for in-process evaluation — this
    writes a full second copy of the weights.

    Args:
        model_name:  One of the four models upstream released tokens for.
        output_dir:  Destination. Defaults to upstream's
                     ``<model_name>-5DefensiveTokens``.
        torch_dtype: Load dtype. Defaults to the checkpoint's own dtype.
        device_map:  Left as ``None`` (single-device load) on purpose —
                     ``"auto"`` can place the embedding table on ``meta``,
                     which makes the row writes unperformable.

    Returns:
        The output directory path.
    """
    import transformers

    key = resolve_model_key(model_name)
    out = Path(output_dir) if output_dir else Path(key + OUTPUT_DIR_SUFFIX)

    log.info("[DefensiveToken] Building %s -> %s", key, out)
    kwargs: dict = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = transformers.AutoModelForCausalLM.from_pretrained(key, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(key)

    apply_defensive_tokens(
        model, tokenizer,
        model_name=key,
        tokens_path=tokens_path,
        allow_download=allow_download,
    )

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    log.info(
        "[DefensiveToken] Saved %d-token checkpoint to %s",
        NUM_DEFENSIVE_TOKENS, out,
    )
    return str(out)
