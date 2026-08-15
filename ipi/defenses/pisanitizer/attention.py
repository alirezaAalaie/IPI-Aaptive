"""
Attention reconstruction for PISanitizer.

Port of ``code/defense/PISanitizer-main/PISanitizer/attention_utils.py``, which
itself adapts the AT2 codebase.

Why reconstruct instead of ``output_attentions=True``: the SDPA and
flash-attention kernels every recent model defaults to never materialise the
attention matrix, so ``output_attentions=True`` either silently falls back to
the slow eager path or returns nothing. This recomputes attention for *one
query position* from the cached hidden states — the projections are re-run, but
only one row of the (seq_len x seq_len) matrix is ever formed, which is what
makes the method viable on long contexts.

IPI adaptations vs original
---------------------------
* Upstream hard-codes ``.to("cuda:0")`` before applying RoPE, which crashes on
  CPU and silently moves tensors off-device under a sharded ``device_map``.
  Everything here follows the layer's own device.
* Upstream allocates a zero tensor of the full attention shape at the top of
  ``get_attention_weights_one_layer`` and immediately overwrites it with the
  return value of the real computation. Dropped — it was dead, and on a long
  context it is a large allocation repeated once per layer.
* ``position_ids`` and the causal mask do not depend on the layer, so they are
  computed once by the caller instead of once per layer.
* Model-family detection accepts an explicit ``model_type`` and raises with the
  supported list rather than a bare KeyError when the internals have moved.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Substring of the model id -> the ``transformers.models.<name>`` module whose
#: ``apply_rotary_pos_emb`` / ``repeat_kv`` implement that family's attention.
MODEL_TYPE_KEYWORDS = {
    "llama": "llama",
    "gpt": "gpt_oss",
    "glm": "glm",
    "phi": "phi3",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "gemma": "gemma3",
}

#: Families whose q/k live in separate projections (vs. a fused qkv_proj).
_SPLIT_QKV = ("llama", "qwen2", "qwen1.5", "qwen3", "gemma3", "glm")
_FUSED_QKV = ("phi3",)

#: Families that apply a norm to q/k before RoPE.
_QK_NORM = ("gemma3", "qwen3")


def infer_model_type(model: Any) -> str:
    """Guess the ``transformers.models`` submodule from the model id."""
    name = str(getattr(model, "name_or_path", "")).lower()
    for keyword, model_type in MODEL_TYPE_KEYWORDS.items():
        if keyword in name:
            return model_type
    raise ValueError(
        f"Cannot infer a model family from {name!r}. Pass model_type= "
        f"explicitly; supported: {sorted(set(MODEL_TYPE_KEYWORDS.values()))}."
    )


def get_helpers(model_type: str):
    """Fetch ``(apply_rotary_pos_emb, repeat_kv)`` for a model family."""
    import transformers.models

    if not hasattr(transformers.models, model_type):
        raise ValueError(
            f"transformers has no models.{model_type} — either the family is "
            "unsupported or this transformers version renamed it."
        )
    model_module = getattr(transformers.models, model_type)
    try:
        modeling_module = getattr(model_module, f"modeling_{model_type}")
        return modeling_module.apply_rotary_pos_emb, modeling_module.repeat_kv
    except AttributeError as exc:
        raise RuntimeError(
            f"transformers.models.{model_type}.modeling_{model_type} does not "
            f"expose apply_rotary_pos_emb/repeat_kv ({exc}). PISanitizer "
            "reconstructs attention from model internals and is therefore "
            "coupled to the transformers version; upstream pins 4.56.2."
        ) from exc


def _language_model(model: Any, model_type: str):
    """The decoder stack. Multimodal wrappers nest it one level deeper."""
    inner = getattr(model, "model", model)
    if model_type == "gemma3" or not hasattr(inner, "layers"):
        inner = getattr(inner, "language_model", inner)
    if not hasattr(inner, "layers"):
        raise RuntimeError(
            f"Could not locate the decoder layers on {type(model).__name__}. "
            "PISanitizer needs direct access to the attention projections."
        )
    return inner


def num_layers(model: Any, model_type: Optional[str] = None) -> int:
    return len(_language_model(model, model_type or infer_model_type(model)).layers)


def build_position_ids_and_mask(model: Any, seq_len: int):
    """
    Causal additive mask and position ids for a full sequence.

    Shape is ``(1, 1, seq_len, seq_len + 1)``; the extra column is upstream's
    and is trimmed against the actual key length before use.
    """
    import torch

    device = model.device
    dtype = model.dtype
    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
    mask = torch.ones(seq_len, seq_len + 1, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    mask *= torch.finfo(dtype).min
    return position_ids, mask[None, None]


def layer_attention_weights(
    model: Any,
    hidden_states: Sequence[Any],
    layer_index: int,
    position_ids: Any,
    attention_mask: Any,
    attribution_start: int,
    attribution_end: int,
    model_type: Optional[str] = None,
):
    """
    Recompute layer ``layer_index``'s attention for query positions
    ``[attribution_start - 1, attribution_end - 1)``.

    Args:
        hidden_states: The tuple from ``output_hidden_states=True``. Entry
                       ``layer_index`` is the *input* to that layer, which is
                       what its ``input_layernorm`` and q/k projections consume.

    Returns:
        Tensor ``(1, num_heads, n_query_positions, seq_len)`` of softmaxed
        attention weights.
    """
    import torch

    model_type = model_type or infer_model_type(model)
    language_model = _language_model(model, model_type)
    n_layers = len(language_model.layers)
    if not 0 <= layer_index < n_layers:
        raise IndexError(f"layer_index {layer_index} outside 0..{n_layers - 1}")

    layer = language_model.layers[layer_index]
    self_attn = layer.self_attn

    hs = layer.input_layernorm(hidden_states[layer_index])
    bsz, q_len, _ = hs.size()

    cfg = language_model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = self_attn.head_dim

    if model_type in _SPLIT_QKV:
        query_states = self_attn.q_proj(hs)
        key_states = self_attn.k_proj(hs)
    elif model_type in _FUSED_QKV:
        qkv = self_attn.qkv_proj(hs)
        query_pos = n_heads * head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos: query_pos + n_kv_heads * head_dim]
    else:
        raise ValueError(
            f"PISanitizer does not know how to read q/k for model family "
            f"{model_type!r}."
        )

    query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    if model_type in _QK_NORM:
        query_states = self_attn.q_norm(query_states)
        key_states = self_attn.k_norm(key_states)

    if model_type == "gemma3" and getattr(self_attn, "is_sliding", False):
        cos, sin = language_model.rotary_emb_local(hs, position_ids)
    else:
        cos, sin = language_model.rotary_emb(hs, position_ids)

    apply_rotary_pos_emb, repeat_kv = get_helpers(model_type)

    # Upstream pins these to cuda:0; follow the layer's own device instead so
    # this works on CPU and under a sharded device_map.
    device = query_states.device
    cos, sin = cos.to(device), sin.to(device)
    key_states = key_states.to(device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

    lo, hi = attribution_start - 1, attribution_end - 1
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]][:, :, lo:hi].to(device)
    query_states = query_states[:, :, lo:hi]

    attn = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn = attn + causal_mask
    return torch.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)


def token_attention_signal(
    model: Any,
    hidden_states: Sequence[Any],
    attribution_start: int,
    attribution_end: int,
    n_context_tokens: int,
    mode: str,
    model_type: Optional[str] = None,
):
    """
    Reduce every layer's attention into one score per input token.

    ``mode`` is ``"<across-layers>-<across-heads>"``: heads are reduced first
    (``max`` / ``avg`` / ``top5``), then layers. Upstream's default ``max-avg``
    means "average over heads, then take the strongest layer".

    Returns a 1-D float tensor of length ``n_context_tokens``.
    """
    import torch

    model_type = model_type or infer_model_type(model)
    seq_len = hidden_states[0].shape[1]
    position_ids, mask = build_position_ids_and_mask(model, seq_len)

    per_layer_max, per_layer_avg = [], []
    with torch.no_grad():
        for layer_index in range(num_layers(model, model_type)):
            attn = layer_attention_weights(
                model, hidden_states, layer_index,
                position_ids, mask,
                attribution_start, attribution_end,
                model_type=model_type,
            )
            # (1, heads, n_query, seq_len) -> (heads, n_context_tokens);
            # the query axis is a single position, and the final column (the
            # newly generated token attending to itself) is dropped.
            per_token = attn[0, :, :, :n_context_tokens].mean(dim=1).float().cpu()
            per_layer_max.append(per_token.max(dim=0).values)
            per_layer_avg.append(per_token.mean(dim=0))

    across_layers, across_heads = mode.split("-")
    stacked = torch.stack(per_layer_max if across_heads == "max" else per_layer_avg)

    if across_layers == "max":
        return stacked.max(dim=0).values
    if across_layers == "avg":
        return stacked.mean(dim=0)
    if across_layers == "top5":
        k = min(5, stacked.shape[0])
        return stacked.topk(k, dim=0).values.mean(dim=0)
    raise ValueError(f"Unknown layer reduction {across_layers!r} in mode {mode!r}")


def reduce_signal_modes(per_layer_max, per_layer_avg, mode: str):
    """Standalone reducer, exposed for analysis over cached per-layer scores."""
    import torch

    across_layers, across_heads = mode.split("-")
    stacked = torch.stack(
        [torch.as_tensor(x) for x in (per_layer_max if across_heads == "max" else per_layer_avg)]
    )
    if across_layers == "max":
        return stacked.max(dim=0).values
    if across_layers == "avg":
        return stacked.mean(dim=0)
    if across_layers == "top5":
        k = min(5, stacked.shape[0])
        return stacked.topk(k, dim=0).values.mean(dim=0)
    raise ValueError(f"Unknown layer reduction {across_layers!r}")


def hidden_states_for(model: Any, input_ids: Any) -> Tuple[Any, ...]:
    """One forward pass returning every layer's input hidden states."""
    import torch

    with torch.no_grad():
        return model(input_ids, output_hidden_states=True).hidden_states
