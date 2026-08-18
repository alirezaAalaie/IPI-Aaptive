"""
The GCG token-gradient step — the one white-box mutation primitive.

Phase E's deferred item. The rest of ``ipi/mutation/`` rewrites *text*: an LLM
paraphrases a template, or a rule base64-encodes it. This family rewrites *tokens*,
guided by the gradient of the target's own loss, and so it is the only operator that
needs white-box access.

    build_input_ids            [system][user: prefix+suffix][assistant: target] + slices
    compute_loss_and_grads     forward + backward -> (loss, d loss / d one-hot suffix)
    get_embedding_matrix       the model's token embedding weights
    sample_candidates          gradient top-k -> B candidate suffixes, one swap each
    score_candidates           teacher-forced CE for every candidate, batched
    build_not_allowed_ids      the non-ASCII / special-token exclusion set

**Extracted from our own ``attacks/gcg.py``, not from upstream.** EasyJailbreak's
``MutationTokenGradient`` is written against their ``WhiteBoxModelBase`` — the layer our
``Victim`` contract replaces — so porting it would have meant porting their model stack.
These functions are byte-identical to the ones ``attacks/gcg.py`` has always used; that
file now imports them from here, so there is one implementation.

Why there is no ``MutationBase`` subclass here yet
--------------------------------------------------
Every other operator is 1-to-n over ``Instance`` *text*. This one is 1-to-B over token
**ids**, at ~512 candidates per step, scored in batches that share a KV prefix. Wrapping
each candidate in an ``Instance`` and selecting with ``ReferenceLossSelector`` — which
re-encodes text and runs one unbatched forward per instance — would be correct and
roughly two orders of magnitude slower. The wrapper lands with the GCG recipe migration,
once that trade-off has been decided and measured. See ``docs/refactor-handoff.md`` §6a.

``torch`` is imported at module load, so this module is torch-gated like the white-box
attacks: ``ipi/mutation/__init__.py`` must not import it eagerly.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "build_input_ids",
    "compute_loss_and_grads",
    "get_embedding_matrix",
    "score_candidates",
    "sample_candidates",
    "build_not_allowed_ids",
]


# ---------------------------------------------------------------------------
# Token-slice utilities (universal, no FastChat dependency)
# ---------------------------------------------------------------------------

def build_input_ids(
    tokenizer,
    system_prompt: str,
    adv_prefix: str,
    adv_suffix_ids: list[int],
    target_str: str,
    device,
) -> tuple[torch.Tensor, slice, slice]:
    """
    Build the full input token sequence:
      [system] [user: adv_prefix + adv_suffix] [assistant: target_str]

    Returns:
        input_ids: (1, L) long tensor on `device`.
        suffix_slice: slice into input_ids for the adv_suffix tokens.
        target_slice: slice into input_ids for the target_str tokens.
    """
    # Build messages list
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": adv_prefix})  # prefix only for now

    # Encode the prompt (without suffix and target)
    try:
        prompt_text: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt_text = (
            (f"[SYSTEM] {system_prompt}\n\n" if system_prompt else "")
            + f"[USER] {adv_prefix}\n[ASSISTANT] "
        )

    prompt_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Target tokens
    target_ids: list[int] = tokenizer.encode(target_str, add_special_tokens=False)

    # Full sequence: prompt + adv_suffix + target
    full_ids = prompt_ids + adv_suffix_ids + target_ids

    suffix_start = len(prompt_ids)
    suffix_end   = suffix_start + len(adv_suffix_ids)
    target_start = suffix_end
    target_end   = target_start + len(target_ids)

    suffix_slice = slice(suffix_start, suffix_end)
    target_slice = slice(target_start, target_end)

    ids_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    return ids_tensor, suffix_slice, target_slice


def compute_loss_and_grads(
    model,
    input_ids: torch.Tensor,
    suffix_slice: slice,
    target_slice: slice,
) -> tuple[float, torch.Tensor]:
    """
    Forward + backward pass.

    Returns:
        loss:  scalar float (cross-entropy loss over target tokens).
        grads: (suffix_len, vocab_size) gradient tensor w.r.t. one-hot
               embedding at each suffix position.
    """
    embed_weights = get_embedding_matrix(model)   # (vocab, d_model)
    vocab_size, d_model = embed_weights.shape

    # One-hot encode suffix tokens
    suffix_ids = input_ids[0, suffix_slice]                              # (S,)
    one_hot = torch.zeros(len(suffix_ids), vocab_size,
                          device=embed_weights.device, dtype=embed_weights.dtype)
    one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)

    # Embed the full input: prefix uses normal embed lookup, suffix uses one_hot @ W
    with torch.enable_grad():
        full_embed = embed_weights[input_ids[0]]   # (L, d_model)

        # Replace suffix embedding positions with differentiable one_hot @ W
        suffix_embed = one_hot @ embed_weights     # (S, d_model)
        full_embed = full_embed.clone()
        full_embed[suffix_slice] = suffix_embed

        # Forward
        output = model(inputs_embeds=full_embed.unsqueeze(0),
                       use_cache=False, return_dict=True)
        logits = output.logits[0]                  # (L, vocab)

        # Loss over target positions
        tgt_ids = input_ids[0, target_slice]       # (T,)
        shift   = target_slice.start - 1           # predict from position before target start
        log_probs = F.log_softmax(logits[shift: shift + len(tgt_ids)], dim=-1)
        loss = -log_probs[torch.arange(len(tgt_ids)), tgt_ids].mean()

        loss.backward()

    grads = one_hot.grad.detach().clone()          # (S, vocab)
    loss_val = loss.item()
    return loss_val, grads


def get_embedding_matrix(model) -> torch.Tensor:
    """Extract the token embedding weight matrix from a HF model."""
    for name in ("embed_tokens", "word_embeddings", "wte"):
        for mod in model.modules():
            if hasattr(mod, name):
                return getattr(mod, name).weight
    # Fallback: first embedding layer found
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            return mod.weight
    raise RuntimeError("Could not locate embedding matrix in model.")


@torch.no_grad()
def score_candidates(
    model,
    input_ids: torch.Tensor,   # (1, L) current ids
    suffix_slice: slice,
    target_slice: slice,
    candidate_ids: torch.Tensor,  # (B, S) candidate suffix ids
    batch_size_eval: int = 128,
) -> torch.Tensor:
    """
    Evaluate cross-entropy loss for each candidate suffix.
    Returns (B,) float tensor of losses (lower = better).
    """
    losses: list[float] = []
    L = input_ids.shape[1]
    S = suffix_slice.stop - suffix_slice.start

    for b_start in range(0, candidate_ids.shape[0], batch_size_eval):
        batch_cands = candidate_ids[b_start: b_start + batch_size_eval]  # (b, S)
        b = batch_cands.shape[0]

        # Expand input_ids to batch and replace suffix
        batch_ids = input_ids.expand(b, -1).clone()               # (b, L)
        batch_ids[:, suffix_slice] = batch_cands                   # (b, L)

        out = model(input_ids=batch_ids, use_cache=False, return_dict=True)
        logits = out.logits   # (b, L, V)

        tgt_ids = input_ids[0, target_slice]                       # (T,)
        T = len(tgt_ids)
        shift = target_slice.start - 1
        # logits for predicting target tokens
        tgt_logits = logits[:, shift: shift + T, :]                # (b, T, V)
        tgt_ids_exp = tgt_ids.unsqueeze(0).expand(b, -1)           # (b, T)

        loss = F.cross_entropy(
            tgt_logits.reshape(b * T, -1),
            tgt_ids_exp.reshape(b * T),
            reduction="none",
        ).reshape(b, T).mean(dim=1)                                # (b,)

        losses.extend(loss.tolist())

        del batch_ids, out, logits
        gc.collect()

    return torch.tensor(losses)


def sample_candidates(
    grads: torch.Tensor,           # (S, vocab)
    suffix_ids: torch.Tensor,      # (S,) current suffix token ids
    top_k: int,
    batch_size: int,
    not_allowed_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample `batch_size` candidate suffix substitutions using gradient-guided top-k.

    For each candidate:
      1. Randomly pick a suffix position p.
      2. From the top-k tokens at position p (by gradient), randomly pick one.
      3. Build a new suffix with that single substitution.

    Returns: (batch_size, S) long tensor of candidate suffix ids.
    """
    S = suffix_ids.shape[0]
    vocab = grads.shape[1]

    # Optionally zero out not-allowed token ids
    if not_allowed_ids is not None:
        grads[:, not_allowed_ids] = float("inf")

    # Top-k candidates per position: negative gradient (lower gradient = better next token)
    top_k_ids = torch.topk(-grads, k=min(top_k, vocab), dim=1).indices   # (S, top_k)

    # Sample candidates
    cand_positions = torch.randint(0, S, (batch_size,))                    # (B,)
    cand_token_idx = torch.randint(0, min(top_k, vocab), (batch_size,))   # (B,)
    new_token_ids  = top_k_ids[cand_positions, cand_token_idx]             # (B,)

    # Build candidate suffixes: copy current, replace one position
    candidates = suffix_ids.unsqueeze(0).expand(batch_size, -1).clone()   # (B, S)
    candidates[torch.arange(batch_size), cand_positions] = new_token_ids

    return candidates


# ---------------------------------------------------------------------------
# Non-ASCII / special token filter (optional, mirrors original allow_non_ascii=False)
# ---------------------------------------------------------------------------

def build_not_allowed_ids(tokenizer, allow_non_ascii: bool = False) -> Optional[torch.Tensor]:
    """
    Build a list of token ids that should not appear in adversarial suffixes.
    If allow_non_ascii=False (default), excludes non-ASCII and special tokens.
    """
    if allow_non_ascii:
        return None
    disallowed: list[int] = []
    for tok_id in range(tokenizer.vocab_size):
        try:
            tok_str = tokenizer.decode([tok_id])
        except Exception:
            disallowed.append(tok_id)
            continue
        if not tok_str.isascii():
            disallowed.append(tok_id)
    if disallowed:
        return torch.tensor(disallowed, dtype=torch.long)
    return None
