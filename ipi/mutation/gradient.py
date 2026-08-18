"""
The white-box token operators — the mutations that rewrite *tokens*, not text.

Phase E's deferred item. The rest of ``ipi/mutation/`` rewrites *text*: an LLM
paraphrases a template, or a rule base64-encodes it. This family rewrites *tokens*,
guided by the gradient of the target's own loss, and so it is the only operator that
needs white-box access.

    TokenGradientMutation      GCG's step as a MutationBase: 1 parent -> B children
    BeamTokenExpansion         BEAST's step as a MutationBase: 1 parent -> k children

    build_input_ids            head + adv suffix + tail + target ids, and their slices
    compute_loss_and_grads     forward + backward -> (loss, d loss / d one-hot suffix)
    get_embedding_matrix       the model's token embedding weights
    sample_candidates          gradient top-k -> B candidate suffixes, one swap each
    score_candidates           teacher-forced CE for every candidate, batched
    build_not_allowed_ids      the non-ASCII / special-token exclusion set

A candidate carries its adversarial span in ``attack_attrs["adv_ids"]``
(``selector.ADV_IDS``), and ``selector.TokenLossSelector`` scores a batch of them by
concatenation alone — no per-candidate re-tokenization. That is what makes the carrier
affordable here: the objection was never the ``Instance`` wrapper, it was re-encoding a
prompt 512 times per step.

**Extracted from our own ``attacks/gcg.py``, not from upstream.** EasyJailbreak's
``MutationTokenGradient`` is written against their ``WhiteBoxModelBase`` — the layer our
``Victim`` contract replaces — so porting it would have meant porting their model stack.
These functions are byte-identical to the ones ``attacks/gcg.py`` has always used; that
file now imports them from here, so there is one implementation.

Why these are ``MutationBase`` subclasses over a *token* field
--------------------------------------------------------------
Every other operator is 1-to-n over ``Instance`` text. These are 1-to-B over token ids,
at ~512 candidates per step. Wrapping each candidate in an ``Instance`` and selecting
with ``ReferenceLossSelector`` would re-render and re-tokenize the prompt once per
candidate — correct, and roughly two orders of magnitude slower. ``TokenLossSelector``
removes that cost by keeping the prefix and suffix as fixed id lists, so the recipes get
the component shape at the price of an object allocation per candidate.

``MutationBase.mutate`` guards *text* (empty output, a dropped ``{query}``), which does
not apply to an id list, so both classes override ``_get_mutated_instance`` directly and
leave ``_mutate`` unused. They still go through ``new_child``, so lineage is recorded the
same way as everywhere else.

``torch`` is imported at module load, so this module is torch-gated like the white-box
attacks: ``ipi/mutation/__init__.py`` must not import it eagerly.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from ..selector.token_loss import ADV_IDS
from .base import MutationBase

log = logging.getLogger(__name__)

__all__ = [
    "TokenGradientMutation",
    "BeamTokenExpansion",
    "sample_top_p",
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
    adv_suffix_ids: list[int],
    target_str: str,
    device,
    head_text: str,
    tail_text: str = "",
    add_special_tokens: bool = False,
) -> tuple[torch.Tensor, slice, slice]:
    """
    Assemble the full input token sequence around the adversarial suffix.

        encode(head_text) + adv_suffix_ids + encode(tail_text) + encode(target_str)

    ``head_text`` / ``tail_text`` come from ``harness.split_optimization_prompt`` (the
    real victim prompt, carrier and defense included) or ``llm_unified.bare_prompt_split``
    (a plain ``[system][user]`` pair). Both are rendered from one
    ``apply_chat_template`` call and split on a marker, which is what puts the suffix
    *inside* the user turn.

    It used to build its own ``[system][user: adv_prefix]`` prompt with
    ``add_generation_prompt=True`` and append the suffix after it. That was wrong twice
    over: the suffix landed in the *assistant* turn, after the generation prompt, and
    the prompt had none of the IPI carrier the victim is actually given. Both failures
    are silent — the loss goes down, the attack just is not attacking the right string.

    Returns:
        input_ids: (1, L) long tensor on `device`.
        suffix_slice: slice into input_ids for the adv_suffix tokens.
        target_slice: slice into input_ids for the target_str tokens.
    """
    head_ids:   list[int] = tokenizer.encode(head_text, add_special_tokens=add_special_tokens)
    tail_ids:   list[int] = tokenizer.encode(tail_text, add_special_tokens=False) if tail_text else []
    target_ids: list[int] = tokenizer.encode(target_str, add_special_tokens=False)

    full_ids = head_ids + adv_suffix_ids + tail_ids + target_ids

    suffix_start = len(head_ids)
    suffix_end   = suffix_start + len(adv_suffix_ids)
    target_start = suffix_end + len(tail_ids)
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

        tgt_ids = input_ids[0, target_slice]                       # (T,)
        T = len(tgt_ids)

        # The target span sits at the end of the sequence, so only the last T+1 logit
        # positions are ever read. Asking the model for just those is ~46x less
        # activation memory at a typical prompt length — the difference between GCG
        # running and OOMing on a 16 GB card.
        from ..selector.token_loss import forward_last_logits
        logits = forward_last_logits(model, batch_ids, T + 1)      # (b, T+1, V)
        tgt_logits = logits[:, :T, :]                              # (b, T, V)
        tgt_ids_exp = tgt_ids.unsqueeze(0).expand(b, -1)           # (b, T)

        loss = F.cross_entropy(
            tgt_logits.reshape(b * T, -1),
            tgt_ids_exp.reshape(b * T),
            reduction="none",
        ).reshape(b, T).mean(dim=1)                                # (b,)

        losses.extend(loss.tolist())

        del batch_ids, logits
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


@torch.no_grad()
def sample_top_p(probs: torch.Tensor, p: float, return_tokens: int = 0) -> torch.Tensor:
    """
    Top-p (nucleus) sampling. Returns the sampled token indices.

    Exact port of ``arutils.sample_top_p``. Lived in ``attacks/beast.py`` until
    ``BeamTokenExpansion`` needed it too.

    Args:
        probs:         Softmax probabilities, ``(batch, vocab)``.
        p:             Nucleus mass to keep.
        return_tokens: Tokens to draw per row. 0 means 1.

    Returns:
        ``(batch, max(1, return_tokens))`` of token indices.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_cum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_cum - probs_sort) > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=max(1, return_tokens))
    return torch.gather(probs_idx, -1, next_token)


# ---------------------------------------------------------------------------
# The operators
# ---------------------------------------------------------------------------

class _TokenMutation(MutationBase):
    """
    Shared plumbing for the two token operators.

    ``attr_name`` is nominal — neither writes an ``Instance`` text field. What they
    rewrite is ``attack_attrs[ADV_IDS]``, and ``_child`` is the one place that happens,
    so the lineage edge and the span are always set together.
    """

    attr_name = "jailbreak_prompt"

    @staticmethod
    def _adv_ids(instance) -> list[int]:
        ids = instance.attack_attrs.get(ADV_IDS)
        if ids is None:
            raise ValueError(
                f"instance {instance.id!r} has no attack_attrs[{ADV_IDS!r}]; seed the "
                "search with one before mutating")
        return list(ids)

    def _child(self, parent, adv_ids: list[int]):
        child = self.new_child(parent)
        child.attack_attrs[ADV_IDS] = list(adv_ids)
        child.target_responses = []
        child.eval_results = []
        return child

    def _mutate(self, text: str, **kwargs) -> str:      # pragma: no cover - unused
        raise NotImplementedError(
            f"{type(self).__name__} rewrites token ids, not text; use the dataset path")


class TokenGradientMutation(_TokenMutation):
    """
    GCG's step: one parent, ``batch_size`` children, each one token-swap away.

    A forward+backward pass gives the gradient of the target loss w.r.t. the one-hot
    adversarial tokens; each child picks a random position and one of the top-``top_k``
    tokens the gradient favours there. The *selection* among them is
    ``selector.TokenLossSelector`` — this operator proposes, it does not score.

    Args:
        model:      The victim's HF model.
        head_ids / tail_ids / target_ids: the fixed spans around the adversarial one,
                    from ``harness.split_optimization_prompt``.
        device:     Where to build the tensors.
        batch_size: Children per call. GCG's default is 512.
        top_k:      Gradient-favoured tokens considered per position. Default 256.
        not_allowed_ids: Token ids excluded from substitution (non-ASCII / specials).
    """

    def __init__(self, model, head_ids: list[int], tail_ids: list[int],
                 target_ids: list[int], device, batch_size: int = 512,
                 top_k: int = 256, not_allowed_ids=None, attr_name: Optional[str] = None):
        super().__init__(attr_name)
        self.model = model
        self.head_ids = list(head_ids)
        self.tail_ids = list(tail_ids)
        self.target_ids = list(target_ids)
        self.device = device
        self.batch_size = batch_size
        self.top_k = top_k
        self.not_allowed_ids = not_allowed_ids
        #: Set by each call — the parent's own loss, which the recipe reports.
        self.last_loss: float = float("inf")

    def _build_ids(self, adv_ids: list[int]):
        """``(input_ids, suffix_slice, target_slice)`` for one candidate."""
        full = self.head_ids + adv_ids + self.tail_ids + self.target_ids
        start = len(self.head_ids)
        suffix_slice = slice(start, start + len(adv_ids))
        tgt_start = suffix_slice.stop + len(self.tail_ids)
        target_slice = slice(tgt_start, tgt_start + len(self.target_ids))
        return (torch.tensor([full], dtype=torch.long, device=self.device),
                suffix_slice, target_slice)

    def _get_mutated_instance(self, instance, **kwargs) -> list:
        adv_ids = self._adv_ids(instance)
        input_ids, suffix_slice, target_slice = self._build_ids(adv_ids)

        self.model.zero_grad()
        loss, grads = compute_loss_and_grads(
            model=self.model, input_ids=input_ids,
            suffix_slice=suffix_slice, target_slice=target_slice)
        self.last_loss = float(loss)

        candidates = sample_candidates(
            grads=grads,
            suffix_ids=torch.tensor(adv_ids, dtype=torch.long, device=self.device),
            top_k=self.top_k, batch_size=self.batch_size,
            not_allowed_ids=self.not_allowed_ids)

        children = [self._child(instance, row) for row in candidates.tolist()]
        del input_ids, grads
        gc.collect()
        return children


class BeamTokenExpansion(_TokenMutation):
    """
    BEAST's step: one parent, ``k`` children, each one sampled token longer.

    A single forward pass over ``head + adv`` gives the next-token distribution; ``k``
    tokens are drawn from its top-p nucleus and each becomes a child. Upstream calls
    this ``k2``. The beam pruning that follows is ``selector.TokenLossSelector(keep=k1)``.

    The tokens are sampled from the *unclosed* user turn — no generation prompt in
    ``head_ids`` — so they extend the user message, which is where the victim will read
    them. That is the whole point of the head/tail split.

    Args:
        victim:  A local ``Victim`` (needs ``generate_n_tokens_batch``).
        head_ids: Ids before the adversarial span.
        k:       Children per parent (upstream's ``k2``). Default 15.
        top_p / top_k / temperature: sampling controls, passed straight through.
    """

    def __init__(self, victim, head_ids: list[int], k: int = 15, top_p: float = 1.0,
                 top_k: Optional[int] = None, temperature: float = 1.0,
                 attr_name: Optional[str] = None):
        super().__init__(attr_name)
        self.victim = victim
        self.head_ids = list(head_ids)
        self.k = k
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def __call__(self, dataset, **kwargs):
        """
        Batched across the whole beam — one forward pass for every parent at once.

        The default ``MutationBase.__call__`` walks instances one at a time; for BEAST
        that would be ``k1`` forward passes per step instead of one, which is most of
        the attack's cost.
        """
        from ..datasets import AttackDataset

        parents = list(dataset)
        if not parents:
            return AttackDataset([])

        seqs = [self.head_ids + self._adv_ids(p) for p in parents]
        max_bs = getattr(self.victim, "max_bs", 50) or 50
        sampled: list[list[int]] = []
        for start in range(0, len(seqs), max_bs):
            batch = seqs[start: start + max_bs]
            logits, _ = self.victim.generate_n_tokens_batch(
                batch, max_gen_len=1, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k)
            probs = torch.softmax(logits[:, 0], dim=-1)
            drawn = sample_top_p(probs, self.top_p, return_tokens=self.k)
            sampled.extend(drawn[:, : self.k].tolist())

        children = []
        for parent, next_ids in zip(parents, sampled):
            adv = self._adv_ids(parent)
            children.extend(self._child(parent, adv + [int(t)]) for t in next_ids)
        return AttackDataset(children)

    def _get_mutated_instance(self, instance, **kwargs) -> list:
        from ..datasets import AttackDataset
        return list(self(AttackDataset([instance])))
