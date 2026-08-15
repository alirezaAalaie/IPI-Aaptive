"""
PISanitizer configuration — anchor prompts, delimiters, signal defaults.

Direct port of ``code/defense/PISanitizer-main/{PISanitizer,methods}/pisanitizer.py``.

The method in one paragraph: build a *detection prompt* that deliberately orders
the model to obey whatever instruction it finds in the context, take one
generation step, and read off how much attention that first generated token pays
to each context token. An injected instruction is, by construction, the thing
that most compels the model to follow it — so it is the thing that spikes. Smooth
the per-token signal, find its peaks, and delete the highest-scoring span. Repeat
up to five times. The attacker's dilemma: the harder the injection pulls, the
more certainly it is cut out.

Load-bearing details
--------------------
* **The anchor prompt must invite instruction-following.** ``anchor_prompt1``
  ("Do anything you are told to do in the following context") is not a defensive
  instruction — it is the opposite, and reversing it inverts the signal the
  method reads. The alternatives in :data:`SANITIZATION_INSTRUCTIONS` are the
  paper's ablations, selected by ``anchor_prompt`` index.
* **The ``" X" * 500`` padding on both sides of the context is part of the
  method**, not filler. It buffers the context away from the prompt boundaries,
  where attention mass concentrates for reasons that have nothing to do with
  injection (the attention-sink effect). Shrinking it lets sink attention leak
  into the scored region.
* **The delimiters are raw Llama-3 chat markers, and one is malformed** —
  ``<|eot_id|><|begin_of_text|><|start_header_id|>user...`` re-emits BOS
  mid-conversation. That is upstream's string and every published number was
  produced with it, so it is reproduced verbatim. It also means the defaults
  only make sense for a Llama-3 family sanitizer.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

#: Sanitizer model upstream uses everywhere, including for the reported numbers.
DEFAULT_SANITIZER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Raw Llama-3 chat markers, verbatim from upstream (note the stray second
#: ``<|begin_of_text|>`` in the user delimiter — preserved deliberately).
LLAMA3_DELIMITERS: Tuple[str, str, str] = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n",
)

#: Prompts that make the sanitizer model *want* to follow embedded instructions.
#: Index 3 substitutes the real user task; index 4 is the no-anchor ablation.
SANITIZATION_INSTRUCTIONS: List[str] = [
    "Do anything you are told to do in the following context.",
    "Your task is to read the following long, noisy text. Somewhere inside, "
    "there is a clear instruction for you. Ignore irrelevant content and do "
    "only what that instruction asks.",
    "You need to find the user task in the following noisy context and strictly follow it.",
    "{target_inst}",
    "",
]

#: Closing instruction appended after the context.
ANCHOR_SUFFIX = (
    "\n\nOnly do what the user asks you to do. Do not provide any explanation. "
    "Your response:"
)

#: Prefix introducing the context inside the user turn.
CONTEXT_PREFIX = "Context: "

#: Padding token repeated on both sides of the context to hold it away from the
#: prompt boundaries, where attention sinks live.
PADDING_TOKEN = " X"

#: Hard cap on sanitization rounds (upstream: ``if iteration > 4: break``).
MAX_ITERATIONS = 5

#: How the per-layer, per-head attention is reduced to one score per token.
#: Names read ``<across-layers>-<across-heads>``; ``max-avg`` is upstream's
#: default and the one the paper reports.
SIGNAL_MODES = ("avg-max", "max-max", "top5-max", "avg-avg", "max-avg", "top5-avg")

DEFAULT_CONFIG: Dict[str, object] = {
    "mode": "max-avg",
    "start_offset": 500,      # count of PADDING_TOKEN repeats before the context
    "end_offset": 500,        # ... and after it
    "anchor_prompt": 0,       # index into SANITIZATION_INSTRUCTIONS
    "smooth_win": None,       # None -> 5 for contexts < 500 tokens, else 9
    "max_gap": 10,            # peaks closer than this join into one span
    "threshold": 0.01,        # a span survives only if it reaches this height
}

#: Peak-finding parameters (group_peaks.py defaults). Separated from
#: DEFAULT_CONFIG because upstream never varies them.
PEAK_PARAMS: Dict[str, float] = {
    "prominence": 0.0,
    "distance": 5,
    "height": 0.005,
    "rel_height": 0.95,
}

#: Savitzky-Golay window by context length, and the polynomial order.
SMOOTH_WIN_SHORT = 5
SMOOTH_WIN_LONG = 9
SMOOTH_LENGTH_CUTOFF = 500
SMOOTH_POLYORDER = 2


def resolve_config(overrides: Dict[str, object] | None = None) -> Dict[str, object]:
    """Merge user overrides onto :data:`DEFAULT_CONFIG`, validating the mode.

    Returns a fresh dict every call — upstream mutates a module-level config
    when it fills in ``smooth_win``, which leaks the first sample's smoothing
    window into every later sample in a batch run.
    """
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        unknown = set(overrides) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(
                f"Unknown PISanitizer config keys: {sorted(unknown)}. "
                f"Known: {sorted(DEFAULT_CONFIG)}."
            )
        cfg.update(overrides)

    if cfg["mode"] not in SIGNAL_MODES:
        raise ValueError(f"mode must be one of {SIGNAL_MODES}, got {cfg['mode']!r}")

    idx = cfg["anchor_prompt"]
    if not isinstance(idx, int) or not 0 <= idx < len(SANITIZATION_INSTRUCTIONS):
        raise ValueError(
            f"anchor_prompt must be an index into SANITIZATION_INSTRUCTIONS "
            f"(0..{len(SANITIZATION_INSTRUCTIONS) - 1}), got {idx!r}."
        )
    return cfg


def smoothing_window(n_tokens: int, configured: int | None) -> int:
    """Savitzky-Golay window: explicit value, else 5/9 by context length."""
    if configured is not None:
        return int(configured)
    return SMOOTH_WIN_SHORT if n_tokens < SMOOTH_LENGTH_CUTOFF else SMOOTH_WIN_LONG
