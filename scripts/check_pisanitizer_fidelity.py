#!/usr/bin/env python3
"""
Differential test of the PISanitizer port against the vendored upstream.

The parts that can be checked without a GPU are the two that decide *what gets
deleted*: the peak-grouping logic and the detection-prompt construction. Both
are ported rather than copied, so this runs upstream's implementation and this
repo's side by side on the same inputs and requires identical output.

The attention reconstruction itself (``attention.py``) needs real model weights
and is exercised by ``experiments/pisanitizer_defense_notebook.ipynb``.

Usage::

    python3 scripts/check_pisanitizer_fidelity.py
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO / "code" / "defense" / "PISanitizer-main"

sys.path.insert(0, str(REPO))
from ipi.defenses.pisanitizer.config import (  # noqa: E402
    ANCHOR_SUFFIX,
    CONTEXT_PREFIX,
    LLAMA3_DELIMITERS,
    MAX_ITERATIONS,
    PADDING_TOKEN,
    SANITIZATION_INSTRUCTIONS,
)
from ipi.defenses.pisanitizer.peaks import group_peaks as ported_group_peaks  # noqa: E402


def load_upstream_module(name: str, path: Path):
    """Import a single upstream file without importing its package (which pulls
    in torch and transformers)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signals():
    """A spread of shapes: flat, single spike, clustered spikes, noise."""
    rng = random.Random(0)
    yield "flat", [0.001] * 300
    yield "below-threshold", [0.004 + 0.0005 * (i % 7) for i in range(300)]
    yield "one-peak", [0.001] * 140 + [0.05, 0.09, 0.12, 0.09, 0.05] + [0.001] * 155
    yield "two-peaks-far", (
        [0.001] * 60 + [0.04, 0.08, 0.04] + [0.001] * 100
        + [0.05, 0.11, 0.05] + [0.001] * 134
    )
    yield "two-peaks-near", (
        [0.001] * 100 + [0.04, 0.08, 0.04] + [0.002] * 6
        + [0.05, 0.07, 0.05] + [0.001] * 182
    )
    yield "short", [0.002, 0.05, 0.09, 0.03, 0.002]
    for trial in range(12):
        yield f"random-{trial}", [rng.random() * 0.06 for _ in range(rng.randint(30, 600))]
    for trial in range(8):
        n = rng.randint(200, 800)
        sig = [rng.random() * 0.004 for _ in range(n)]
        start = rng.randint(10, n - 40)
        for j in range(start, start + rng.randint(8, 30)):
            sig[j] += rng.random() * 0.15
        yield f"planted-{trial}", sig


def main() -> int:
    if not UPSTREAM_DIR.exists():
        print(f"SKIP: vendored upstream not found at {UPSTREAM_DIR}")
        return 0

    upstream = load_upstream_module(
        "_upstream_group_peaks", UPSTREAM_DIR / "PISanitizer" / "group_peaks.py"
    )

    failures = 0
    checked = 0

    # --- 1. group_peaks differential --------------------------------------
    for name, sig in signals():
        for smooth_win in (5, 9):
            for max_gap in (10, 20):
                for threshold in (0.01, 0.025):
                    want_smooth, want_spans = upstream.group_peaks(
                        sig, smooth_win=smooth_win, max_gap=max_gap, threshold=threshold
                    )
                    got_smooth, got_spans = ported_group_peaks(
                        sig, smooth_win=smooth_win, max_gap=max_gap, threshold=threshold
                    )
                    checked += 1
                    label = f"{name} win={smooth_win} gap={max_gap} thr={threshold}"
                    if list(want_spans) != list(got_spans):
                        failures += 1
                        print(f"  FAIL spans   {label}: {want_spans} != {got_spans}")
                    elif len(want_smooth) != len(got_smooth) or any(
                        abs(a - b) > 1e-9 for a, b in zip(want_smooth, got_smooth)
                    ):
                        failures += 1
                        print(f"  FAIL smooth  {label}")
    print(f"  group_peaks: {checked} configurations checked")

    # --- 2. group_consecutive_peaks differential --------------------------
    from ipi.defenses.pisanitizer.peaks import group_consecutive_peaks
    rng = random.Random(1)
    for _ in range(200):
        peaks = sorted(rng.sample(range(0, 500), rng.randint(0, 20)))
        for gap in (1, 5, 10, 20):
            if upstream.group_consecutive_peaks(peaks, gap) != group_consecutive_peaks(peaks, gap):
                failures += 1
                print(f"  FAIL group_consecutive_peaks({peaks}, {gap})")
    print("  group_consecutive_peaks: 800 cases checked")

    # --- 3. detection-prompt construction ---------------------------------
    up_pi = (UPSTREAM_DIR / "PISanitizer" / "pisanitizer.py").read_text()

    if tuple(eval_delims(up_pi)) != tuple(LLAMA3_DELIMITERS):
        failures += 1
        print(f"  FAIL delimiters drifted:\n"
              f"       upstream: {eval_delims(up_pi)}\n"
              f"       ported:   {list(LLAMA3_DELIMITERS)}")

    # anchor_prompt1/2 are local string literals; read them out of the AST so
    # escape sequences are decoded the same way Python decodes them.
    up_anchors = eval_locals(up_pi, "pisanitizer", ("anchor_prompt1", "anchor_prompt2"))
    for got, want, label in (
        (SANITIZATION_INSTRUCTIONS[0], up_anchors["anchor_prompt1"], "anchor prompt 0"),
        (ANCHOR_SUFFIX, up_anchors["anchor_prompt2"], "anchor suffix"),
    ):
        if got != want:
            failures += 1
            print(f"  FAIL {label} drifted:\n       upstream: {want!r}\n       ported:   {got!r}")

    for literal, label in (
        (CONTEXT_PREFIX, "context prefix"),
        (PADDING_TOKEN, "padding token"),
    ):
        if literal not in up_pi:
            failures += 1
            print(f"  FAIL {label} drifted from upstream")

    # Upstream's loop guard is `if iteration > 4: break` before `iteration += 1`,
    # which allows MAX_ITERATIONS rounds.
    if f"iteration > {MAX_ITERATIONS - 1}" not in up_pi:
        failures += 1
        print(f"  FAIL iteration cap: upstream does not break at > {MAX_ITERATIONS - 1}")

    # The padding offsets are part of the method, not tuning knobs.
    if f'" X" * 500' not in up_pi:
        failures += 1
        print("  FAIL padding offset is no longer 500 upstream")
    print("  detection prompt: delimiters, anchors, padding and iteration cap checked")

    # --- 4. the batch-path anchor list must match -------------------------
    up_methods = (UPSTREAM_DIR / "methods" / "pisanitizer.py").read_text()
    for i, instruction in enumerate(SANITIZATION_INSTRUCTIONS):
        if instruction and instruction not in up_methods:
            failures += 1
            print(f"  FAIL SANITIZATION_INSTRUCTIONS[{i}] drifted from upstream")
    print("  anchor prompt list: checked against methods/pisanitizer.py")

    print("\nFAILED" if failures else "\nOK — port matches upstream")
    return 1 if failures else 0


def eval_locals(source: str, func_name: str, names: tuple[str, ...]) -> dict:
    """Pull constant local assignments out of a function without running it."""
    import ast

    found: dict = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    target = getattr(stmt.targets[0], "id", "")
                    if target in names and isinstance(stmt.value, ast.Constant):
                        found[target] = stmt.value.value
    missing = set(names) - set(found)
    if missing:
        raise SystemExit(f"Could not read {sorted(missing)} from upstream {func_name}()")
    return found


def eval_delims(source: str):
    """Pull upstream's DELIMITERS list out of its source without importing it."""
    import ast

    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DELIMITERS":
            return ast.literal_eval(node.value)
    raise SystemExit("No DELIMITERS in upstream pisanitizer.py")


if __name__ == "__main__":
    raise SystemExit(main())
