#!/usr/bin/env python3
"""
Assert that the ported DefensiveToken chat templates are byte-identical to the
vendored upstream ones.

The templates are the defense's contract with the released embeddings: they are
*not* the models' stock chat templates, and a single character of drift moves
the optimised tokens into a format they were never evaluated in — silently, with
no error and a plausible-looking ASR. ``ipi/defenses/defensive_token/config.py``
therefore builds them by concatenation instead of spelling them out; this script
checks that construction against ``code/defense/DefensiveToken-main/setup.py``.

Usage::

    python3 scripts/check_defensivetoken_fidelity.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "code" / "defense" / "DefensiveToken-main" / "setup.py"

sys.path.insert(0, str(REPO))
from ipi.defenses.defensive_token.config import (  # noqa: E402
    CHAT_TEMPLATES,
    DEFENSIVE_TOKEN_NAMES,
    NUM_DEFENSIVE_TOKENS,
    SUPPORTED_MODELS,
)


def upstream_templates() -> dict[str, str]:
    """Read upstream's ``chat_templates`` dict without executing the module
    (it imports torch and reads a 1.7 MB weights file at import time)."""
    tree = ast.parse(UPSTREAM.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "chat_templates":
            return ast.literal_eval(node.value)
    raise SystemExit(f"No chat_templates assignment found in {UPSTREAM}")


def main() -> int:
    if not UPSTREAM.exists():
        print(f"SKIP: vendored upstream not found at {UPSTREAM}")
        return 0

    upstream = upstream_templates()
    failures = 0

    missing = set(upstream) ^ set(CHAT_TEMPLATES)
    if missing:
        print(f"FAIL: model-key mismatch: {sorted(missing)}")
        failures += 1

    for key in sorted(set(upstream) & set(CHAT_TEMPLATES)):
        if upstream[key] == CHAT_TEMPLATES[key]:
            print(f"  ok   {key}")
        else:
            failures += 1
            print(f"  FAIL {key}")
            print(f"       upstream: {upstream[key]!r}")
            print(f"       ported:   {CHAT_TEMPLATES[key]!r}")

    if set(SUPPORTED_MODELS) != set(upstream):
        failures += 1
        print(f"FAIL: SUPPORTED_MODELS disagrees with upstream: {SUPPORTED_MODELS}")

    if len(DEFENSIVE_TOKEN_NAMES) != NUM_DEFENSIVE_TOKENS:
        failures += 1
        print("FAIL: DEFENSIVE_TOKEN_NAMES length disagrees with NUM_DEFENSIVE_TOKENS")

    for name in DEFENSIVE_TOKEN_NAMES:
        for template in CHAT_TEMPLATES.values():
            if name not in template:
                failures += 1
                print(f"FAIL: {name} absent from a chat template")
                break

    print("\nFAILED" if failures else f"\nOK — {len(upstream)} templates byte-identical")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
