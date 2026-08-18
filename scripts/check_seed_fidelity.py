#!/usr/bin/env python3
"""
Assert every ``original`` entry in the seed registry is byte-identical to the
vendored reference implementation.

The registry (``ipi/seed/seed_templates.json``) carries two variants per method:
``original`` (verbatim from the paper's code) and ``ipi`` (our reframing). Only the
first is checkable against an external source — this script is the audit trail for it,
in the same spirit as ``check_pisanitizer_fidelity.py`` and
``check_defensivetoken_fidelity.py``.

The one permitted transformation is placeholder normalisation: upstream's AutoDAN pool
uses ``[PROMPT]`` where our convention is ``{query}``. That substitution is inverted
before comparison, so a drift in the template text itself still fails.

Two usages are checked against a different source, because upstream inlines those strings
in Python rather than putting them in its seed file: ``mutation`` (the ``jailbreak_prompt``
each rule operator attaches, and the CodeChameleon decryption-function constants) and
``constraint`` (each filter's judge prompt). Both are pulled out of the vendored modules by
AST.

Usage:  python3 scripts/check_seed_fidelity.py
"""
from __future__ import annotations

import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

UPSTREAM = os.path.join(
    ROOT, "code/attack/EasyJailbreak-master/easyjailbreak/seed/seed_template.json")
UPSTREAM_RULES = os.path.join(
    ROOT, "code/attack/EasyJailbreak-master/easyjailbreak/mutation/rule")
UPSTREAM_CONSTRAINTS = os.path.join(
    ROOT, "code/attack/EasyJailbreak-master/easyjailbreak/constraint")

# our constraint.<method> -> (vendored module, the attribute it is assigned to)
CONSTRAINT_SOURCES = {
    "DeleteOffTopic": ("DeleteOffTopic.py", "system_prompt"),
    "DeleteHarmLess": ("DeleteHarmLess.py", "_prompt"),
}

# our mutation.<method> -> (vendored module, what to pull out of it).
# "jailbreak_prompt" = the string the operator assigns to instance.jailbreak_prompt;
# anything else names a module-level constant.
MUTATION_SOURCES = {
    "Base64":                  ("Base64.py",           "jailbreak_prompt"),
    "Base64InputOnly":         ("Base64_input_only.py", "jailbreak_prompt"),
    "Rot13":                   ("Rot13.py",            "jailbreak_prompt"),
    "Combination1":            ("Combination_1.py",    "jailbreak_prompt"),
    "Combination2":            ("Combination_2.py",    "jailbreak_prompt"),
    "Combination3":            ("Combination_3.py",    "jailbreak_prompt"),
    "Translate":               ("Translate.py",        "jailbreak_prompt"),
    # the ProblemSolver wrapper is identical in all four CodeChameleon schemes
    "CodeChameleon":           ("Reverse.py",          "jailbreak_prompt"),
    "CodeChameleonReverse":    ("Reverse.py",          "REVERSE"),
    "CodeChameleonOddEven":    ("OddEven.py",          "ODD_EVEN"),
    "CodeChameleonLength":     ("Length.py",           "LENGTH"),
    "CodeChameleonBinaryTree": ("BinaryTree.py",       "BINARY_TREE"),
}

# our (usage, method) -> upstream (usage, method); None = no upstream counterpart
MAPPING = {
    ("attack", "Gptfuzzer"):     ("attack", "Gptfuzzer"),
    ("attack", "ICA"):           ("attack", "ICA"),
    ("attack", "DeepInception"): ("attack", "DeepInception"),
    ("attack", "ReNeLLM"):       ("attack", "ReNeLLM"),
    ("attack", "AutoDAN"):       ("attack", "AutoDAN-a"),
}

# Templates whose placeholder was renamed on import: our token -> upstream token.
PLACEHOLDER_RENAMES = {("attack", "AutoDAN"): ("{query}", "[PROMPT]")}

# `original` pools with no upstream counterpart to diff against, and why.
NO_UPSTREAM_SOURCE = {
    ("attack", "TAP"):            "reworded in this repo; upstream's is a 2-part variant",
    ("attack", "PAIR"):           "reworded in this repo; upstream's is a 2-part variant",
    ("judge", "PAIR"):            "reworded in this repo",
    ("demo", "ICA"):              "AdvBench sample, not part of upstream's seed file",
}


def _from_module(directory: str, filename: str, what: str) -> str:
    """
    Pull a verbatim string out of a vendored module.

    ``what`` is either an attribute name (``self.<what> = "..."``) or a module-level
    constant name.
    """
    tree = ast.parse(open(os.path.join(directory, filename), encoding="utf-8").read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == what:
                return node.value.value
            if isinstance(target, ast.Name) and target.id == what:
                return node.value.value
    raise AssertionError(f"{filename}: no {what} found")


def check_mutation_wrappers(ours: dict, failures: list[str]) -> int:
    """The ``mutation`` usage, against the vendored rule modules."""
    section = ours.get("mutation", {})
    checked = 0
    for method, variants in sorted(section.items()):
        if method not in MUTATION_SOURCES:
            failures.append(f"mutation.{method} has no fidelity mapping")
            continue
        filename, what = MUTATION_SOURCES[method]
        if not os.path.exists(os.path.join(UPSTREAM_RULES, filename)):
            print(f"skip  mutation.{method} — vendored {filename} not found")
            continue
        upstream = _from_module(UPSTREAM_RULES, filename, what)
        if variants["original"] != [upstream]:
            failures.append(f"mutation.{method}.original differs from {filename}:{what}")
            continue
        checked += 1
        print(f"ok    mutation.{method}.original — verbatim from rule/{filename}")
    missing = sorted(set(MUTATION_SOURCES) - set(section))
    if missing:
        failures.append(f"mutation entries missing from the registry: {missing}")
    return checked


def check_constraint_prompts(ours: dict, failures: list[str]) -> int:
    """The ``constraint`` usage, against the vendored constraint modules."""
    section = ours.get("constraint", {})
    checked = 0
    for method, variants in sorted(section.items()):
        if "original" not in variants:
            continue
        if ("constraint", method) in NO_UPSTREAM_SOURCE:
            print(f"skip  constraint.{method}.original — "
                  f"{NO_UPSTREAM_SOURCE[('constraint', method)]}")
            continue
        if method not in CONSTRAINT_SOURCES:
            failures.append(f"constraint.{method}.original has no fidelity mapping")
            continue
        filename, what = CONSTRAINT_SOURCES[method]
        if not os.path.exists(os.path.join(UPSTREAM_CONSTRAINTS, filename)):
            print(f"skip  constraint.{method} — vendored {filename} not found")
            continue
        upstream = _from_module(UPSTREAM_CONSTRAINTS, filename, what)
        if variants["original"] != [upstream]:
            failures.append(f"constraint.{method}.original differs from {filename}:{what}")
            continue
        checked += 1
        print(f"ok    constraint.{method}.original — verbatim from constraint/{filename}")
    missing = sorted(set(CONSTRAINT_SOURCES) - set(section))
    if missing:
        failures.append(f"constraint entries missing from the registry: {missing}")
    return checked


def main() -> int:
    if not os.path.exists(UPSTREAM):
        print(f"SKIP: vendored upstream not found at {UPSTREAM}")
        return 0

    from ipi.seed import load_seed_templates

    ours = load_seed_templates()
    up = json.load(open(UPSTREAM, encoding="utf-8"))

    failures: list[str] = []
    checked = 0

    for usage, methods in ours.items():
        if usage.startswith("_") or usage in ("mutation", "constraint"):
            continue   # checked separately, against the vendored Python modules
        for method, variants in methods.items():
            if "original" not in variants:
                continue
            key = (usage, method)
            if key in NO_UPSTREAM_SOURCE:
                print(f"skip  {usage}.{method}.original — {NO_UPSTREAM_SOURCE[key]}")
                continue
            if key not in MAPPING:
                failures.append(f"{usage}.{method}.original has no fidelity mapping")
                continue

            up_usage, up_method = MAPPING[key]
            up_pool = up.get(up_usage, {}).get(up_method)
            if up_pool is None:
                failures.append(f"{usage}.{method}: upstream {up_usage}.{up_method} missing")
                continue

            pool = list(variants["original"])
            if key in PLACEHOLDER_RENAMES:
                ours_tok, up_tok = PLACEHOLDER_RENAMES[key]
                pool = [t.replace(ours_tok, up_tok) for t in pool]

            if len(pool) != len(up_pool):
                failures.append(
                    f"{usage}.{method}: {len(pool)} templates, upstream has {len(up_pool)}")
                continue

            diffs = [i for i, (a, b) in enumerate(zip(pool, up_pool)) if a != b]
            if diffs:
                failures.append(
                    f"{usage}.{method}: {len(diffs)} template(s) differ from upstream "
                    f"(first at index {diffs[0]})")
                continue

            checked += len(pool)
            print(f"ok    {usage}.{method}.original — {len(pool)} templates verbatim")

    checked += check_mutation_wrappers(ours, failures)
    checked += check_constraint_prompts(ours, failures)

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print(f"all original pools verbatim ({checked} templates checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
