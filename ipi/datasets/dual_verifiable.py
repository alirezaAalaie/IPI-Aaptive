"""
DualVerifiableDataset — the IPI benchmark, loaded as ``Instance`` objects.

"Dual-verifiable" means every scenario carries **two** deterministic ground truths, so
neither utility nor ASR needs an LLM to adjudicate:

  1. ``user_target``  — the correct answer to the legitimate user task, for utility.
  2. ``target_str``   — the literal string a successful injection produces, for ASR,
     plus ``optimization_target``, a real token sequence for gradient/search attacks
     (GCG, RS, Beam-RS, BEAST) that must never be a sentinel.

Built by ``scripts/build_dual_verifiable_dataset.py``; ships in ``ipi/data/``.

Field mapping onto the carrier
------------------------------
The JSON on disk is unchanged — the translation happens here.

    injection_goal      -> query                 (fills a seed template's {query})
    target_str          -> reference_responses   (Match / PrefixExactMatch read this)
    id                  -> id
    everything else     -> attack_attrs

``query`` is the *attacker's* goal, not the user task: it is the string a seed template
wraps, exactly as upstream's harmful request is. The user task is IPI-specific context
and lives in ``attack_attrs``.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .attack_dataset import AttackDataset
from .instance import Instance

__all__ = ["DualVerifiableDataset", "load_dual_verifiable"]

_DATA_PACKAGE = "ipi.data"
_DATA_FILENAME = "dual_verifiable_dataset.json"


def _locate_dataset() -> str:
    """Find the packaged JSON, falling back to the working tree."""
    try:
        from importlib.resources import files
        candidate = files(_DATA_PACKAGE) / _DATA_FILENAME
        if candidate.is_file():
            return str(candidate)
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        pass

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(here, "data", _DATA_FILENAME)
    if os.path.exists(candidate):
        return candidate
    return os.path.join("ipi", "data", _DATA_FILENAME)


def _record_to_instance(s: dict) -> Instance:
    """Translate one JSON record into an Instance."""
    meta = s.get("metadata", {}) or {}
    return Instance(
        id=s["id"],
        query=s["injection_goal"],
        reference_responses=[s["target_str"]],
        attack_attrs={
            # Attacker side
            "target_str":          s["target_str"],
            "optimization_target": s["optimization_target"],
            "attack_eval_mode":    s.get("attack_eval_mode", "contains"),
            # User side (utility)
            "user_task":           s["user_task"],
            "user_target":         s["user_target"],
            "user_eval_mode":      s.get("user_eval_mode", "contains"),
            # Context the injection lands in
            "clean_context":       s["clean_context"],
            "pipeline_context":    s.get("pipeline_context", s["clean_context"]),
            "tool_schema":         s.get("tool_schema", ""),
            # Provenance
            "dataset_type":        "dual_verifiable",
            "domain":              meta.get("domain", ""),
            "attack_name":         meta.get("attack_name", ""),
        },
    )


def load_dual_verifiable(json_path: Optional[str] = None) -> AttackDataset:
    """
    Load the dual-verifiable benchmark as an ``AttackDataset``.

    Args:
        json_path: Path to the dataset JSON. ``None`` auto-locates the packaged copy.

    Raises:
        FileNotFoundError: if the dataset is missing (it is a build artifact).
    """
    path = json_path if json_path is not None else _locate_dataset()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dual-verifiable dataset JSON not found at {path}. "
            "Run `python3 scripts/build_dual_verifiable_dataset.py` to construct it."
        )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return AttackDataset([_record_to_instance(r) for r in records])


class DualVerifiableDataset(AttackDataset):
    """
    ``AttackDataset`` preloaded with the dual-verifiable benchmark.

    Exists so callers can write ``DualVerifiableDataset()`` instead of
    ``load_dual_verifiable()``; it is an ordinary AttackDataset in every other respect,
    so slicing, ``subset()`` and ``merge()`` all return plain AttackDatasets.
    """

    def __init__(self, json_path: Optional[str] = None, shuffle: bool = False,
                 seed: Optional[int] = None):
        super().__init__(load_dual_verifiable(json_path), shuffle=shuffle, seed=seed)
