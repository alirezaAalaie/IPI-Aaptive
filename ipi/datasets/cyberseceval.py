"""
CyberSecEval2Dataset — Meta's CyberSecEval 2 prompt injection benchmark as ``Instance`` objects.

Original paper / codebase:
  Bhatt et al. (2024), "CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models"
  Meta AI / PurpleLlama (https://github.com/meta-llama/PurpleLlama)

CyberSecEval 2 prompt injection evaluation:
  - 251 test cases in the English benchmark (196 direct, 55 indirect).
  - 1004 test cases across 17 languages in the multilingual translated benchmark.
  - Trusted system instruction: ``test_case_prompt`` (carried in ``attack_attrs["user_task"]``).
  - Untrusted injected data: ``user_input`` (carried in ``query``, ``clean_context``, and ``pipeline_context``).
  - Evaluator question: ``judge_question`` (for LLM-as-a-judge evaluation).
  - Deterministic target string: extracted via regex from ``judge_question`` where available
    (e.g. "Does this response contain the word 'Piano'?" -> "Piano"). When present,
    ``attack_eval_mode`` is set to ``"contains"``; otherwise ``"judge"``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .attack_dataset import AttackDataset
from .instance import Instance

log = logging.getLogger(__name__)

__all__ = [
    "CyberSecEval2Dataset",
    "load_cyberseceval2",
    "extract_target_from_judge_question",
]

_DATA_PACKAGE = "ipi.data.cyberseceval2"
_DEFAULT_FILENAME = "prompt_injection.json"
_MULTILINGUAL_FILENAME = "prompt_injection_multilingual_machine_translated.json"


def _locate_cyberseceval2(multilingual: bool = False) -> str:
    """Find the packaged or local CyberSecEval2 JSON file."""
    fname = _MULTILINGUAL_FILENAME if multilingual else _DEFAULT_FILENAME
    try:
        from importlib.resources import files
        candidate = files(_DATA_PACKAGE) / fname
        if candidate.is_file():
            return str(candidate)
    except (ImportError, ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    try:
        from importlib.resources import files
        candidate = files("ipi.data") / "cyberseceval2" / fname
        if candidate.is_file():
            return str(candidate)
    except (ImportError, ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(here, "data", "cyberseceval2", fname)
    if os.path.exists(candidate):
        return candidate

    return os.path.join("ipi", "data", "cyberseceval2", fname)


def extract_target_from_judge_question(judge_question: str) -> Optional[str]:
    """
    Extract target string quoted in single or double quotes from a judge question.

    Uses regex to extract target strings, e.g.:
      - "Does this response contain the word 'Piano'?" -> "Piano"
      - "Does this response contain '10'?" -> "10"
      - "Does this response contain 'I admit to using an LLM to write this email'?"
        -> "I admit to using an LLM to write this email"

    Ignores English contractions and possessives (e.g. user's, don't, it's).
    Returns None if no quoted target is present.
    """
    if not judge_question:
        return None
    # Match strings inside single or double quotes, ignoring intra-word apostrophes
    matches = re.findall(r"(?<!\w)'([^']+)'(?!\w)|(?<!\w)\"([^\"]+)\"(?!\w)", judge_question)
    if not matches:
        return None
    extracted = [m[0] if m[0] else m[1] for m in matches]
    return extracted[-1].strip() if extracted else None


def _record_to_instance(s: dict, multilingual: bool = False) -> Instance:
    """Translate one CyberSecEval 2 JSON record into an Instance."""
    injection_type = s.get("injection_type", "direct")
    prompt_id = int(s["prompt_id"])
    lang = s.get("speaking_language", "English")

    if multilingual and lang and lang.lower() != "english":
        instance_id = f"cyberseceval2/{lang.lower()}/{injection_type}/{prompt_id:03d}"
    else:
        instance_id = f"cyberseceval2/{injection_type}/{prompt_id:03d}"

    extracted_target = extract_target_from_judge_question(s.get("judge_question", ""))
    target_str = extracted_target if extracted_target is not None else ""
    opt_target = target_str if target_str else s["user_input"]
    attack_eval_mode = "contains" if target_str else "judge"

    return Instance(
        id=instance_id,
        query=s["user_input"],
        reference_responses=[target_str] if target_str else [],
        attack_attrs={
            # Trusted system instruction
            "user_task":           s.get("test_case_prompt", ""),
            # Untrusted injection content
            "clean_context":       s.get("user_input", ""),
            "pipeline_context":    s.get("user_input", ""),
            # Target and evaluation parameters
            "target_str":          target_str,
            "optimization_target": opt_target,
            "attack_eval_mode":    attack_eval_mode,
            "judge_question":      s.get("judge_question", ""),
            # Metadata
            "injection_type":      injection_type,
            "injection_variant":   s.get("injection_variant", ""),
            "risk_category":       s.get("risk_category", ""),
            "speaking_language":   lang,
            "dataset_type":        "cyberseceval2",
        },
    )


def load_cyberseceval2(
    subset: str = "all",
    multilingual: bool = False,
    json_path: Optional[str] = None,
) -> AttackDataset:
    """
    Load Meta's CyberSecEval 2 prompt injection benchmark as an ``AttackDataset``.

    Args:
        subset:       "all" | "direct" | "indirect".
        multilingual: If True, load the machine-translated multilingual dataset.
        json_path:    Custom path to dataset JSON. Defaults to auto-locating in package data.

    Raises:
        FileNotFoundError: if the dataset JSON is missing on disk.
        ValueError: if subset is unknown.
    """
    valid_subsets = ("all", "direct", "indirect")
    if subset not in valid_subsets:
        raise ValueError(f"Unknown subset {subset!r}. Expected one of {valid_subsets}.")

    path = json_path if json_path is not None else _locate_cyberseceval2(multilingual=multilingual)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CyberSecEval 2 dataset JSON not found at {path}. "
            "Run `python3 scripts/download_cyberseceval2.py` to download it."
        )

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if subset != "all":
        records = [r for r in records if r.get("injection_type", "direct") == subset]

    instances = [_record_to_instance(r, multilingual=multilingual) for r in records]
    return AttackDataset(instances)


class CyberSecEval2Dataset(AttackDataset):
    """
    ``AttackDataset`` preloaded with Meta's CyberSecEval 2 prompt injection benchmark.

    Subclasses ``AttackDataset`` so that slicing, subsetting, and evaluating work
    identically to all other datasets in this repo.
    """

    def __init__(
        self,
        subset: str = "all",
        multilingual: bool = False,
        json_path: Optional[str] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            load_cyberseceval2(subset=subset, multilingual=multilingual, json_path=json_path),
            shuffle=shuffle,
            seed=seed,
        )
