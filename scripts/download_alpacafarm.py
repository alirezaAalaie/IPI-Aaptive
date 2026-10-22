#!/usr/bin/env python3
"""
Download and prepare the AlpacaFarm prompt injection evaluation dataset.

Source:
  Dubois et al. (2023), "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback"
  Hugging Face: https://huggingface.co/datasets/tatsu-lab/alpaca_farm/raw/main/alpaca_farm_evaluation.json

DefensiveToken Threat Formulation:
  arXiv:2507.07974 ("Defending Against Prompt Injection With a Few DefensiveTokens", Chen et al.)
  Evaluates white-box GCG exclusively on the 208 samples of AlpacaFarm having a non-empty `input` field.

Saves:
  - Raw 805 samples -> `ipi/data/alpacafarm/alpaca_farm_evaluation.json`
  - Filtered 208 samples -> `ipi/data/alpacafarm/alpacafarm_eval_208.json`

Usage:
  python3 scripts/download_alpacafarm.py
  python3 scripts/download_alpacafarm.py --force
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("download_alpacafarm")

ALPACAFARM_EVAL_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_farm/raw/main/alpaca_farm_evaluation.json"
)
RAW_FILENAME = "alpaca_farm_evaluation.json"
FILTERED_208_FILENAME = "alpacafarm_eval_208.json"


def get_default_target_dir() -> str:
    """Return default target dir in ipi/data/alpacafarm/ relative to repo root."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "ipi", "data", "alpacafarm")


def download_file(url: str, dest_path: str, force: bool = False) -> bool:
    """Download url to dest_path if not already present (or if force=True)."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not force:
        log.info("File already exists: %s (%d bytes). Skipping.", dest_path, os.path.getsize(dest_path))
        return True

    log.info("Downloading %s -> %s...", url, dest_path)
    tmp_path = dest_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
        log.info("Successfully downloaded %s (%d bytes).", dest_path, os.path.getsize(dest_path))
        return True
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        log.error("Failed to download %s: %s", url, e)
        return False


def download_alpacafarm(target_dir: Optional[str] = None, force: bool = False) -> bool:
    """Download AlpacaFarm evaluation dataset, filter 208 samples with inputs, and save."""
    target_dir = target_dir or get_default_target_dir()
    os.makedirs(target_dir, exist_ok=True)

    # Ensure __init__.py exists in target_dir so importlib.resources works
    init_path = os.path.join(target_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write('"""AlpacaFarm package data."""\n')

    raw_path = os.path.join(target_dir, RAW_FILENAME)
    filtered_path = os.path.join(target_dir, FILTERED_208_FILENAME)

    # Download raw evaluation json
    if not download_file(ALPACAFARM_EVAL_URL, raw_path, force=force):
        return False

    # Filter to samples with non-empty input
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        log.info("Loaded %d raw AlpacaFarm evaluation records.", len(records))

        filtered_208 = [r for r in records if bool(r.get("input", "").strip())]
        log.info("Filtered %d records with non-empty 'input' field.", len(filtered_208))

        if len(filtered_208) != 208:
            log.warning("Expected exactly 208 samples, got %d!", len(filtered_208))

        if not os.path.exists(filtered_path) or force:
            with open(filtered_path, "w", encoding="utf-8") as f:
                json.dump(filtered_208, f, indent=2, ensure_ascii=False)
            log.info("Saved filtered dataset to %s (%d samples).", filtered_path, len(filtered_208))
        else:
            log.info("Filtered dataset already exists at %s. Skipping (use --force to overwrite).", filtered_path)

        return True
    except Exception as e:
        log.error("Failed to process AlpacaFarm dataset: %s", e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AlpacaFarm benchmark evaluation dataset.")
    parser.add_argument("--target-dir", type=str, default=None, help="Directory to save downloaded files.")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-filtering.")
    args = parser.parse_args()

    success = download_alpacafarm(target_dir=args.target_dir, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
