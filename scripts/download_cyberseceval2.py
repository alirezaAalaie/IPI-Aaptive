#!/usr/bin/env python3
"""
Download Meta's CyberSecEval 2 prompt injection benchmark datasets.

Downloads:
  1. prompt_injection.json (English benchmark, 251 samples: 196 direct, 55 indirect)
  2. prompt_injection_multilingual_machine_translated.json (Multilingual benchmark, 1004 samples across 17 languages)

Saves to `ipi/data/cyberseceval2/` by default.

Usage:
  python3 scripts/download_cyberseceval2.py
  python3 scripts/download_cyberseceval2.py --force
"""
from __future__ import annotations

import argparse
import logging
import os
import urllib.request
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("download_cyberseceval2")

BASE_URL = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets/prompt_injection"
DATASETS = {
    "prompt_injection.json": f"{BASE_URL}/prompt_injection.json",
    "prompt_injection_multilingual_machine_translated.json": f"{BASE_URL}/prompt_injection_multilingual_machine_translated.json",
}


def get_default_target_dir() -> str:
    """Return default target dir in ipi/data/cyberseceval2/ relative to repo root."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "ipi", "data", "cyberseceval2")


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


def download_cyberseceval2(target_dir: Optional[str] = None, force: bool = False) -> bool:
    """Download all CyberSecEval 2 datasets into target_dir."""
    target_dir = target_dir or get_default_target_dir()
    os.makedirs(target_dir, exist_ok=True)

    # Ensure an __init__.py exists in ipi/data/cyberseceval2 so importlib.resources works
    init_path = os.path.join(target_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write('"""CyberSecEval 2 package data."""\n')

    all_ok = True
    for filename, url in DATASETS.items():
        dest_path = os.path.join(target_dir, filename)
        ok = download_file(url, dest_path, force=force)
        if not ok:
            all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CyberSecEval 2 benchmark datasets.")
    parser.add_argument("--target-dir", type=str, default=None, help="Directory to save downloaded files.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files already exist.")
    args = parser.parse_args()

    success = download_cyberseceval2(target_dir=args.target_dir, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
