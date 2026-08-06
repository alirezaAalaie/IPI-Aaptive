"""
Data Setup Script for StruQ Defense Paper Experiments.

Downloads data dependencies used in the original StruQ paper:
  - alpaca_data_cleaned.json
  - davinci_003_outputs.json
  - alpaca_data.json

Usage:
  python scripts/setup_struq_data.py
"""
import os
import urllib.request
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("StruQSetup")

DATA_URLS = [
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json",
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json",
]


def download_struq_data(target_dir: str = "data") -> None:
    os.makedirs(target_dir, exist_ok=True)
    for url in DATA_URLS:
        filename = url.split("/")[-1]
        filepath = os.path.join(target_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            log.info(f"File '{filepath}' already exists. Skipping download.")
            continue
        log.info(f"Downloading {url} -> {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            log.info(f"Successfully downloaded '{filepath}' ({os.path.getsize(filepath)} bytes).")
        except Exception as e:
            log.error(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    download_struq_data()
