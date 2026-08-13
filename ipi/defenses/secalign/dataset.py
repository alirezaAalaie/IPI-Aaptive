"""
SecAlign Preference Dataset Construction & Delimiter Randomization.
"""
from __future__ import annotations

import os
import json
import random
import logging
import urllib.request
from copy import deepcopy
from typing import Dict, List, Optional, Any

from ..channels import format_with_other_delimiters as _format_with_other_delimiters
from .config import (
    DELIMITERS,
    OTHER_DELM_TOKENS,
    OTHER_DELM_FOR_TEST,
    PROMPT_FORMAT,
    PAPER_DATA_URLS,
)

log = logging.getLogger(__name__)


def format_with_other_delimiters(
    text: str,
    test: bool = False,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Replace real delimiters with randomised stand-ins, so the aligned model does
    not key on one literal delimiter style. Thin wrapper binding the shared
    implementation to SecAlign's delimiter tables.
    """
    return _format_with_other_delimiters(
        text,
        delimiters=DELIMITERS,
        other_delm_tokens=OTHER_DELM_TOKENS,
        other_delm_for_test=OTHER_DELM_FOR_TEST,
        test=test,
        rng=rng,
    )


def load_paper_datasets(
    data_dir: str = "./data",
    pad_token: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Loads or downloads the original SecAlign paper training datasets:
      1. `alpaca_data_cleaned.json` (51.7k clean samples)
      2. `alpaca_data.json` (reference responses for completion attack generation)

    Args:
        data_dir:  Where the two JSON files are cached.
        pad_token: Stripped from each reference instruction before it is used as
                   a lookup key, matching upstream ``align.py:38``. Pass the
                   tokenizer's pad token; the default no-ops.
    """
    os.makedirs(data_dir, exist_ok=True)
    cleaned_path = os.path.join(data_dir, "alpaca_data_cleaned.json")
    stanford_path = os.path.join(data_dir, "alpaca_data.json")

    if not os.path.exists(cleaned_path):
        log.info(f"Downloading {PAPER_DATA_URLS['alpaca_cleaned']} to {cleaned_path}...")
        urllib.request.urlretrieve(PAPER_DATA_URLS['alpaca_cleaned'], cleaned_path)

    if not os.path.exists(stanford_path):
        log.info(f"Downloading {PAPER_DATA_URLS['alpaca_stanford']} to {stanford_path}...")
        urllib.request.urlretrieve(PAPER_DATA_URLS['alpaca_stanford'], stanford_path)

    with open(cleaned_path, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    ref_inst_resp = {}
    with open(stanford_path, "r", encoding="utf-8") as f:
        stanford_data = json.load(f)
        for ref_sample in stanford_data:
            key = ref_sample['instruction']
            if pad_token:
                key = key.replace(pad_token, '')
            ref_inst_resp[key] = ref_sample['output']

    return clean_data, ref_inst_resp


def _pick_injected(
    samples: List[Dict[str, Any]],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Draw a random sample to build the injected instruction from.

    Upstream uses ``np.random.choice`` directly and would raise on a corpus
    entry with an empty instruction (it indexes ``[0]`` on it). Alpaca has none,
    but user-supplied corpora might, so we resample instead.
    """
    for _ in range(10):
        cand = samples[rng.randrange(len(samples))]
        if cand.get('instruction', ''):
            return deepcopy(cand)
    return None


def generate_secalign_preference_data(
    clean_data: Optional[List[Dict[str, Any]]] = None,
    ref_inst_resp: Optional[Dict[str, str]] = None,
    frontend_delimiters: str = "TextTextText",
    attack: str = "NaiveCompletion",
    alignment: str = "dpo",
    eos_token: str = "</s>",
    naive_proportion: float = 0.9,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generates preference dataset for SecAlign matching the original paper's `align.py` algorithm.

    ``attack`` accepts only the two values upstream implements — 'Naive' and
    'NaiveCompletion'. Anything else raises, rather than silently substituting a
    made-up attack and producing a corpus that is not SecAlign's.
    """
    rng = random.Random(seed)

    if attack not in ('Naive', 'NaiveCompletion'):
        raise NotImplementedError(
            f"Unknown SecAlign attack {attack!r}; upstream align.py implements "
            "'Naive' and 'NaiveCompletion' only."
        )

    if clean_data is None:
        clean_data, ref_inst_resp = load_paper_datasets()

    if ref_inst_resp is None:
        ref_inst_resp = {s.get("instruction", ""): s.get("output", "") for s in clean_data}

    prompt_dict = PROMPT_FORMAT.get(frontend_delimiters, PROMPT_FORMAT["TextTextText"])
    preference_data = []

    indices = list(range(len(clean_data)))
    if max_samples and max_samples < len(indices):
        indices = indices[:max_samples]

    for i in indices:
        item = clean_data[i]
        if item.get("input", "") == "":
            continue

        current_sample = deepcopy(item)
        if current_sample['input'] and current_sample['input'][-1] != '.':
            current_sample['input'] += '.'

        injected_sample = _pick_injected(clean_data, rng)
        if injected_sample is None:
            continue
        inj_inst = injected_sample['instruction']
        inj_inp = injected_sample.get('input', '')

        if inj_inst[-1] == '?':
            injected_prompt = 'answer the following question. ' + inj_inst + ' ' + inj_inp
        else:
            injected_prompt = inj_inst[0].lower() + inj_inst[1:] + ' ' + inj_inp

        if attack == 'Naive':
            current_sample['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
        elif attack == 'NaiveCompletion':
            if rng.random() < naive_proportion:
                current_sample['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
            else:
                current_sample['input'] += (
                    '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' +
                    ref_inst_resp.get(current_sample['instruction'], current_sample['output']) + '\n\n' +
                    DELIMITERS['SpclSpclSpcl'][0] + '\n' + injected_prompt.capitalize()
                )
                if injected_sample.get('input', '') != '':
                    current_sample['input'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + injected_sample['input']
                current_sample['input'] = format_with_other_delimiters(
                    current_sample['input'], test=False, rng=rng,
                )

        if alignment in ('dpo', 'orpo'):
            preference_data.append({
                'prompt': prompt_dict["prompt_input"].format_map(current_sample),
                'chosen': current_sample['output'] + eos_token,
                'rejected': injected_sample['output'] + eos_token,
            })
        elif alignment in ('kto', 'bco'):
            preference_data.append({
                'prompt': prompt_dict["prompt_input"].format_map(current_sample),
                'completion': current_sample['output'] + eos_token,
                'label': True
            })
            preference_data.append({
                'prompt': prompt_dict["prompt_input"].format_map(current_sample),
                'completion': injected_sample['output'] + eos_token,
                'label': False
            })
        else:
            raise NotImplementedError(
                f"Unknown alignment {alignment!r}; expected 'dpo', 'orpo', 'kto' or 'bco'."
            )

    return preference_data
