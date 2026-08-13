"""
StruQ Dataset Generation Utilities.

Port of ``generate_training_data`` / ``SupervisedDataset`` from
``code/defense/StruQ-main/struq.py``.

IPI adaptations vs original
---------------------------
* Returns ``[{"prompt", "output"}]`` dicts instead of ``(sources, targets)``
  tuples, so the result can go straight into a ``datasets.Dataset``.
* ``np.random`` is replaced by an explicitly seeded ``random.Random`` — upstream
  relies on global numpy state, which makes runs irreproducible here.
* Upstream crashes on a corpus entry with an empty ``instruction`` (it indexes
  ``[0]`` on it). We resample instead; Alpaca has no such entries, but
  user-supplied corpora might.
"""
from __future__ import annotations

import logging
import random
import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..channels import format_with_other_delimiters
from .config import (
    IGNORE_ATTACK_SENTENCES_TRAIN,
    OTHER_DELM_FOR_TEST,
    OTHER_DELM_TOKENS,
    PROMPT_FORMAT,
    STRUQ_DELIMITERS,
)

log = logging.getLogger(__name__)

_SUPPORTED_ATTACKS = ("Naive", "Ignore", "Completion")


def _format(sample: Mapping[str, str], prompt_dict: Mapping[str, str]) -> str:
    """Render one sample through prompt_input / prompt_no_input."""
    key = "prompt_input" if sample.get("input", "") != "" else "prompt_no_input"
    return prompt_dict[key].format_map(
        {"instruction": sample.get("instruction", ""), "input": sample.get("input", "")}
    )


def _pick_injected(samples: Sequence[Mapping[str, str]], rng: random.Random) -> Optional[Dict[str, str]]:
    """Draw a random sample to build an injected instruction from."""
    for _ in range(10):
        cand = samples[rng.randrange(len(samples))]
        if cand.get("instruction", ""):
            return deepcopy(dict(cand))
    return None


def _injected_prompt(injected_sample: Mapping[str, str]) -> str:
    """Turn a clean sample into an imperative injected instruction."""
    inst = injected_sample.get("instruction", "")
    inp = injected_sample.get("input", "")
    if inst[-1] == "?":
        return "answer the following question. " + inst + " " + inp
    return inst[0].lower() + inst[1:] + " " + inp


def _build_sources(
    samples: Sequence[Mapping[str, str]],
    prompt_dict: Mapping[str, str],
    attack: str,
    rng: random.Random,
    ref_inst_resp: Mapping[str, str],
) -> List[str]:
    """
    Build one attacked copy of the corpus. The returned list is index-aligned
    with ``samples`` — samples with no input get the clean no-input rendering,
    exactly as upstream, which is what keeps the downsample block arithmetic
    valid.
    """
    if attack == "None":
        return [_format(s, prompt_dict) for s in samples]
    if attack not in _SUPPORTED_ATTACKS:
        raise NotImplementedError(
            f"Unknown StruQ attack {attack!r}; expected one of {_SUPPORTED_ATTACKS} "
            "(or a concatenation such as 'NaiveCompletion')."
        )

    sources: List[str] = []
    for sample in samples:
        # No anti-instruction tuning when there is no data channel to inject into.
        if sample.get("input", "") == "":
            sources.append(_format(sample, prompt_dict))
            continue

        injected_sample = _pick_injected(samples, rng)
        if injected_sample is None:
            sources.append(_format(sample, prompt_dict))
            continue
        injected_prompt = _injected_prompt(injected_sample)

        item = deepcopy(dict(sample))
        if item["input"][-1] != ".":
            item["input"] += "."

        if attack == "Naive":
            item["input"] += " " + injected_prompt[0].upper() + injected_prompt[1:]
        elif attack == "Ignore":
            item["input"] += " " + rng.choice(IGNORE_ATTACK_SENTENCES_TRAIN).format(
                injected_prompt=injected_prompt
            )
        elif attack == "Completion":
            # Upstream hardcodes the SpclSpclSpcl delimiters here regardless of
            # the scheme being trained, then randomises them away below. The net
            # effect is that the model never sees its *real* delimiters inside
            # the data channel during training — at inference those are stripped
            # by recursive_filter instead.
            delm = STRUQ_DELIMITERS["SpclSpclSpcl"]
            item["input"] += (
                "\n\n" + delm[2] + "\n"
                + ref_inst_resp.get(item["instruction"], item.get("output", ""))
                + "\n\n" + delm[0] + "\n" + injected_prompt.capitalize()
            )
            if injected_sample.get("input", "") != "":
                item["input"] += "\n\n" + delm[1] + "\n" + injected_sample["input"]
            item["input"] = format_with_other_delimiters(
                item["input"],
                delimiters=STRUQ_DELIMITERS,
                other_delm_tokens=OTHER_DELM_TOKENS,
                other_delm_for_test=OTHER_DELM_FOR_TEST,
                test=False,
                rng=rng,
            )

        sources.append(_format(item, prompt_dict))
    return sources


def generate_struq_training_data(
    clean_samples: List[Dict[str, str]],
    attack_type: str = "NaiveCompletion",
    delimiter_scheme: str = "SpclSpclSpcl",
    downsample: bool = True,
    seed: int = 42,
    ref_inst_resp: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, str]]:
    """
    Generate the anti-instruction fine-tuning corpus for StruQ.

    Args:
        clean_samples:    Alpaca-style dicts with 'instruction', 'input', 'output'.
        attack_type:      'None', or a concatenation of 'Naive' / 'Ignore' /
                          'Completion' (e.g. the paper's 'NaiveCompletion').
        delimiter_scheme: Key into ``STRUQ_DELIMITERS``.
        downsample:       Paper behaviour — emit exactly ``len(clean_samples)``
                          examples, half of them clean. With False, emit one
                          full attacked copy per attack and no extra clean data
                          (upstream then scales the LR by the copy count).
        seed:             Seed for the local RNG.
        ref_inst_resp:    instruction -> response map used for the fake response
                          in the Completion attack. Upstream reads this from
                          Stanford ``alpaca_data.json``; falls back to each
                          sample's own output when a key is missing.

    Returns:
        List of ``{"prompt", "output"}`` dicts.
    """
    rng = random.Random(seed)
    prompt_dict = PROMPT_FORMAT.get(delimiter_scheme) or PROMPT_FORMAT["SpclSpclSpcl"]
    ref_inst_resp = ref_inst_resp or {}
    n = len(clean_samples)
    if n == 0:
        return []

    targets = [s.get("output", "") for s in clean_samples]
    clean_sources = _build_sources(clean_samples, prompt_dict, "None", rng, ref_inst_resp)

    if attack_type == "None":
        return [{"prompt": p, "output": o} for p, o in zip(clean_sources, targets)]

    # 'NaiveCompletion' -> ['Naive', 'Completion']
    attacks = re.findall("[A-Z][^A-Z]*", attack_type)
    if not attacks:
        raise NotImplementedError(f"Could not parse StruQ attack_type {attack_type!r}.")

    # Blocks are laid out [attack_1, clean, attack_2, clean, ...], each of
    # length n, so block b index i is flat index b * n + i.
    sources: List[str] = []
    all_targets: List[str] = []
    for a in attacks:
        sources += _build_sources(clean_samples, prompt_dict, a, rng, ref_inst_resp)
        all_targets += targets
        if downsample:
            sources += clean_sources
            all_targets += targets

    if downsample:
        n_blocks = len(attacks) * 2
        picked = [rng.randrange(n_blocks) * n + i for i in range(n)]
        sources = [sources[j] for j in picked]
        all_targets = [all_targets[j] for j in picked]

    return [{"prompt": p, "output": o} for p, o in zip(sources, all_targets)]
