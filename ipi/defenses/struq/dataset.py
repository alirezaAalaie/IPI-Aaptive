"""
StruQ Dataset Generation Utilities.
"""
import random
import logging
from typing import Dict, List, Any

from .config import STRUQ_DELIMITERS, IGNORE_ATTACK_SENTENCES_TRAIN
from .defense import format_struq_prompt

log = logging.getLogger(__name__)


def generate_struq_training_data(
    clean_samples: List[Dict[str, str]],
    attack_type: str = "NaiveCompletion",
    delimiter_scheme: str = "SpclSpclSpcl",
    downsample: bool = True,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generates an anti-instruction fine-tuning dataset for StruQ.

    Converts clean instruction tuning samples (instruction, input, output) into
    structured samples where untrusted inputs contain injected prompt attacks,
    and targets train the model to respond ONLY to the instruction channel.

    Args:
        clean_samples: List of dicts with 'instruction', 'input', 'output'.
        attack_type: 'None', 'Naive', 'Ignore', 'Completion', or 'NaiveCompletion'.
        delimiter_scheme: Delimiter scheme name ('SpclSpclSpcl', 'TextTextText', etc.).
        downsample: If True, balances 50% clean and 50% injected examples.
        seed: Random seed for reproducibility.

    Returns:
        List of dataset dicts containing 'prompt' (structured query) and 'output' (target text).
    """
    random.seed(seed)
    delm = STRUQ_DELIMITERS.get(delimiter_scheme, STRUQ_DELIMITERS["SpclSpclSpcl"])
    dataset: List[Dict[str, str]] = []

    if len(clean_samples) < 2:
        log.warning("Clean dataset has fewer than 2 samples; returning standard clean formatted data.")
        for sample in clean_samples:
            prompt_str = format_struq_prompt(
                instruction=sample.get("instruction", ""),
                input_text=sample.get("input", ""),
                delimiter_scheme=delimiter_scheme,
            )
            dataset.append({"prompt": prompt_str, "output": sample.get("output", "")})
        return dataset

    for i, sample in enumerate(clean_samples):
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")

        # 1. Clean sample
        clean_prompt = format_struq_prompt(instruction=inst, input_text=inp, delimiter_scheme=delimiter_scheme)
        dataset.append({"prompt": clean_prompt, "output": out})

        if attack_type == "None" or not inp:
            continue

        # 2. Injected sample
        other_sample = random.choice([s for j, s in enumerate(clean_samples) if j != i])
        inj_inst = other_sample.get("instruction", "")
        inj_inp = other_sample.get("input", "")

        if inj_inst.endswith("?"):
            inj_prompt = f"answer the following question. {inj_inst} {inj_inp}".strip()
        else:
            inj_prompt = f"{inj_inst[0].lower()}{inj_inst[1:]} {inj_inp}".strip()

        modified_inp = inp
        if not modified_inp.endswith("."):
            modified_inp += "."

        sub_attack = attack_type
        if attack_type == "NaiveCompletion":
            sub_attack = random.choice(["Naive", "Completion"])

        if sub_attack == "Naive":
            modified_inp += f" {inj_prompt.capitalize()}"
        elif sub_attack == "Ignore":
            prefix = random.choice(IGNORE_ATTACK_SENTENCES_TRAIN).format(injected_prompt=inj_prompt)
            modified_inp += f" {prefix}"
        elif sub_attack == "Completion":
            modified_inp += (
                f"\n\n{delm[2]}\n{out}\n\n{delm[0]}\n{inj_prompt.capitalize()}"
            )
            if inj_inp:
                modified_inp += f"\n\n{delm[1]}\n{inj_inp}"

        injected_prompt_str = format_struq_prompt(
            instruction=inst,
            input_text=modified_inp,
            delimiter_scheme=delimiter_scheme,
        )
        dataset.append({"prompt": injected_prompt_str, "output": out})

    if downsample and len(dataset) > len(clean_samples):
        # Keep dataset balanced
        random.shuffle(dataset)
        dataset = dataset[: len(clean_samples) * 2]

    return dataset
