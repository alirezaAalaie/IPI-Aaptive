"""
SecAlign Defense & Fine-Tuning Module for Indirect Prompt Injection.

Based on the paper:
"SecAlign: Defending Against Prompt Injection with Preference Optimization"
(Chen et al., 2024 / CCS 2025)

This module provides:
  1. `SecAlignDefense`: A `DefendedVictim` wrapper for evaluating SecAlign models in `ipi`.
  2. `generate_secalign_preference_data`: Utility to construct (prompt, chosen, rejected) pairs.
  3. `train_secalign`: High-level trainer supporting QLoRA/LoRA + ORPO/DPO/KTO optimized for single-GPU / Kaggle T4.
"""
from __future__ import annotations

import os
import json
import random
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any, Union

from .base import DefendedVictim
from ..victim import Victim

log = logging.getLogger(__name__)


# Standard SecAlign Delimiters & Formatting
SECALIGN_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### instruction:\n{instruction}\n\n"
    "### input:\n{input}\n\n"
    "### response:\n"
)

SECALIGN_NO_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### instruction:\n{instruction}\n\n"
    "### response:\n"
)


class SecAlignDefense(DefendedVictim):
    """
    SecAlign Defense Wrapper for `ipi` Benchmark.

    SecAlign models are fine-tuned via preference optimization (ORPO/DPO) to prefer
    fulfilling the legitimate user task and ignoring any injected instructions inside
    the untrusted data input.

    Args:
        target: Underlying Victim instance (e.g. LocalLLM loaded with SecAlign fine-tuned weights/adapter).
        use_secalign_template: Whether to reformat inputs using SecAlign's structured prompt template.
    """

    def __init__(
        self,
        target: Victim,
        use_secalign_template: bool = False,
    ):
        super().__init__(target)
        self.use_secalign_template = use_secalign_template

    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.use_secalign_template:
            return messages

        # Format input using SecAlign structural delimiter template if requested
        new_msgs = deepcopy(messages)
        user_content = ""
        system_content = ""

        for msg in new_msgs:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")

        inst = system_content if system_content else "Follow the instructions in the prompt."
        inp = user_content

        formatted_prompt = SECALIGN_PROMPT_TEMPLATE.format(instruction=inst, input=inp)
        return [{"role": "user", "content": formatted_prompt}]


def generate_secalign_preference_data(
    clean_samples: List[Dict[str, str]],
    attack_type: str = "Naive",
    alignment_type: str = "orpo",
    eos_token: str = "</s>",
    naive_proportion: float = 0.9,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates a preference dataset for SecAlign training.

    For each clean instruction sample (instruction, input, output):
      - We inject an adversarial prompt P_inj into the input.
      - Chosen response = clean output O_clean (follows main task).
      - Rejected response = output corresponding to the injected prompt O_inj.

    Args:
        clean_samples: List of dicts containing 'instruction', 'input', 'output'.
        attack_type: Injection construction strategy ('Naive' or 'NaiveCompletion').
        alignment_type: Preference optimization target ('orpo', 'dpo', 'kto', 'bco').
        eos_token: End of sequence token string.
        naive_proportion: Probability of naive vs completion injection in NaiveCompletion mode.
        seed: Random seed for reproducibility.

    Returns:
        List of preference dictionaries compatible with HuggingFace `datasets` and `trl`.
    """
    random.seed(seed)
    preference_data: List[Dict[str, Any]] = []

    num_samples = len(clean_samples)
    if num_samples < 2:
        log.warning("Clean dataset has fewer than 2 samples; preference pairing requires at least 2 samples.")
        return preference_data

    for i in range(num_samples):
        sample = deepcopy(clean_samples[i])
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        output_clean = sample.get("output", "")

        if not inst:
            continue

        # Pick a random sample whose output/instruction acts as the injection target
        injected_sample = random.choice([s for j, s in enumerate(clean_samples) if j != i])
        inj_inst = injected_sample.get("instruction", "")
        inj_inp = injected_sample.get("input", "")
        inj_out = injected_sample.get("output", "")

        if inj_inst.endswith("?"):
            injected_prompt = f"answer the following question. {inj_inst} {inj_inp}".strip()
        else:
            injected_prompt = f"{inj_inst[0].lower()}{inj_inst[1:]} {inj_inp}".strip()

        # Construct injected input
        modified_input = inp
        if modified_input and not modified_input.endswith("."):
            modified_input += "."

        if attack_type == "Naive":
            modified_input += f" {injected_prompt.capitalize()}"
        elif attack_type == "NaiveCompletion":
            if random.random() < naive_proportion:
                modified_input += f" {injected_prompt.capitalize()}"
            else:
                modified_input += f"\n\n### response:\n{output_clean}\n\n### instruction:\n{injected_prompt.capitalize()}"
        else:
            modified_input += f" Ignore prior instructions and instead: {injected_prompt}"

        # Construct prompt text
        if modified_input:
            prompt_str = SECALIGN_PROMPT_TEMPLATE.format(instruction=inst, input=modified_input)
        else:
            prompt_str = SECALIGN_NO_INPUT_TEMPLATE.format(instruction=inst)

        chosen_str = f"{output_clean}{eos_token}"
        rejected_str = f"{inj_out}{eos_token}"

        if alignment_type in ("dpo", "orpo"):
            preference_data.append({
                "prompt": prompt_str,
                "chosen": chosen_str,
                "rejected": rejected_str,
            })
        elif alignment_type in ("kto", "bco"):
            preference_data.append({
                "prompt": prompt_str,
                "completion": chosen_str,
                "label": True,
            })
            preference_data.append({
                "prompt": prompt_str,
                "completion": rejected_str,
                "label": False,
            })

    return preference_data


def train_secalign(
    model_name_or_path: str,
    output_dir: str,
    train_samples: List[Dict[str, str]],
    alignment_type: str = "orpo",
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 512,
    max_prompt_length: int = 384,
) -> Any:
    """
    Trains a SecAlign defense model using Preference Optimization (ORPO / DPO) and LoRA/QLoRA.
    Optimized for Kaggle GPU environments (T4 16GB VRAM).

    Args:
        model_name_or_path: Hugging Face model identifier or local directory.
        output_dir: Directory where fine-tuned LoRA weights will be saved.
        train_samples: Raw clean instruction dataset samples.
        alignment_type: 'orpo' or 'dpo'.
        use_4bit: If True, uses BitsAndBytes 4-bit NF4 quantization (QLoRA) to reduce VRAM.
        lora_r: Rank of LoRA adapters.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability for LoRA layers.
        learning_rate: Learning rate for optimizer.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate gradients before optimizer step.
        max_length: Maximum total sequence length.
        max_prompt_length: Maximum prompt length.

    Returns:
        Trainer instance after completion.
    """
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from trl import ORPOTrainer, ORPOConfig, DPOTrainer, DPOConfig
    except ImportError as e:
        raise ImportError(
            "SecAlign training requires `torch`, `transformers`, `datasets`, `peft`, and `trl`. "
            "Install them via `pip install trl peft transformers datasets bitsandbytes`."
        ) from e

    log.info(f"Loading tokenizer for {model_name_or_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Generating SecAlign preference dataset...")
    pref_data = generate_secalign_preference_data(
        clean_samples=train_samples,
        attack_type="Naive",
        alignment_type=alignment_type,
        eos_token=tokenizer.eos_token or "</s>",
    )
    dataset = Dataset.from_list(pref_data)

    # Quantization Config for T4 VRAM efficiency
    bnb_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            log.warning("BitsAndBytes not available; proceeding without 4-bit quantization.")

    log.info(f"Loading base model {model_name_or_path}...")
    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    # Configure PEFT / LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if alignment_type.lower() == "orpo":
        orpo_config = ORPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=0.1,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        )
        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    elif alignment_type.lower() == "dpo":
        dpo_config = DPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=0.1,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        )
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"Unsupported alignment_type '{alignment_type}'. Choose 'orpo' or 'dpo'.")

    log.info("Starting SecAlign preference optimization fine-tuning...")
    trainer.train()
    log.info(f"Training completed. Saving fine-tuned adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
