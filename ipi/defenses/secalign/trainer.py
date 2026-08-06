"""
SecAlign Training Pipeline using PEFT LoRA and DPO / ORPO.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from .dataset import generate_secalign_preference_data

log = logging.getLogger(__name__)


def train_secalign(
    model_name_or_path: str,
    output_dir: str,
    clean_data: Optional[List[Dict[str, Any]]] = None,
    frontend_delimiters: str = "TextTextText",
    attack: str = "NaiveCompletion",
    alignment: str = "dpo",
    use_4bit: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
    learning_rate: Optional[float] = None,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 512,
    max_prompt_length: int = 384,
    max_samples: Optional[int] = None,
) -> Any:
    """
    Trains SecAlign model via DPO/ORPO using original paper hyperparameters.
    """
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from trl import DPOTrainer, DPOConfig, ORPOTrainer, ORPOConfig
    except ImportError as e:
        raise ImportError(
            "SecAlign training requires `torch`, `transformers`, `datasets`, `peft`, and `trl`. "
            "Install them via `pip install trl peft transformers datasets bitsandbytes`."
        ) from e

    # Determine paper default learning rate if not specified
    if learning_rate is None:
        if 'Llama-3' in model_name_or_path:
            learning_rate = 1.6e-4
        elif 'Mistral' in model_name_or_path:
            learning_rate = 1.4e-4
        else:
            learning_rate = 2.0e-4

    log.info(f"Loading tokenizer for {model_name_or_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Generating preference data (attack={attack}, alignment={alignment})...")
    pref_data = generate_secalign_preference_data(
        clean_data=clean_data,
        frontend_delimiters=frontend_delimiters,
        attack=attack,
        alignment=alignment,
        eos_token=tokenizer.eos_token or "</s>",
        max_samples=max_samples,
    )
    dataset = Dataset.from_list(pref_data)

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
            log.warning("BitsAndBytes not available; training in full precision.")

    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    # Exact PEFT / LoRA parameters from paper align.py (r=64, lora_alpha=8, dropout=0.1)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )

    if alignment.lower() == "dpo":
        dpo_config = DPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=0.1,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
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
    elif alignment.lower() == "orpo":
        orpo_config = ORPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=0.1,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
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
    else:
        raise ValueError(f"Unsupported alignment '{alignment}'. Choose 'dpo' or 'orpo'.")

    log.info("Starting SecAlign training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
