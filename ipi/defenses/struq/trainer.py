"""
StruQ Fine-Tuning & Tokenizer Resizing Module.
"""
from __future__ import annotations

import inspect
import logging
from typing import Dict, List, Optional, Any

from .config import SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS
from .dataset import generate_struq_training_data

log = logging.getLogger(__name__)


def smart_tokenizer_and_embedding_resize(
    tokenizer: Any,
    model: Any,
    special_tokens: Optional[List[str]] = None,
    textual_tokens: Optional[List[str]] = None,
) -> int:
    """
    Enlarges vocabulary with new special StruQ delimiter tokens and initializes
    their token embeddings with the embeddings of corresponding textual tokens.

    Textual mapping:
      '[INST]' -> 'instruction'
      '[INPT]' -> 'input'
      '[RESP]' -> 'response'
      '[MARK]' -> '###'
      '[COLN]' -> ':'
    """
    import torch
    if special_tokens is None:
        special_tokens = SPECIAL_DELM_TOKENS
    if textual_tokens is None:
        textual_tokens = TEXTUAL_DELM_TOKENS

    assert len(special_tokens) == len(textual_tokens)
    pad_added = tokenizer.pad_token is None
    num_new_tokens = tokenizer.add_special_tokens({
        'pad_token': '[PAD]' if pad_added else tokenizer.pad_token,
        'additional_special_tokens': special_tokens,
    })

    if num_new_tokens > 0:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))

        try:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            delimiter_init_indices = [
                tokenizer.encode(v, add_special_tokens=False)[0] for v in textual_tokens
            ]

            if pad_added:
                input_embeddings[-num_new_tokens] = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings[-num_new_tokens] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            for i in range(len(special_tokens)):
                index = -num_new_tokens + i + (1 if pad_added else 0)
                if abs(index) <= len(input_embeddings):
                    input_embeddings[index] = input_embeddings[delimiter_init_indices[i]]
                    output_embeddings[index] = output_embeddings[delimiter_init_indices[i]]
            log.info(f"Initialized {len(special_tokens)} special delimiter token embeddings from textual counterparts.")
        except Exception as e:
            log.warning(f"Could not warm-start delimiter token embeddings: {e}")

    return num_new_tokens


def train_struq(
    model_name_or_path: str,
    output_dir: str,
    train_samples: List[Dict[str, str]],
    attack_type: str = "NaiveCompletion",
    delimiter_scheme: str = "SpclSpclSpcl",
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 512,
) -> Any:
    """
    Trains a StruQ defense model using Supervised Fine-Tuning (SFT) and LoRA/QLoRA.
    Optimized for Kaggle GPU environments (T4 16GB VRAM).

    Args:
        model_name_or_path: Hugging Face model identifier or local directory.
        output_dir: Directory where fine-tuned LoRA weights will be saved.
        train_samples: Raw clean instruction dataset samples.
        attack_type: Anti-instruction injection strategy ('NaiveCompletion', 'Naive', 'Ignore').
        delimiter_scheme: Delimiter configuration scheme key ('SpclSpclSpcl', etc.).
        use_4bit: If True, uses BitsAndBytes 4-bit NF4 quantization (QLoRA).
        lora_r: Rank of LoRA adapters.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability for LoRA layers.
        learning_rate: Learning rate for optimizer.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate gradients before optimizer step.
        max_length: Maximum sequence length.

    Returns:
        Trainer instance after completion.
    """
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        raise ImportError(
            "StruQ training requires `torch`, `transformers`, `datasets`, `peft`, and `trl`. "
            "Install them via `pip install trl peft transformers datasets bitsandbytes`."
        ) from e

    log.info(f"Loading tokenizer for {model_name_or_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Generating StruQ anti-instruction fine-tuning dataset...")
    raw_dataset = generate_struq_training_data(
        clean_samples=train_samples,
        attack_type=attack_type,
        delimiter_scheme=delimiter_scheme,
    )

    # Format into SFT text field
    formatted_data = [
        {"text": f"{d['prompt']}{d['output']}{tokenizer.eos_token or '</s>'}"}
        for d in raw_dataset
    ]
    dataset = Dataset.from_list(formatted_data)

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

    # Add special delimiter tokens and warm-start embeddings
    if delimiter_scheme == "SpclSpclSpcl":
        smart_tokenizer_and_embedding_resize(tokenizer=tokenizer, model=model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Build SFTConfig dynamically according to available parameters in installed trl version
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    sft_kwargs = {
        "output_dir": output_dir,
        "dataset_text_field": "text",
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": torch.cuda.is_available(),
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
    }
    if "max_seq_length" in sft_params:
        sft_kwargs["max_seq_length"] = max_length
    elif "max_length" in sft_params:
        sft_kwargs["max_length"] = max_length

    sft_config = SFTConfig(**sft_kwargs)

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "max_seq_length" in trainer_params and "max_seq_length" not in sft_kwargs and "max_length" not in sft_kwargs:
        trainer_kwargs["max_seq_length"] = max_length

    trainer = SFTTrainer(**trainer_kwargs)


    log.info("Starting StruQ anti-instruction fine-tuning...")
    trainer.train()
    log.info(f"Training completed. Saving fine-tuned adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
