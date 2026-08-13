"""
SecAlign Training Pipeline using PEFT LoRA and DPO / ORPO.
"""
from __future__ import annotations

import inspect
import logging
from typing import Dict, List, Optional, Any

from .dataset import generate_secalign_preference_data

log = logging.getLogger(__name__)


def train_secalign(
    model_name_or_path: str,
    output_dir: str,
    clean_data: Optional[List[Dict[str, Any]]] = None,
    ref_inst_resp: Optional[Dict[str, str]] = None,
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
    seed: int = 42,
) -> Any:
    """
    Train a SecAlign model via DPO/ORPO using the paper's hyperparameters.

    Args:
        model_name_or_path: HF model id or local path. SecAlign is applied on
                            top of a model already instruction-tuned on the
                            target delimiter scheme (upstream chains it after
                            the StruQ SFT stage); pointing it at a raw base
                            model trains a different thing than the paper's.
        output_dir:         Where the adapter is written.
        clean_data:         Alpaca-style corpus. None downloads the paper's.
        ref_inst_resp:      instruction -> response map for the Completion half
                            of the attack. Pass the map from
                            :func:`load_paper_datasets`; omitting it falls back
                            to each sample's own output.
        frontend_delimiters:Must match what the base model was tuned on.
        attack:             'Naive' or 'NaiveCompletion'.
        alignment:          'dpo' or 'orpo'.

    IPI adaptations vs original
    ---------------------------
    Upstream runs FSDP across 4 GPUs at effective batch 64. This runs QLoRA on
    one GPU at a much smaller effective batch, so loss curves will not line up
    with the paper's even though the LoRA rank, alpha, dropout, beta, LR and
    epoch count do.
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
        ref_inst_resp=ref_inst_resp,
        frontend_delimiters=frontend_delimiters,
        attack=attack,
        alignment=alignment,
        eos_token=tokenizer.eos_token or "</s>",
        max_samples=max_samples,
        seed=seed,
    )
    dataset = Dataset.from_list(pref_data)

    bnb_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
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
        cfg_cls, trainer_cls = DPOConfig, DPOTrainer
    elif alignment.lower() == "orpo":
        cfg_cls, trainer_cls = ORPOConfig, ORPOTrainer
    else:
        raise ValueError(f"Unsupported alignment '{alignment}'. Choose 'dpo' or 'orpo'.")

    # DPO compares log-prob differences between two forward passes; fp16 eats
    # enough precision there to matter, so prefer bf16 wherever it exists.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    cfg_params = set(inspect.signature(cfg_cls.__init__).parameters.keys())
    cfg_kwargs = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "beta": 0.1,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "seed": seed,
    }
    if "max_length" in cfg_params:
        cfg_kwargs["max_length"] = max_length
    elif "max_seq_length" in cfg_params:
        cfg_kwargs["max_seq_length"] = max_length

    if "max_prompt_length" in cfg_params:
        cfg_kwargs["max_prompt_length"] = max_prompt_length

    config_obj = cfg_cls(**cfg_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": config_obj,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }

    orig_forward = model.forward

    try:
        trainer = trainer_cls(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        try:
            trainer = trainer_cls(tokenizer=tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = trainer_cls(**trainer_kwargs)

    # TRL's newer trainers read `num_valid_tokens` / `entropy_sum` off the model
    # output, and reach them via a chunked-CE path that miscomputes under
    # device_map="auto". Populating the two fields ourselves keeps the trainer
    # happy without entering that path.
    #
    # This is assigned on the *base* model rather than trainer.model on purpose:
    # TRL wrapped `model` in a PeftModel above, and PEFT dispatches through the
    # base module, where nn.Module.__call__ picks up this instance attribute.
    def safe_forward(*args, **kwargs):
        outputs = orig_forward(*args, **kwargs)
        if outputs is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            if not hasattr(outputs, "num_valid_tokens"):
                labels = kwargs.get("labels", None)
                if labels is not None and hasattr(labels, "device"):
                    num_valid = (labels != -100).sum().to(outputs.loss.device)
                else:
                    num_valid = torch.tensor(1, device=outputs.loss.device)
                try:
                    setattr(outputs, "num_valid_tokens", num_valid)
                except Exception:
                    pass
            if not hasattr(outputs, "entropy_sum"):
                entropy_sum = torch.tensor(0.0, device=outputs.loss.device)
                try:
                    setattr(outputs, "entropy_sum", entropy_sum)
                except Exception:
                    pass
        return outputs

    model.forward = safe_forward






    log.info("Starting SecAlign training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
