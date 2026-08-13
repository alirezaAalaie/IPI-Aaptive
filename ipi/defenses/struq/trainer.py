"""
StruQ Fine-Tuning & Tokenizer Resizing Module.

Port of ``code/defense/StruQ-main/train.py``.

IPI adaptations vs original
---------------------------
* Upstream does full-parameter FSDP SFT across 4 GPUs (3 epochs, lr 2e-5,
  effective batch 128). This runs QLoRA on a single Kaggle GPU. Absolute
  numbers are therefore not comparable to the paper's; the delimiter scheme,
  data construction and loss masking are.
* ``modules_to_save`` keeps ``embed_tokens`` / ``lm_head`` trainable. Without
  it the five delimiter tokens added below would stay frozen at their
  warm-start values and the structured-query scheme would get no training
  signal at all — LoRA on the attention/MLP projections cannot reach them.
* Ordering matters and is load-bearing: the tokenizer is resized *before* the
  corpus is tokenized. Tokenizing first splits ``[MARK] [INST][COLN]`` into
  ordinary subwords, while inference (which loads the saved, resized tokenizer)
  sees five atomic special tokens — training and eval would disagree.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .config import (
    DEFAULT_TOKENS,
    IGNORE_INDEX,
    SPECIAL_DELM_TOKENS,
    SPECIAL_TOKEN_SCHEMES,
    TEXTUAL_DELM_TOKENS,
)
from .dataset import generate_struq_training_data

log = logging.getLogger(__name__)


def smart_tokenizer_and_embedding_resize(
    tokenizer: Any,
    model: Any,
    special_tokens: Optional[List[str]] = None,
    textual_tokens: Optional[List[str]] = None,
) -> int:
    """
    Enlarge the vocabulary with the StruQ delimiter tokens and warm-start their
    embeddings from the textual tokens they stand in for:

      ``[INST]`` <- ``instruction``   ``[INPT]`` <- ``input``
      ``[RESP]`` <- ``response``      ``[MARK]`` <- ``###``
      ``[COLN]`` <- ``:``

    A ``[PAD]`` token is added when the tokenizer has none. Aliasing pad to eos
    instead would be wrong here: the collator derives ``attention_mask`` from
    ``input_ids.ne(pad_token_id)``, which would mask out every genuine trailing
    eos in the corpus.

    Returns the number of tokens added.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_DELM_TOKENS
    if textual_tokens is None:
        textual_tokens = TEXTUAL_DELM_TOKENS
    assert len(special_tokens) == len(textual_tokens)

    # Resolve the warm-start ids BEFORE adding the new tokens, so the lookup
    # cannot accidentally hit a token we are about to define.
    delimiter_init_indices = [
        tokenizer.encode(v, add_special_tokens=False)[0] for v in textual_tokens
    ]

    pad_added = tokenizer.pad_token is None
    to_add: Dict[str, Any] = {'additional_special_tokens': special_tokens}
    if pad_added:
        to_add['pad_token'] = DEFAULT_TOKENS['pad_token']
    num_new_tokens = tokenizer.add_special_tokens(to_add)
    if num_new_tokens == 0:
        return 0

    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # New rows sit at the end of the matrix: [PAD] (if added) then the five
    # delimiters, in the order they were passed to add_special_tokens.
    if pad_added:
        input_embeddings[-num_new_tokens] = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    offset = 1 if pad_added else 0
    for i in range(len(special_tokens)):
        index = -num_new_tokens + i + offset
        log.info(
            "Initialising %s from the embedding of %s",
            tokenizer.decode(len(tokenizer) + index),
            tokenizer.decode(delimiter_init_indices[i]),
        )
        input_embeddings[index] = input_embeddings[delimiter_init_indices[i]]
        output_embeddings[index] = output_embeddings[delimiter_init_indices[i]]

    return num_new_tokens


class DataCollatorForStruQDataset(object):
    """
    Pad ``input_ids`` with the pad token and ``labels`` with ``IGNORE_INDEX``.
    Mirrors ``DataCollatorForSupervisedDataset`` in upstream ``train.py``.
    """

    def __init__(self, tokenizer: Any, ignore_index: int = IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in instances]
        labels = [torch.tensor(inst["labels"], dtype=torch.long) for inst in instances]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        return dict(
            input_ids=input_ids_padded,
            labels=labels_padded,
            attention_mask=input_ids_padded.ne(self.tokenizer.pad_token_id),
        )


def build_struq_tokenized_dataset(
    raw_dataset: Sequence[Mapping[str, str]],
    tokenizer: Any,
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    """
    Tokenize ``{"prompt", "output"}`` pairs, masking the prompt out of the loss.

    The tokenizer must already carry the delimiter special tokens.
    """
    sources = [d["prompt"] for d in raw_dataset]
    targets = [f"{d['output']}{tokenizer.eos_token or ''}" for d in raw_dataset]
    examples = [s + t for s, t in zip(sources, targets)]

    examples_tokenized = tokenizer(examples, max_length=max_length, truncation=True, padding=False)
    sources_tokenized = tokenizer(sources, max_length=max_length, truncation=True, padding=False)

    out: List[Dict[str, Any]] = []
    dropped = 0
    for ex_ids, src_ids in zip(examples_tokenized["input_ids"], sources_tokenized["input_ids"]):
        labels = list(ex_ids)
        for i in range(min(len(src_ids), len(labels))):
            labels[i] = IGNORE_INDEX
        # A prompt longer than max_length leaves nothing supervised. Cross-entropy
        # over an all-IGNORE_INDEX row is 0/0 = nan, and a single nan poisons the
        # whole gradient-accumulation step, so drop these rather than train on them.
        if all(l == IGNORE_INDEX for l in labels):
            dropped += 1
            continue
        out.append({"input_ids": ex_ids, "labels": labels})

    if dropped:
        log.warning(
            "Dropped %d/%d examples whose prompt alone exceeded max_length=%d "
            "(no supervised tokens left). Raise max_length to keep them.",
            dropped, dropped + len(out), max_length,
        )
    return out


def train_struq(
    model_name_or_path: str,
    output_dir: str,
    train_samples: List[Dict[str, str]],
    attack_type: str = "NaiveCompletion",
    delimiter_scheme: str = "SpclSpclSpcl",
    ref_inst_resp: Optional[Mapping[str, str]] = None,
    downsample: bool = True,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 512,
    seed: int = 42,
) -> Any:
    """
    Fine-tune a StruQ defense model with ``transformers.Trainer``.

    Args:
        model_name_or_path: HF model id or local directory.
        output_dir:         Where the adapter and resized tokenizer are written.
        train_samples:      Clean Alpaca-style corpus.
        attack_type:        See :func:`generate_struq_training_data`.
        delimiter_scheme:   Key into ``STRUQ_DELIMITERS``.
        ref_inst_resp:      instruction -> response map for the Completion attack.
        downsample:         Paper's 50%-clean downsampling.
        use_4bit:           QLoRA via BitsAndBytes NF4.
        lora_r/alpha/dropout, learning_rate, num_train_epochs,
        per_device_train_batch_size, gradient_accumulation_steps, max_length:
                            Training hyperparameters.
        seed:               Dataset construction seed.

    Returns:
        The ``Trainer`` after ``train()``.
    """
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "StruQ training requires `torch`, `transformers`, `datasets`, `peft`. "
            "Install them via `pip install peft transformers datasets bitsandbytes`."
        ) from e

    uses_special_tokens = delimiter_scheme in SPECIAL_TOKEN_SCHEMES

    log.info("Loading tokenizer for %s...", model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True,
    )

    # --- Model first: the embedding resize needs it, and it has to happen
    #     before the corpus is tokenized. -----------------------------------
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
            log.warning("BitsAndBytes not available; proceeding without 4-bit quantization.")

    log.info("Loading base model %s...", model_name_or_path)
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if uses_special_tokens:
        n_added = smart_tokenizer_and_embedding_resize(tokenizer=tokenizer, model=model)
        log.info("Added %d special tokens for scheme %s.", n_added, delimiter_scheme)
    elif tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_TOKENS["pad_token"]})
        model.resize_token_embeddings(len(tokenizer))

    # --- Data: built and tokenized with the resized tokenizer. -------------
    log.info("Generating StruQ anti-instruction fine-tuning dataset...")
    raw_dataset = generate_struq_training_data(
        clean_samples=train_samples,
        attack_type=attack_type,
        delimiter_scheme=delimiter_scheme,
        downsample=downsample,
        seed=seed,
        ref_inst_resp=ref_inst_resp,
    )
    dataset = Dataset.from_list(
        build_struq_tokenized_dataset(raw_dataset, tokenizer, max_length=max_length)
    )
    data_collator = DataCollatorForStruQDataset(tokenizer=tokenizer)

    if use_4bit and bnb_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # embed_tokens / lm_head must stay trainable — see module docstring.
    modules_to_save = ["embed_tokens", "lm_head"] if uses_special_tokens else None
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,   # paper run.py
        weight_decay=0.0,    # paper run.py
        seed=seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": data_collator,
    }
    # transformers >= 4.46 renamed Trainer's `tokenizer` argument to
    # `processing_class`; support both.
    try:
        trainer = transformers.Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        try:
            trainer = transformers.Trainer(tokenizer=tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = transformers.Trainer(**trainer_kwargs)

    log.info("Starting StruQ anti-instruction fine-tuning...")
    trainer.train()
    log.info("Training completed. Saving adapter and resized tokenizer to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
