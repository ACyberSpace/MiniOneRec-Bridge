"""Stage-2 collaborative alignment for MiniOneRec.

Run regular ``sft.py`` first. This stage freezes the SFT model and pretrained
DIN encoder, then learns only the projector that aligns collaborative behavior
representations with the LLM input embedding space.
"""

from __future__ import annotations

import os
import random

import fire
import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from minionerec.collaborative import (
    COLLAB_TOKEN,
    CollaborativeCausalLM,
    CollaborativeDataCollator,
    CollaborativeSidSFTDataset,
    load_din_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_hf_dataset(dataset) -> HFDataset:
    rows = [dataset[index] for index in range(len(dataset))]
    return HFDataset.from_list(rows)


def train(
    base_model: str,
    din_checkpoint: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    category: str = "Office_Products",
    max_history_len: int = 50,
    cutoff_len: int = 512,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    projector_hidden_dim: int = 256,
    projector_dropout: float = 0.1,
    train_behavior_encoder: bool = False,
    sample: int = -1,
    seed: int = 42,
    bf16: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "minionerec-collm",
):
    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project
    category_names = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
        "Arts": "Arts_Crafts_and_Sewing",
    }
    category_text = category_names.get(category, category)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [COLLAB_TOKEN]})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    collab_token_id = tokenizer.convert_tokens_to_ids(COLLAB_TOKEN)
    if len(tokenizer.encode(COLLAB_TOKEN, add_special_tokens=False)) != 1:
        raise ValueError("collaborative placeholder must tokenize to exactly one token")

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16 if bf16 else None)
    base.resize_token_embeddings(len(tokenizer))
    behavior_encoder = load_din_checkpoint(din_checkpoint)
    model = CollaborativeCausalLM(
        base,
        behavior_encoder,
        collab_token_id,
        projector_hidden_dim,
        projector_dropout,
    )
    model.freeze_backbones(train_behavior_encoder=train_behavior_encoder)
    model.config.use_cache = False

    dataset_args = dict(
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category_text,
        max_history_len=max_history_len,
        padding_idx=behavior_encoder.config.padding_idx,
    )
    train_dataset = CollaborativeSidSFTDataset(train_file=train_file, **dataset_args)
    eval_dataset = CollaborativeSidSFTDataset(train_file=eval_file, **dataset_args)
    train_dataset = _to_hf_dataset(train_dataset).shuffle(seed=seed)
    eval_dataset = _to_hf_dataset(eval_dataset)

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    collator = CollaborativeDataCollator(
        transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            bf16=bf16,
            optim="adamw_torch",
            remove_unused_columns=False,
            report_to="wandb" if wandb_project else "none",
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.4f}%)")
    trainer.train()
    model.save_collaborative_adapter(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved collaborative adapter to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
