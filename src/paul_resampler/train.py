from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .corpus import read_corpus
from .modeling import best_dtype


class TokenBlockDataset(Dataset):
    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def make_blocks(tokenizer, docs: list[str], *, block_size: int, min_tokens: int = 64):
    blocks: list[list[int]] = []
    eos = tokenizer.eos_token_id
    for doc in docs:
        ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids = ids + [eos]
        for start in range(0, len(ids), block_size):
            block = ids[start : start + block_size]
            if len(block) >= min_tokens:
                blocks.append(block)
    if not blocks:
        raise ValueError("Corpus produced no training blocks. Add more text or reduce --block-size.")
    return blocks


def train_style_adapter(
    *,
    corpus_path: str,
    output_dir: str,
    base_model_name: str,
    block_size: int = 512,
    epochs: float = 2.0,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
    lora_r: int = 8,
    lora_alpha: int = 16,
    seed: int = 42,
):
    docs = read_corpus(corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = make_blocks(tokenizer, docs, block_size=block_size)
    dataset = TokenBlockDataset(blocks)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=best_dtype(),
    )
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    # Needed by some Transformers/PEFT combinations when gradient
    # checkpointing is used with a frozen base model.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        gradient_checkpointing=True,
        seed=seed,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    token_count = sum(len(b) for b in blocks)
    print(f"Saved adapter to {out}")
    print(f"Corpus documents: {len(docs)}")
    print(f"Training blocks: {len(blocks)}")
    print(f"Approx tokens used: {token_count}")
