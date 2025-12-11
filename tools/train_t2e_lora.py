#!/usr/bin/env python3
"""
Minimal Text-to-Emotion (T2E) distillation trainer (Aligns with IndexTTS2 Section 3.3).

Input dataset format (JSONL or JSON list):
    {"text": "...", "probs": [p1, p2, ..., p7]}   # probs sum to 1 for 7 emotions

Usage example:
    python tools/train_t2e_lora.py \
        --model qwen/Qwen2-1.5B-Instruct \
        --data /mnt/sda1/datasets/t2e_1000.jsonl \
        --output /mnt/sda1/models/index-tts-ko/t2e_lora \
        --lr 2e-4 --batch 8 --epochs 3

Notes:
  - Uses PEFT LoRA on the base causal LM.
  - Small, self-contained; relies only on transformers + peft + datasets.
  - Keeps tokenizer padding side = right for CLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import IterableDataset

EMO_LABELS = ["anger", "happiness", "fear", "disgust", "sadness", "surprise", "neutral"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="qwen/Qwen2-1.5B-Instruct", help="Base causal LM.")
    p.add_argument("--data", type=str, required=True, help="JSONL/JSON with fields: text, probs[7].")
    p.add_argument("--output", type=str, required=True, help="Output directory for LoRA adapter.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def format_example(text: str, probs: list[float]) -> str:
    kv = ", ".join(f'"{lbl}": {float(p):.4f}' for lbl, p in zip(EMO_LABELS, probs))
    return (
        "Given the input sentence, return a JSON object with probabilities for each of the 7 emotions "
        "(sum to 1). Round to 4 decimals.\n"
        f"Input: {text}\nOutput: {{{kv}}}"
    )


def main() -> None:
    args = parse_args()
    ds = load_dataset("json", data_files=args.data, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess(batch):
        texts = [format_example(t, p) for t, p in zip(batch["text"], batch["probs"])]
        tok = tokenizer(
            texts,
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tok["input_ids"].clone()
        return {**tok, "labels": labels}

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 else None,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=50,
        save_steps=500,
        fp16=args.fp16,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"✅ LoRA adapter saved to: {args.output}")


if __name__ == "__main__":
    main()
