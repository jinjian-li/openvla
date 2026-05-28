#!/usr/bin/env python3
"""
OpenVLA-7B LoRA fine-tuning on Bridge V2 dataset.
Target: RTX 4060 8GB — uses 4-bit quantization + LoRA to fit in VRAM.

Usage:
    python openvla_lora_finetune.py --steps 1000 --batch_size 4
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="openvla/openvla-7b")
    parser.add_argument("--dataset", default="embodied-bridge/bridge-v2")
    parser.add_argument("--output_dir", default="./openvla-lora-bridgev2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--grad_checkpoint", action="store_true", default=True)
    return parser.parse_args()


def format_bridge_v2(example):
    """Format Bridge V2 example into OpenVLA text format."""
    instruction = example.get("instruction", example.get("text", ""))
    return {"text": f"In: {instruction}\nOut:"}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if device == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("[load] processor ...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print("[load] model (4-bit) ...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    print(f"[data] loading {args.dataset} ...")
    raw = load_dataset(args.dataset, split="train")
    dataset = raw.map(format_bridge_v2, remove_columns=raw.column_names)

    def tokenize(examples):
        return processor(
            text=examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        max_steps=args.steps,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
    )

    print("[train] starting ...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[done] model saved to {final_dir}")


if __name__ == "__main__":
    main()
