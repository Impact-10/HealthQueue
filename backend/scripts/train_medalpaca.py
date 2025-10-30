"""Fine-tune a MedAlpaca-compatible causal LM using Transformers Trainer.

Defaults are CPU-friendly (distilgpt2). To train real MedAlpaca, set:
  export BASE_MODEL="medalpaca/medalpaca-7b"
and ensure adequate GPU resources, or adapt to LoRA/PEFT.

Saves checkpoints and final model to ./medalpaca-custom-diagnosis
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT.parent / "medalpaca-custom-diagnosis"


def get_model_name() -> str:
    return os.getenv("BASE_MODEL", "distilgpt2")


def load_qna_dataset() -> Dict[str, List[Dict[str, str]]]:
    jsonl_path = DATA_DIR / "med_qna.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}. Run generate_qna_dataset.py first.")
    # Hugging Face datasets can read JSON lines directly
    ds = load_dataset("json", data_files=str(jsonl_path))
    # Split train/validation
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    return {"train": split["train"], "validation": split["test"]}


def build_prompt(example: Dict[str, str]) -> str:
    # Simple instruction format aligned with MedAlpaca-style dialogs
    question = example["question"].strip()
    answer = example["answer"].strip()
    return (
        "You are a helpful medical assistant. Provide concise, evidence-based guidance.\n"
        f"Patient: {question}\nAssistant: {answer}\n"
    )


def tokenize_function(example: Dict[str, str], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, List[int]]:
    text = build_prompt(example)
    toks = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
    )
    # Labels equal to input_ids for Causal LM
    toks["labels"] = toks["input_ids"].copy()
    return toks


def main() -> None:
    set_seed(42)
    base_model = get_model_name()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    data = load_qna_dataset()

    max_length = int(os.getenv("MAX_LENGTH", "256"))
    tokenized_train = data["train"].map(lambda ex: tokenize_function(ex, tokenizer, max_length))
    tokenized_val = data["validation"].map(lambda ex: tokenize_function(ex, tokenizer, max_length))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "2")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "2")),
        num_train_epochs=float(os.getenv("EPOCHS", "1")),
        learning_rate=float(os.getenv("LR", "5e-5")),
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "1")),
        fp16=os.getenv("FP16", "0") == "1",
        bf16=os.getenv("BF16", "0") == "1",
        report_to=["none"],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    # Save final model
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


