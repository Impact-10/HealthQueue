#!/usr/bin/env python3
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
)

# Optional PEFT/LoRA
HAVE_PEFT = True
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    HAVE_PEFT = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT.parent / "biogpt-diabetology-custom"  # as you requested

def get_model_name() -> str:
    # Default to smaller BioGPT to avoid huge downloads on Windows; override via BASE_MODEL
    return os.getenv("BASE_MODEL", "distilgpt2")

def load_qna_dataset():
    jsonl_path = DATA_DIR / "diabetology_qna.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}. Run generate_diabetology_dataset.py first.")
    ds = load_dataset("json", data_files=str(jsonl_path))
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    return {"train": split["train"], "validation": split["test"]}

SEP = "\n"

def build_prompt(example: Dict[str, str]) -> str:
    q = (example["question"] or "").strip()
    a = (example["answer"] or "").strip()
    # Instruction header encourages causes + self-care + follow-up + red flags in 3–5 sentences
    return (
        "You are a biomedical AI assistant specialized in diabetology. "
        "Respond in 3–5 sentences with likely factors, practical self-care, follow-up, and red flags. "
        "Avoid URLs, citations, or legal/news boilerplate."
        f"{SEP}Patient: {q}{SEP}Assistant: {a}{SEP}"
    )

def tokenize_function(example: Dict[str, str], tokenizer: AutoTokenizer, max_length: int):
    text = build_prompt(example)
    toks = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    toks["labels"] = toks["input_ids"].copy()
    return toks

def maybe_lora(model, use_peft: bool, r=16, alpha=32, dropout=0.05):
    if not use_peft:
        return model
    # For distilgpt2, use appropriate target modules
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],  # distilgpt2 specific
    )
    return get_peft_model(model, lcfg)

def main() -> None:
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_model = get_model_name()
    use_peft = os.getenv("USE_LORA", "1") == "1" and HAVE_PEFT

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device auto-map works on single-GPU or CPU
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )
    if use_peft:
        model = maybe_lora(
            model,
            True,
            r=int(os.getenv("LORA_R","16")),
            alpha=int(os.getenv("LORA_ALPHA","32")),
            dropout=float(os.getenv("LORA_DROPOUT","0.05")),
        )
        print("Using LoRA adapters.")
    else:
        print("Training without LoRA (full SFT). Set USE_LORA=1 and install peft for adapter training.")

    data = load_qna_dataset()
    max_length = int(os.getenv("MAX_LENGTH", "384"))
    tokenized_train = data["train"].map(lambda ex: tokenize_function(ex, tokenizer, max_length))
    tokenized_val = data["validation"].map(lambda ex: tokenize_function(ex, tokenizer, max_length))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Reasonable defaults; adjust by env vars
    training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    overwrite_output_dir=True,
    per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "2")),
    per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "2")),
    gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "8" if use_peft else "2")),
    learning_rate=float(os.getenv("LR", "1.5e-4" if use_peft else "5e-5")),
    num_train_epochs=float(os.getenv("EPOCHS", "3")),
    warmup_ratio=float(os.getenv("WARMUP_RATIO","0.03")),
    lr_scheduler_type=os.getenv("SCHED","cosine"),
    logging_steps=int(os.getenv("LOG_STEPS","100")),
    eval_steps=int(os.getenv("EVAL_STEPS","250")),
    save_steps=int(os.getenv("SAVE_STEPS","1000")),
    save_total_limit=int(os.getenv("SAVE_LIMIT","3")),
    fp16=os.getenv("FP16","1") == "1",
    bf16=os.getenv("BF16","0") == "1",
    gradient_checkpointing=False,  # Disable for now to avoid issues
    report_to=["none"],
    push_to_hub=False,
    do_eval=True,
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    start = time.time()
    trainer.train()
    hrs = (time.time() - start) / 3600.0
    print(f"Training finished in ~{hrs:.2f} hours")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
