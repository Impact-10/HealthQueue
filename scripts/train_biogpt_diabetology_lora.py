#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, random
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
)

HAVE_PEFT = True
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    HAVE_PEFT = False

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "scripts") else Path.cwd()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "biogpt-diabetology-json-lora"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_BEGIN, JSON_END = "<json>", "</json>"

SCHEMA_EXAMPLES = [
    {
        "q": "I have high sugar and itchy skin",
        "a_json": {
            "summary": "Itchy/dry skin with elevated glucose suggests xerosis and suboptimal control; moisturize and review glycemic regimen.",
            "possible_diagnoses": ["Diabetes mellitus with xerosis"],
            "rationale": ["Hyperglycemia and skin barrier dryness can cause pruritus"],
            "recommended_tests": ["Fasting glucose", "HbA1c", "2-hr post-meal glucose"],
            "next_steps": ["Fragrance-free moisturizer twice daily", "Hydration", "Review A1c and meds timing"],
            "warning_signs": ["Widespread rash with fever", "Weeping lesions or infection"]
        }
    },
    {
        "q": "My foot wound is not healing for 2 weeks",
        "a_json": {
            "summary": "Slow-healing wound with high sugars raises infection/perfusion concerns; prioritize wound care and glycemic optimization.",
            "possible_diagnoses": ["Diabetic foot ulcer (possible infection)"],
            "rationale": ["Hyperglycemia impairs leukocyte function and collagen synthesis"],
            "recommended_tests": ["Wound exam", "Glucose/A1c", "CBC if infection suspected"],
            "next_steps": ["Daily wound care and offloading", "Optimize glycemic regimen", "Early clinician review"],
            "warning_signs": ["Spreading redness", "Foul discharge", "Fever"]
        }
    },
    {
        "q": "I urinate frequently and lost weight recently",
        "a_json": {
            "summary": "Polyuria and weight loss suggest significant hyperglycemia; check fasting and post-meal levels and review regimen promptly.",
            "possible_diagnoses": ["Diabetes mellitus with poor glycemic control"],
            "rationale": ["Osmotic diuresis and catabolism from uncontrolled glucose"],
            "recommended_tests": ["Fasting glucose", "HbA1c", "2-hr post-meal glucose"],
            "next_steps": ["Hydration", "Dietary review", "Regimen adjustment with clinician"],
            "warning_signs": ["Very high sugars with dehydration", "Vomiting or confusion"]
        }
    },
    {
        "q": "I am on basal–bolus but post-meal sugars spike",
        "a_json": {
            "summary": "Post-meal spikes suggest carb ratio/correction factor or timing adjustments may be needed.",
            "possible_diagnoses": ["Post-prandial hyperglycemia"],
            "rationale": ["Insufficient prandial coverage or timing mismatch"],
            "recommended_tests": ["Pre- and 2-hr post-meal glucose logs"],
            "next_steps": ["Log meals/glucose for 3 days", "Review carb ratio/correction factor", "Discuss timing with clinician"],
            "warning_signs": ["Frequent hypoglycemia", "Symptoms of DKA if very high sugars"]
        }
    }
]

def synthesize_dataset(n_train=120, n_val=24, seed=42) -> Path:
    random.seed(seed)
    items: List[Dict[str, str]] = []
    pool = SCHEMA_EXAMPLES
    variants = [
        ("itchy skin", "dry itchy skin", "skin itching"),
        ("slow-healing wound", "wound not healing", "foot ulcer"),
        ("urinate frequently", "peeing a lot", "frequent urination"),
        ("lost weight", "unintentional weight loss", "weight dropping"),
        ("post-meal spike", "after eating high sugar", "postprandial high"),
    ]
    for _ in range(n_train + n_val):
        ex = random.choice(pool)
        q = ex["q"]
        for a, b, c in variants:
            if a in q:
                q = q.replace(a, random.choice([a, b, c]))
        items.append({"question": q, "answer_json": json.dumps(ex["a_json"], ensure_ascii=False)})
    path = DATA_DIR / "diabetology_json.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path

def get_model_name() -> str:
    return os.getenv("BASE_MODEL", "microsoft/BioGPT")

def build_prompt(example: Dict[str, str]) -> str:
    q = (example["question"] or "").strip()
    a_json = (example["answer_json"] or "").strip()
    return (
        "You are a biomedical AI assistant specialized in diabetology.\n"
        "Return ONLY a JSON object with keys exactly: "
        '{"summary": string, "possible_diagnoses": string[], "rationale": string[], '
        '"recommended_tests": string[], "next_steps": string[], "warning_signs": string[]}. '
        "No extra text.\n\n"
        f"{JSON_BEGIN}\n{a_json}\n{JSON_END}\n\n"
        "Output must start with { and end with }.\n\n"
        f"Patient symptoms and history:\n{q}\n"
    )

def tokenize_function(example: Dict[str, str], tokenizer: AutoTokenizer, max_length: int):
    text = build_prompt(example)
    toks = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    toks["labels"] = toks["input_ids"].copy()
    return toks

def maybe_lora(model, use_peft: bool, r=16, alpha=32, dropout=0.05):
    if not use_peft:
        return model
    if not HAVE_PEFT:
        raise RuntimeError("peft not available. pip install peft")

    # Try common attention/MLP module names for BioGPT-like GPT architectures
    # Fallback to 'auto' if manual names fail.
    try:
        lcfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "attn.c_attn", "attn.c_proj",              # some GPT variants
                "mlp.c_fc", "mlp.c_proj",                  # MLP
                "q_proj","k_proj","v_proj","o_proj",       # HF-style
                "query","key","value","out_proj"           # alt names
            ],
        )
        return get_peft_model(model, lcfg)
    except Exception as e:
        print(f"[LoRA] Manual target_modules failed: {e}. Trying auto mapping...")
        lcfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules="auto",
        )
        return get_peft_model(model, lcfg)


def main() -> None:
    set_seed(42)
    # 1) Build tiny dataset
    jsonl_path = synthesize_dataset()
    ds = load_dataset("json", data_files=str(jsonl_path))["train"]
    split = ds.train_test_split(test_size=0.1667, seed=42)
    train_ds: Dataset = split["train"]
    val_ds: Dataset = split["test"]

    # 2) Load base + tokenizer
    base_model = get_model_name()
    use_peft = os.getenv("USE_LORA","1") == "1"
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
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
        print("Training without LoRA (full SFT). Set USE_LORA=1 for adapters.")

    # 3) Tokenize
    max_length = int(os.getenv("MAX_LENGTH","384"))
    tokenized_train = train_ds.map(lambda ex: tokenize_function(ex, tokenizer, max_length), batched=False)
    tokenized_val = val_ds.map(lambda ex: tokenize_function(ex, tokenizer, max_length), batched=False)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4) Short schedule
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE","2")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE","2")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM","4")),
        learning_rate=float(os.getenv("LR","1.5e-4" if use_peft else "5e-5")),
        num_train_epochs=float(os.getenv("EPOCHS","3")),
        warmup_ratio=float(os.getenv("WARMUP_RATIO","0.03")),
        lr_scheduler_type=os.getenv("SCHED","cosine"),
        logging_steps=int(os.getenv("LOG_STEPS","50")),
        eval_steps=int(os.getenv("EVAL_STEPS","200")),
        save_steps=int(os.getenv("SAVE_STEPS","500")),
        save_total_limit=int(os.getenv("SAVE_LIMIT","2")),
        fp16=os.getenv("FP16","1") == "1",
        bf16=os.getenv("BF16","0") == "1",
        gradient_checkpointing=False,
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
    print(f"Saved LoRA-tuned model to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
