# -*- coding: utf-8 -*-
"""
SAFE LAPTOP VERSION - Fine-tune BERT with thermal protection and conservative settings

This version is specifically designed for laptops with thermal constraints:
- Batch size 1 (minimal GPU load)
- num_workers=0 (no extra CPU threads)
- Frequent cache clearing
- Temperature monitoring (if pynvml available)
- Slower but SAFE

Usage:
  python backend/scripts/train_bert_qa_safe.py \
    --train_json data/squad/medquad_train.json \
    --val_json data/squad/medquad_val.json \
    --output_dir bert-medqa-custom \
    --epochs 3
"""
from __future__ import annotations

import sys
import io
# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import json
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    default_data_collator,
)
from transformers.optimization import get_linear_schedule_with_warmup
import evaluate

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITOR_AVAILABLE = True
except:
    GPU_MONITOR_AVAILABLE = False
    print("⚠️  pynvml not available - GPU temperature monitoring disabled")
    print("   Install with: pip install nvidia-ml-py3")


def check_gpu_temp():
    """Check GPU temperature and return temp in Celsius, or None if unavailable."""
    if not GPU_MONITOR_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return temp
    except:
        return None


def check_cuda_and_note():
    has_cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"CUDA available: {has_cuda} | Device: {device}")
    
    if has_cuda:
        temp = check_gpu_temp()
        if temp:
            print(f"   Current GPU Temperature: {temp}°C")
            if temp > 75:
                print(f"   ⚠️  GPU is already warm ({temp}°C)! Consider cooling before training.")
        
    if not has_cuda:
        print("\n[CUDA INSTALL HINTS] Training on CPU is very slow. Consider using a GPU machine or Colab.\n")
    return has_cuda


def load_squad_examples(json_path: str) -> List[Dict]:
    """Flatten SQuAD-style JSON to a list of {id, question, context, answers}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[Dict] = []
    for entry in data.get("data", []):
        for para in entry.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                ans_list = qa.get("answers", []) or []
                if ans_list:
                    texts = [a.get("text", "") for a in ans_list]
                    starts = [int(a.get("answer_start", 0)) for a in ans_list]
                else:
                    texts, starts = [""], [0]
                items.append(
                    {
                        "id": qa.get("id", ""),
                        "question": qa.get("question", ""),
                        "context": context,
                        "answers": {"text": texts, "answer_start": starts},
                    }
                )
    return items


def make_datasets(train_json: str, val_json: str, subset: float | None) -> DatasetDict:
    train_items = load_squad_examples(train_json)
    val_items = load_squad_examples(val_json)
    if subset and 0 < subset < 1:
        train_items = train_items[: max(1, int(len(train_items) * subset))]
        val_items = val_items[: max(1, int(len(val_items) * subset))]
        print(f"Using subset: train={len(train_items)}, val={len(val_items)}")
    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_items),
            "validation": Dataset.from_list(val_items),
        }
    )
    return ds


def prepare_features(tokenizer: BertTokenizerFast, max_len=384, doc_stride=128):
    def train_mapper(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        start_positions = []
        end_positions = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def val_mapper(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []
        for i in range(len(tokenized["input_ids"])):
            sample_idx = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_idx])
            sequence_ids = tokenized.sequence_ids(i)
            offset = tokenized["offset_mapping"][i]
            tokenized["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
        return tokenized

    return train_mapper, val_mapper


def postprocess_predictions(examples, features, predictions, tokenizer, n_best_size=20, max_answer_length=30):
    start_logits, end_logits = predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feat_id in enumerate(features["example_id"]):
        features_per_example[feat_id].append(i)
    final_predictions = {}
    for example_id, feature_indices in features_per_example.items():
        context = examples[example_id_to_index[example_id]]["context"]
        prelim = []
        for fi in feature_indices:
            start_logit = start_logits[fi]
            end_logit = end_logits[fi]
            offsets = features["offset_mapping"][fi]
            start_indexes = np.argsort(start_logit)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logit)[-n_best_size:][::-1]
            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s:
                        continue
                    length = offsets[e][1] - offsets[s][0]
                    if length <= 0 or length > max_answer_length * 10:
                        continue
                    prelim.append({"score": start_logit[s] + end_logit[e], "start": offsets[s][0], "end": offsets[e][1]})
        if prelim:
            best = max(prelim, key=lambda x: x["score"])
            final_predictions[example_id] = context[best["start"] : best["end"]]
        else:
            final_predictions[example_id] = ""
    return final_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="data/squad/medquad_train.json")
    parser.add_argument("--val_json", type=str, default="data/squad/medquad_val.json")
    parser.add_argument("--output_dir", type=str, default="./bert-medqa-custom")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_temp", type=int, default=90, help="Max GPU temp (°C) before pausing")
    parser.add_argument("--subset", type=float, default=0.0, help="Fraction of data (e.g., 0.01 for quick test)")
    args = parser.parse_args()

    print("=" * 80)
    print("🛡️  SAFE LAPTOP TRAINING MODE - BERT QA")
    print("=" * 80)
    print(f"Dataset: MedQuAD (NIH Medical Q&A)")
    print(f"Model: bert-base-uncased")
    print(f"Approach: Following proven GPU optimization strategy")
    print(f"Max GPU Temp: {args.max_temp}°C (will pause if exceeded)")
    print("=" * 80)

    # GPU optimization setup (following your proven approach)
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"\n🔥 NVIDIA GPU DETECTED: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory}GB")
        
        if "3050" in gpu_name:
            print("🎯 RTX 3050 DETECTED - OPTIMIZED TRAINING!")
        
        # Check initial temperature
        temp = check_gpu_temp()
        if temp:
            print(f"🌡️  Initial GPU Temperature: {temp}°C")
            if temp > 75:
                print(f"   ⚠️  GPU is already warm! Consider cooling before training.")
    else:
        print("\n⚠️  No CUDA GPU detected - training will be very slow on CPU")

    # Disable tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ds = make_datasets(args.train_json, args.val_json, subset=args.subset if args.subset > 0 else None)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_mapper, val_mapper = prepare_features(tokenizer)

    print("\n📝 Tokenizing datasets...")
    sys.stdout.flush()
    
    try:
        train_dataset = ds["train"].map(train_mapper, batched=True, remove_columns=ds["train"].column_names)
        print(f"   ✅ Train tokenization complete")
        sys.stdout.flush()
    except Exception as e:
        print(f"   ❌ Train tokenization failed: {e}")
        sys.stdout.flush()
        raise
    
    try:
        val_dataset = ds["validation"].map(val_mapper, batched=True, remove_columns=ds["validation"].column_names)
        print(f"   ✅ Validation tokenization complete")
        sys.stdout.flush()
    except Exception as e:
        print(f"   ❌ Validation tokenization failed: {e}")
        sys.stdout.flush()
        raise
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    sys.stdout.flush()

    print("\n🔧 Loading BERT model...")
    sys.stdout.flush()
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    print("   ✅ Model loaded from pretrained")
    sys.stdout.flush()
    
    # GPU optimization (following your proven approach)
    if has_cuda:
        print("   🔧 Applying GPU optimizations...")
        sys.stdout.flush()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        model.gradient_checkpointing_enable()
        print("   ✅ GPU optimizations enabled")
        print("   ✅ Gradient checkpointing enabled")
        sys.stdout.flush()

    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"   🔄 Moving model to {device}...")
    sys.stdout.flush()
    model.to(device)
    print(f"   ✅ Model on {device}")
    sys.stdout.flush()

    # CRITICAL: num_workers=0 to avoid CPU overload, pin_memory for faster transfer
    print("\n   🔧 Creating DataLoaders...")
    sys.stdout.flush()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,  # Increased from 1 to 2 (like your training)
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,  # NO multiprocessing workers
        pin_memory=True if has_cuda else False,  # Fast GPU transfer
    )
    print(f"   ✅ Train loader created: {len(train_loader)} batches")
    sys.stdout.flush()
    
    def collate_eval(features):
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        batch = {}
        for k in keys:
            if k in features[0]:
                batch[k] = torch.tensor([f[k] for f in features])
        return batch

    print("   🔧 Creating validation loader...")
    sys.stdout.flush()
    
    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,  # Larger batch for eval (no gradients)
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=0,
        pin_memory=True if has_cuda else False,
    )
    print(f"   ✅ Eval loader created: {len(eval_loader)} batches")
    sys.stdout.flush()

    # Optimizer with gradient accumulation to simulate larger batch
    print("\n   🔧 Creating optimizer and scheduler...")
    sys.stdout.flush()
    
    gradient_accumulation_steps = 8  # Effective batch = 2 * 8 = 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    print("   ✅ Optimizer created")
    sys.stdout.flush()
    
    total_update_steps = (len(train_loader) // gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_update_steps // 10, num_training_steps=total_update_steps
    )
    print("   ✅ Scheduler created")
    sys.stdout.flush()

    scaler = torch.cuda.amp.GradScaler(enabled=has_cuda)
    print("   ✅ GradScaler created")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    sys.stdout.flush()
    print("🏋️  SAFE TRAINING CONFIGURATION (Following Proven Approach)")
    sys.stdout.flush()
    print("=" * 80)
    sys.stdout.flush()
    print(f"Epochs: {args.epochs}")
    sys.stdout.flush()
    print(f"Batch Size: 2 (GPU optimized)")
    sys.stdout.flush()
    print(f"Gradient Accumulation: {gradient_accumulation_steps}")
    sys.stdout.flush()
    print(f"Effective Batch: 16")
    sys.stdout.flush()
    print(f"num_workers: 0 (no CPU parallelism)")
    sys.stdout.flush()
    print(f"pin_memory: True (fast GPU transfer)")
    sys.stdout.flush()
    print(f"cudnn.benchmark: True (optimized conv operations)")
    sys.stdout.flush()
    print(f"Max GPU Temp: {args.max_temp}°C")
    sys.stdout.flush()
    print(f"Total Steps: {total_update_steps}")
    sys.stdout.flush()
    print("=" * 80)
    sys.stdout.flush()

    # Training loop with thermal protection
    print("\n🚀 Starting SAFE training (will be slower but stable)...", flush=True)
    print("=" * 80, flush=True)
    
    model.train()
    
    global_step = 0
    epoch_losses = []
    pauses = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}", flush=True)
        print(f"📍 EPOCH {epoch+1}/{args.epochs}", flush=True)
        print(f"{'='*80}", flush=True)
        running_loss = 0.0
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader, start=1):
            # Thermal check before processing
            if has_cuda and step % 100 == 0:  # Check every 100 steps
                temp = check_gpu_temp()
                if temp and temp > args.max_temp:
                    print(f"\n🌡️  GPU too hot ({temp}°C > {args.max_temp}°C)! Pausing 30s...")
                    pauses += 1
                    torch.cuda.empty_cache()
                    time.sleep(30)
                    temp_after = check_gpu_temp()
                    print(f"   Resumed (temp now {temp_after}°C)")
            
            # Move batch to GPU with non_blocking (like your training)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=has_cuda):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            
            if step % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += loss.item()
            epoch_loss += loss.item()
            
            # Log every 50 steps (more frequent feedback)
            if global_step > 0 and global_step % 50 == 0:
                avg_loss = running_loss / 50
                temp_str = ""
                if has_cuda:
                    temp = check_gpu_temp()
                    if temp:
                        temp_str = f" | GPU: {temp}°C"
                print(f"   Step {global_step:5d} | Loss: {avg_loss:.4f}{temp_str}", flush=True)
                running_loss = 0.0
            
            # Clear cache every 100 steps
            if has_cuda and step % 100 == 0:
                torch.cuda.empty_cache()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"\n✅ Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}", flush=True)
        
        # Save checkpoint after each epoch
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"   💾 Checkpoint saved to {args.output_dir}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("✅ TRAINING COMPLETE!", flush=True)
    print("=" * 80)
    print(f"Total thermal pauses: {pauses}")
    print(f"Final model saved to: {args.output_dir}")
    
    # Save training summary
    summary = {
        "epochs": args.epochs,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "total_steps": global_step,
        "dataset_size": len(train_dataset),
        "model": "bert-base-uncased",
        "task": "question_answering",
        "thermal_pauses": pauses,
        "safe_mode": True
    }
    with open(f"{args.output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Training summary saved to {args.output_dir}/training_summary.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
