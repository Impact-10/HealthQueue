
from __future__ import annotations

import sys
import io
# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import json
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


def check_cuda_and_note():
    has_cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"CUDA available: {has_cuda} | Device: {device}")
    if not has_cuda:
        print("\n[CUDA INSTALL HINTS] Training on CPU is very slow. Consider using a GPU machine or Colab.\n"
              "Windows (NVIDIA): Install latest NVIDIA driver, then CUDA Toolkit (matching PyTorch build),\n"
              "then pip install torch with CUDA (see https://pytorch.org/).\n"
              "Linux: Install NVIDIA driver + CUDA toolkit, then reinstall torch with the CUDA variant.\n")
    return has_cuda


def load_squad_examples(json_path: str) -> List[Dict]:
    """Flatten SQuAD-style JSON to a list of {id, question, context, answers}.
    answers = {"text": [..], "answer_start": [..]} per HF convention.
    """
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
    from collections import defaultdict
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
    parser.add_argument("--train_json", type=str, default="backend/data/squad/medquad_train.json")
    parser.add_argument("--val_json", type=str, default="backend/data/squad/medquad_val.json")
    parser.add_argument("--output_dir", type=str, default="./bert-medqa-custom")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--subset", type=float, default=0.0, help="Fraction of data to use (e.g., 0.01)")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BERT QUESTION ANSWERING TRAINING")
    print("=" * 80)
    print(f"Dataset: MedQuAD (NIH Medical Q&A)")
    print(f"Model: bert-base-uncased (110M parameters)")
    print(f"Task: Extractive Question Answering")
    print("=" * 80)

    has_cuda = check_cuda_and_note()

    # Enable gradient checkpointing to save memory
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ds = make_datasets(args.train_json, args.val_json, subset=args.subset if args.subset > 0 else None)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_mapper, val_mapper = prepare_features(tokenizer)

    print("\n📝 Tokenizing datasets...")
    train_dataset = ds["train"].map(train_mapper, batched=True, remove_columns=ds["train"].column_names)
    val_dataset = ds["validation"].map(val_mapper, batched=True, remove_columns=ds["validation"].column_names)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    print("\n🔧 Loading BERT model...")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    
    # Enable gradient checkpointing for memory efficiency
    if has_cuda:
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled (saves VRAM)")

    # Move model to device
    device = torch.device("cuda" if has_cuda else "cpu")
    model.to(device)
    print(f"   ✅ Model moved to {device}")

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    def collate_eval(features):
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        batch = {}
        for k in keys:
            if k in features[0]:
                batch[k] = torch.tensor([f[k] for f in features])
        return batch

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_eval,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_update_steps = (
        (len(train_loader) // max(1, args.gradient_accumulation_steps)) * max(1, args.epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(0, total_update_steps // 10), num_training_steps=total_update_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=has_cuda)

    print("\n" + "=" * 80)
    print("🏋️  TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.per_device_train_batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning Rate: 3e-5")
    print(f"Total Steps: {total_update_steps}")
    print(f"Mixed Precision (FP16): {has_cuda}")
    print("=" * 80)

    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 80)
    model.train()
    global_step = 0
    epoch_losses = []
    
    try:
        for epoch in range(args.epochs):
            print(f"\n📍 EPOCH {epoch+1}/{args.epochs}")
            print("-" * 80)
            running_loss = 0.0
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_loader, start=1):
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=has_cuda):
                        outputs = model(**batch)
                        loss = outputs.loss / max(1, args.gradient_accumulation_steps)

                    scaler.scale(loss).backward()
                    
                    if step % max(1, args.gradient_accumulation_steps) == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1

                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if global_step > 0 and global_step % 100 == 0:
                        avg_loss = running_loss / 100
                        print(f"   Step {global_step:5d} | Loss: {avg_loss:.4f}")
                        running_loss = 0.0
                        
                        # Clear cache periodically
                        if has_cuda:
                            torch.cuda.empty_cache()
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  GPU Out of Memory at step {step}!")
                        print("   Clearing cache and skipping batch...")
                        if has_cuda:
                            torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        raise e
                        
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            epoch_losses.append(avg_epoch_loss)
            print(f"\n   ✅ Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Saving current model state...")

    # Save model & tokenizer
    print(f"\n💾 Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("   ✅ Model saved successfully!")

    # Save training summary
    summary = {
        "epochs": args.epochs,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "total_steps": global_step,
        "dataset_size": len(train_dataset),
        "model": "bert-base-uncased",
        "task": "question_answering"
    }
    
    import json
    with open(f"{args.output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # EM/F1 evaluation with post-processing
    print("\n" + "=" * 80)
    print("📊 RUNNING EVALUATION")
    print("=" * 80)
    model.eval()
    all_start_logits: List[np.ndarray] = []
    all_end_logits: List[np.ndarray] = []
    
    # Clear CUDA cache before eval to avoid OOM
    if has_cuda:
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
            all_end_logits.append(outputs.end_logits.detach().cpu().numpy())
            
            # Clear batch from GPU immediately
            if has_cuda:
                del batch, outputs
                torch.cuda.empty_cache()

    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    print("\n📈 Computing SQuAD metrics (EM/F1)...")
    squad_metric = evaluate.load("squad")
    predictions = postprocess_predictions(ds["validation"], val_dataset, (start_logits, end_logits), tokenizer)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in ds["validation"]]
    preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    metrics = squad_metric.compute(predictions=preds, references=references)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final Validation Metrics:")
    print(f"   Exact Match (EM): {metrics['exact_match']:.2f}%")
    print(f"   F1 Score: {metrics['f1']:.2f}%")
    print(f"\nModel saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
