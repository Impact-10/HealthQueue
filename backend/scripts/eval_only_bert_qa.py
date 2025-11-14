"""
Run evaluation only on a saved BERT QA model to compute EM/F1 metrics.

Usage:
  python backend/scripts/eval_only_bert_qa.py \
    --model_dir ./bert-medqa-custom \
    --val_json backend/data/squad/medquad_val.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast
import evaluate


def load_squad_examples(json_path: str) -> List[Dict]:
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


def prepare_val_features(tokenizer, examples, max_len=384, doc_stride=128):
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
    parser.add_argument("--model_dir", type=str, default="./bert-medqa-custom")
    parser.add_argument("--val_json", type=str, default="data/squad/medquad_val.json")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_dir}...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForQuestionAnswering.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Load validation data
    print(f"Loading validation data from {args.val_json}...")
    val_items = load_squad_examples(args.val_json)
    ds_val = Dataset.from_list(val_items)

    # Tokenize
    print("Tokenizing validation set...")
    val_dataset = ds_val.map(
        lambda ex: prepare_val_features(tokenizer, ex, max_len=384, doc_stride=128),
        batched=True,
        remove_columns=ds_val.column_names,
    )

    # Custom collator for eval (no start/end positions)
    def collate_eval(features):
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        batch = {}
        for k in keys:
            if k in features[0]:
                batch[k] = torch.tensor([f[k] for f in features])
        return batch

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_eval,
    )

    # Run inference
    print("Running inference on validation set...")
    all_start_logits: List[np.ndarray] = []
    all_end_logits: List[np.ndarray] = []

    if has_cuda:
        torch.cuda.empty_cache()

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
            all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

            if has_cuda:
                del batch, outputs
                torch.cuda.empty_cache()

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(eval_loader)} batches...")

    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process and compute metrics
    print("Post-processing predictions...")
    squad_metric = evaluate.load("squad")
    predictions = postprocess_predictions(ds_val, val_dataset, (start_logits, end_logits), tokenizer)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in ds_val]
    preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    metrics = squad_metric.compute(predictions=preds, references=references)

    print("\n" + "="*60)
    print("FINAL VALIDATION METRICS")
    print("="*60)
    print(f"Exact Match (EM): {metrics['exact_match']:.2f}")
    print(f"F1 Score:         {metrics['f1']:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
