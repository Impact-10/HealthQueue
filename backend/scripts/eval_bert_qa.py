"""
Run inference with the fine-tuned BERT QA model on sample questions.

Usage:
  python backend/scripts/eval_bert_qa.py \
    --model_dir ./bert-medqa-custom \
    --val_json backend/data/squad/medquad_val.json \
    --num_samples 3

Or provide your own context/question via CLI.
"""
from __future__ import annotations

import argparse
import json
import string
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizerFast


# --------------------
# Helper functions for full validation (Exact Match & F1 like SQuAD)
# --------------------
def _normalize_answer(s: str) -> str:
    """Lower, remove punctuation/articles/extra whitespace (SQuAD style)."""
    def lower(text: str) -> str:
        return text.lower()
    def remove_punc(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _load_squad(val_json: str) -> List[Dict[str, str]]:
    with open(val_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    for entry in data.get('data', []):
        for para in entry.get('paragraphs', []):
            context = para.get('context', '')
            for qa in para.get('qas', []):
                q = qa.get('question', '')
                answers = qa.get('answers', []) or qa.get('answer', [])
                # Some medquad entries may store single answer differently; ensure list of dicts with 'text'
                golds = []
                for a in answers:
                    if isinstance(a, dict) and 'text' in a:
                        golds.append(a['text'])
                    elif isinstance(a, str):
                        golds.append(a)
                if not golds:
                    golds = ['']
                samples.append({'question': q, 'context': context, 'answers': golds})
    return samples


def _predict_answer(model, tokenizer, question: str, context: str, max_length: int = 384) -> str:
    encoded = tokenizer(
        question,
        context,
        return_tensors='pt',
        truncation='only_second',
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**{k: v for k, v in encoded.items() if k in {'input_ids', 'token_type_ids', 'attention_mask'}})
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()
    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded['offset_mapping'][0].tolist()
    # Mask non-context tokens
    context_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1 and offsets[i] is not None]
    if not context_indices:
        return ''
    # Build restricted arrays
    start_logits_ctx = {i: start_logits[i] for i in context_indices}
    end_logits_ctx = {i: end_logits[i] for i in context_indices}
    # Choose best span (naive argmax start+end with constraint end>=start and max length)
    best_score = -1e9
    best_span = (None, None)
    max_span_len = 30
    for s in context_indices:
        for e in context_indices:
            if e < s:
                continue
            if (e - s + 1) > max_span_len:
                continue
            score = start_logits_ctx[s] + end_logits_ctx[e]
            if score > best_score:
                best_score = score
                best_span = (s, e)
    if best_span[0] is None:
        return ''
    s, e = best_span
    start_char, _ = offsets[s]
    _, end_char = offsets[e]
    return context[start_char:end_char].strip()


def n_best_answers(
    model,
    tokenizer,
    question: str,
    context: str,
    n_best: int = 3,
    max_answer_length_tokens: int = 30,
) -> List[Tuple[str, float]]:
    """Return top-n answers with confidence scores using logits over context tokens only."""
    model.eval()
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**{k: v for k, v in encoded.items() if k in {"input_ids", "token_type_ids", "attention_mask"}})
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()

    # Consider only context tokens (sequence_id == 1)
    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded["offset_mapping"][0].tolist()
    context_token_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1 and offsets[i] is not None]
    if not context_token_indices:
        return []

    # Softmax scores restricted to context tokens
    start_log_ctx = start_logits[context_token_indices]
    end_log_ctx = end_logits[context_token_indices]
    start_prob_ctx = np.exp(start_log_ctx - np.max(start_log_ctx))
    start_prob_ctx = start_prob_ctx / np.sum(start_prob_ctx)
    end_prob_ctx = np.exp(end_log_ctx - np.max(end_log_ctx))
    end_prob_ctx = end_prob_ctx / np.sum(end_prob_ctx)

    # Map back to full token indices
    ctx_top_s = np.argsort(start_log_ctx)[-n_best:][::-1]
    ctx_top_e = np.argsort(end_log_ctx)[-n_best:][::-1]

    candidates = []
    for si in ctx_top_s:
        s = context_token_indices[int(si)]
        for ei in ctx_top_e:
            e = context_token_indices[int(ei)]
            if e < s:
                continue
            if (e - s + 1) > max_answer_length_tokens:
                continue
            start_char, _ = offsets[s]
            _, end_char = offsets[e]
            ans_text = context[start_char:end_char]
            score = float(start_prob_ctx[int(si)] * end_prob_ctx[int(ei)])
            candidates.append((ans_text.strip(), score))

    # Deduplicate while keeping highest score
    best_map: Dict[str, float] = {}
    for text, sc in candidates:
        if not text:
            continue
        if text not in best_map or sc > best_map[text]:
            best_map[text] = sc

    # Sort by score
    return sorted(best_map.items(), key=lambda x: x[1], reverse=True)[:n_best]


def load_val_samples(val_json: str, n: int) -> List[Dict[str, str]]:
    with open(val_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for entry in data.get("data", []):
        for para in entry.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                q = qa.get("question", "")
                samples.append({"question": q, "context": context})
                if len(samples) >= n:
                    return samples
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./bert-medqa-custom")
    parser.add_argument("--val_json", type=str, default="backend/data/squad/medquad_val.json")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--full_eval", action="store_true", help="Run full EM/F1 eval over entire val set")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of validation examples for quick eval")
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForQuestionAnswering.from_pretrained(args.model_dir)

    if args.question and args.context:
        preds = n_best_answers(model, tokenizer, args.question, args.context, n_best=3)
        print("Q:", args.question)
        for i, (text, score) in enumerate(preds, 1):
            print(f"A{i} [{score:.4f}]: {text}")
        return

    if args.full_eval:
        print("Running full validation (EM/F1)...")
        samples = _load_squad(args.val_json)
        if args.limit and args.limit > 0:
            samples = samples[:args.limit]
        em_total, f1_total = 0.0, 0.0
        for ex in samples:
            pred = _predict_answer(model, tokenizer, ex['question'], ex['context'])
            # Use best matching gold for F1
            em_scores = [_exact_match(pred, g) for g in ex['answers']]
            f1_scores = [_f1_score(pred, g) for g in ex['answers']]
            em_total += max(em_scores)
            f1_total += max(f1_scores)
        em = 100.0 * em_total / len(samples)
        f1 = 100.0 * f1_total / len(samples)
        results = {"exact_match": em, "f1": f1, "total": len(samples)}
        print(json.dumps(results, indent=2))
        # Persist
        out_path = Path(args.model_dir) / "eval_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved eval results to {out_path}")
        return

    # Else run a few sample predictions
    samples = load_val_samples(args.val_json, args.num_samples)
    for i, ex in enumerate(samples, 1):
        preds = n_best_answers(model, tokenizer, ex['question'], ex['context'], n_best=3)
        print(f"\nSample {i}")
        print("Q:", ex['question'])
        for j, (text, score) in enumerate(preds, 1):
            print(f"Pred{j} [{score:.4f}]: {text}")


if __name__ == "__main__":
    main()
