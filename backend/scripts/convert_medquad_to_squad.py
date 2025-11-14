"""
Convert MedQuAD CSV splits to SQuAD v1-style JSON for extractive QA fine-tuning.

Input CSVs (produced earlier): backend/data/processed/train.csv, val.csv
- Columns: transcription ("Question: ...\n\nAnswer: ..."), label (int)

Output JSONs: backend/data/squad/medquad_train.json, medquad_val.json
SQuAD v1 schema (minimal):
{
  "version": "medquad_v1",
  "data": [
    {"title": "MedQuAD", "paragraphs": [
      {
        "context": <answer_full>,
        "qas": [{
          "id": <unique_id>,
          "question": <question_text>,
          "answers": [{"text": <answer_first_sentence>, "answer_start": <index>}]
        }]
      }
    ]}
  ]
}

Assumptions:
- We set context to the entire answer text. The labeled answer span is the first sentence
  of the answer (as requested). If the first sentence can't be found verbatim due to
  whitespace, we fall back to index 0.

Run:
  python backend/scripts/convert_medquad_to_squad.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "squad"


QA_SPLIT_RE = re.compile(r"(?:\\n\\n|\\r\\n\\r\\n|\n\n|\r\n\r\n)\s*Answer\s*:", re.IGNORECASE)
QUESTION_PREFIX_RE = re.compile(r"^\s*Question\s*:\s*", re.IGNORECASE)


def parse_qa(text: str) -> Tuple[str, str]:
    """Extract (question, answer) from a 'Question: ...\n\nAnswer: ...' blob.
    Returns empty strings if parsing fails.
    """
    if not text:
        return "", ""
    parts = QA_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        q_part, a_part = parts
        q = QUESTION_PREFIX_RE.sub("", q_part.strip()).replace("\\n", " ").strip()
        a = a_part.strip().replace("\\n", " ").strip()
        # Normalize whitespace
        q = re.sub(r"\s+", " ", q)
        a = re.sub(r"\s+", " ", a)
        return q, a
    # Fallback: try to heuristically split at first double newline literal
    fallback = re.split(r"\\n\\n|\n\n", text, maxsplit=1)
    if len(fallback) == 2:
        q = QUESTION_PREFIX_RE.sub("", fallback[0].strip()).replace("\\n", " ").strip()
        a = fallback[1].strip().replace("\\n", " ").strip()
        q = re.sub(r"\s+", " ", q)
        a = re.sub(r"\s+", " ", a)
        return q, a
    return "", text.strip()


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text: str) -> str:
    """Return the first sentence by a simple regex split; fallback to whole text."""
    if not text:
        return ""
    parts = SENTENCE_SPLIT_RE.split(text.strip(), maxsplit=1)
    if parts:
        return parts[0].strip()
    return text.strip()


def build_squad(df: pd.DataFrame, split_name: str) -> Dict:
    records = []
    for idx, row in df.iterrows():
        qa_text = str(row.get("transcription", ""))
        question, answer_full = parse_qa(qa_text)
        if not question:
            # Skip malformed rows
            continue
        ans_first = first_sentence(answer_full)
        # Find start index robustly (normalize spaces for search)
        context = answer_full
        try:
            start = context.find(ans_first)
            if start < 0:
                # Fallback to start of context
                start = 0
                ans_first = context[: min(len(context), len(ans_first))]
        except Exception:
            start = 0
            ans_first = context[: min(len(context), 64)]

        qas = [{
            "id": f"medquad-{split_name}-{idx}",
            "question": question,
            "answers": [{"text": ans_first, "answer_start": int(start)}],
        }]
        para = {"context": context, "qas": qas}
        records.append(para)

    data = [{"title": "MedQuAD", "paragraphs": records}]
    return {"version": "medquad_v1", "data": data}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = PROCESSED / "train.csv"
    val_csv = PROCESSED / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing processed CSVs under {PROCESSED}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"Loaded train: {len(train_df)} rows, val: {len(val_df)} rows")

    train_squad = build_squad(train_df, "train")
    val_squad = build_squad(val_df, "val")

    train_out = OUT_DIR / "medquad_train.json"
    val_out = OUT_DIR / "medquad_val.json"
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_squad, f, ensure_ascii=False)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_squad, f, ensure_ascii=False)

    print(f"Wrote: {train_out}")
    print(f"Wrote: {val_out}")


if __name__ == "__main__":
    main()
