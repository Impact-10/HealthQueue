#!/usr/bin/env python3
# Build diabetology_qna.jsonl/.json/.csv from online sources.
# Sources:
# 1) Hugging Face: abdelhakimDZ/diabetes_QA_dataset (QA pairs)
# 2) Kaggle ChatDoctor (optional, if CHATDOCTOR_PATH provided): filter diabetes-related turns

from __future__ import annotations
import os, re, json, random
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = DATA_DIR / "diabetology_qna.jsonl"
OUT_JSON  = DATA_DIR / "diabetology_qna.json"
OUT_CSV   = DATA_DIR / "diabetology_qna.csv"

DIABETES_KEYS = (
    "diabetes", "diabetic", "hypergly", "hypogly", "sugar", "glucose", "a1c", "hba1c", "insulin", "metformin"
)

def norm_space(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def sanitize_answer(a: str) -> str:
    a = norm_space(a)
    # Remove obvious boilerplate/news/legal artifacts
    drops = [
        "image credit", "shutterstock", "associated press", "copyright",
        "all rights reserved", "terms of use", "privacy policy", "editor's note",
        "contact us", "open-access article", "distributed under the terms",
        "for reprints", "press release", "standards editor", "news organization",
        "http://", "https://", "www/"
    ]
    low = a.lower()
    if any(d in low for d in drops):
        # Truncate at first sentence and keep the rest minimal
        a = a.split(".")[0]
    # Limit to ~5 sentences
    parts = [p.strip() for p in re.split(r"[.!?]", a) if p.strip()]
    a = ". ".join(parts[:5])
    if a and not a.endswith("."): a += "."
    return a

def load_hf_diabetes_qa() -> List[Dict[str, str]]:
    ds = load_dataset("abdelhakimDZ/diabetes_QA_dataset")
    rows: List[Dict[str, str]] = []
    for split in ds:
        for ex in ds[split]:
            q = norm_space(ex.get("question") or ex.get("Question") or "")
            a = norm_space(ex.get("answer") or ex.get("Answer") or "")
            if not q or not a: continue
            rows.append({"question": q, "answer": sanitize_answer(a)})
    return rows

def load_chatdoctor_filtered(chatdoctor_path: str) -> List[Dict[str, str]]:
    # Expect a JSON or JSONL with dialogues. We will take first user turn + first assistant turn.
    path = Path(chatdoctor_path)
    if not path.exists(): return []
    rows: List[Dict[str, str]] = []
    def is_diabetes_text(t: str) -> bool:
        low = (t or "").lower()
        return any(k in low for k in DIABETES_KEYS)
    # JSONL
    if path.suffix.lower() in (".jsonl", ".jsonl.gz"):
        import gzip
        opener = gzip.open if path.suffix.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                conv = obj.get("conversations") or obj.get("dialogue") or obj.get("messages")
                if not isinstance(conv, list) or len(conv) < 2: continue
                user = norm_space(conv[0].get("value") or conv[0].get("content") or "")
                bot  = norm_space(conv[1].get("value") or conv[1].get("content") or "")
                if user and bot and is_diabetes_text(user):
                    rows.append({"question": user, "answer": sanitize_answer(bot)})
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                conv = obj.get("conversations") or obj.get("dialogue") or obj.get("messages")
                if not isinstance(conv, list) or len(conv) < 2: continue
                user = norm_space(conv[0].get("value") or conv[0].get("content") or "")
                bot  = norm_space(conv[1].get("value") or conv[1].get("content") or "")
                if user and bot and is_diabetes_text(user):
                    rows.append({"question": user, "answer": sanitize_answer(bot)})
    return rows

def dedup(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for r in rows:
        key = (r["question"].lower(), r["answer"].lower())
        if key in seen: continue
        seen.add(key)
        out.append(r)
    return out

def balance_and_trim(rows: List[Dict[str, str]], max_len: int = 20000) -> List[Dict[str, str]]:
    # Keep up to max_len; shuffle for diversity
    random.shuffle(rows)
    return rows[:max_len]

def save_all(rows: List[Dict[str, str]]):
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(rows)} pairs to:\n- {OUT_JSONL}\n- {OUT_JSON}\n- {OUT_CSV}")

def main():
    random.seed(42)
    rows: List[Dict[str, str]] = []

    # 1) HF diabetes QA
    print("Loading abdelhakimDZ/diabetes_QA_dataset …")
    rows += load_hf_diabetes_qa()

    # 2) Optional ChatDoctor filter
    chat_path = os.getenv("CHATDOCTOR_PATH", "").strip()
    if chat_path:
        print(f"Loading ChatDoctor from: {chat_path}")
        rows += load_chatdoctor_filtered(chat_path)

    # Clean-up
    rows = [r for r in rows if len(r["question"]) >= 10 and len(r["answer"]) >= 20]
    rows = dedup(rows)
    rows = balance_and_trim(rows, max_len=int(os.getenv("MAX_QA", "30000")))

    # Light style normalization: ensure 3–5 sentences max in answers
    normd = []
    for r in rows:
        a = r["answer"]
        parts = [p.strip() for p in re.split(r"[.!?]", a) if p.strip()]
        a = ". ".join(parts[:5])
        if a and not a.endswith("."): a += "."
        normd.append({"question": r["question"], "answer": a})

    save_all(normd)

if __name__ == "__main__":
    main()
