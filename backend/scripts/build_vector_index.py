"""Build and cache sentence-transformer embeddings for hybrid retrieval.

Run:
    python -u scripts/build_vector_index.py --model all-MiniLM-L6-v2

Embeddings saved to: backend/vector_index/{model_name}/embeddings.npy
Metadata saved to:   backend/vector_index/{model_name}/meta.json

If sentence-transformers is not installed, script exits gracefully.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from utils import retriever  # ensures corpus loading

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "vector_index"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if SentenceTransformer is None:
        print("❌ sentence-transformers not installed. Install first to build embeddings.")
        return

    df = retriever._load_corpus()  # noqa: protected access for build script
    texts = df["transcription"].astype(str).tolist()
    print(f"Loaded {len(texts)} documents for embedding.")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print("Encoding...")
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    model_name_sanitized = args.model.replace('/', '_')
    out_path = OUT_DIR / model_name_sanitized
    out_path.mkdir(parents=True, exist_ok=True)

    emb_file = out_path / "embeddings.npy"
    meta_file = out_path / "meta.json"

    np.save(str(emb_file), embeddings)
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({"model": args.model, "count": len(texts)}, f, indent=2)

    print(f"✅ Saved embeddings: {emb_file}")
    print(f"✅ Saved metadata:   {meta_file}")
    print("Hybrid retrieval will auto-load these when available.")


if __name__ == "__main__":
    main()
