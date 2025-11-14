import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Optional sentence-transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # runtime fallback
    SentenceTransformer = None  # type: ignore

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
except ImportError:
    rf_process = None  # type: ignore
    rf_fuzz = None  # type: ignore

from .spell_norm import normalize_query  # spell correction / expansion


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_DIR = BASE_DIR / "data"

_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None
_records: Optional[pd.DataFrame] = None
_label_map: Dict[str, Any] = {}

_emb_model = None  # sentence transformer instance
_embeddings = None  # Dense embedding matrix
_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight; swap to biomedical later


def _load_label_map() -> Dict[str, Any]:
    global _label_map
    label_map_path = PROCESSED_DIR / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            _label_map = json.load(f)
    else:
        _label_map = {}
    return _label_map


def _label_to_class(idx: int) -> Optional[str]:
    if not _label_map:
        _load_label_map()
    # download_medquad saves {'num_classes','classes':[...],'encoding':{name:idx}}
    try:
        classes = _label_map.get("classes", [])
        if isinstance(idx, (int,)) and 0 <= idx < len(classes):
            return classes[idx]
    except Exception:
        pass
    return None


def _parse_qa(text: str) -> Dict[str, str]:
    """Heuristically split combined 'Question:...\n\nAnswer:...' into parts."""
    q, a = None, None
    if text:
        s = text
        # Split on 'Answer:' preceded by blank lines (handle both literal \n and escaped \\n)
        # Pattern matches: one or more whitespace/newlines, then "Answer:"
        parts = re.split(r'(?:\\n\\n|\\r\\n\\r\\n|\n\n|\r\n\r\n)\s*Answer\s*:', s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            q_part, a_part = parts[0], parts[1]
            # Remove leading 'Question:' prefix (handle both literal and escaped newlines)
            q_clean = re.sub(r'^\s*Question\s*:\s*', '', q_part, flags=re.IGNORECASE)
            q_clean = q_clean.replace('\\n', ' ').strip()
            q = q_clean
            # Clean up answer (convert escaped newlines to spaces for readability)
            a_clean = a_part.replace('\\n', ' ').strip()
            a = re.sub(r'\s+', ' ', a_clean)  # normalize whitespace
        else:
            # fallback: keep whole text as answer
            a = text.replace('\\n', ' ').strip()
            a = re.sub(r'\s+', ' ', a)
    return {"question": q or "", "answer": a or (text or "")}


def _load_corpus() -> pd.DataFrame:
    """Load and combine MedQuAD Q&A pairs + MTSamples transcriptions."""
    global _records
    if _records is not None:
        return _records

    dfs: List[pd.DataFrame] = []
    
    # 1. Load MedQuAD Q&A pairs (processed train/val/test)
    medquad_paths = [PROCESSED_DIR / f for f in ("train.csv", "val.csv", "test.csv")]
    for p in medquad_paths:
        if p.exists():
            df = pd.read_csv(p)
            if "transcription" in df.columns:
                df_subset = df[["transcription", "label"]].copy()
                df_subset["source"] = "medquad"
                df_subset["specialty"] = df.get("label", "")
                dfs.append(df_subset)
    
    # 2. Load MTSamples medical transcriptions
    mtsamples_path = DATA_DIR / "mtsamples.csv"
    if mtsamples_path.exists():
        df_mt = pd.read_csv(mtsamples_path)
        if "transcription" in df_mt.columns:
            # Create unified schema
            df_mt_subset = pd.DataFrame({
                "transcription": df_mt["transcription"],
                "label": -1,  # No label mapping for MTSamples
                "source": "mtsamples",
                "specialty": df_mt.get("medical_specialty", ""),
            })
            dfs.append(df_mt_subset)
    
    if not dfs:
        raise FileNotFoundError(
            f"No corpus data found. Checked:\n"
            f"  - MedQuAD: {PROCESSED_DIR}\n"
            f"  - MTSamples: {mtsamples_path}"
        )

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["transcription"]).reset_index(drop=True)
    _records = df_all
    _load_label_map()
    
    print(f"✓ Loaded corpus: {len(df_all)} documents")
    print(f"  - MedQuAD Q&A: {len(df_all[df_all['source'] == 'medquad'])}")
    print(f"  - MTSamples: {len(df_all[df_all['source'] == 'mtsamples'])}")
    
    return _records


def _ensure_index():
    global _vectorizer, _matrix
    if _vectorizer is not None and _matrix is not None:
        return

    df = _load_corpus()
    texts = df["transcription"].astype(str).tolist()
    _vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200_000,
    )
    _matrix = _vectorizer.fit_transform(texts)

def _ensure_embeddings():
    """Load / compute sentence-transformer embeddings lazily."""
    global _emb_model, _embeddings
    if _embeddings is not None and _emb_model is not None:
        return
    if SentenceTransformer is None:
        return  # dependency not installed
    # Try to load pre-computed embeddings first
    import pickle
    import os
    embeddings_path = PROCESSED_DIR / "embeddings.pkl"
    if embeddings_path.exists():
        try:
            with open(embeddings_path, "rb") as f:
                _embeddings, _emb_model_name = pickle.load(f)
            if _emb_model_name == _EMB_MODEL_NAME:
                _emb_model = SentenceTransformer(_EMB_MODEL_NAME)
                print(f"✓ Loaded pre-computed embeddings from {embeddings_path}")
                return
        except Exception as e:
            print(f"Warning: Could not load embeddings cache: {e}")
    
    # Compute embeddings if not cached
    df = _load_corpus()
    texts = df["transcription"].astype(str).tolist()
    print(f"[Retrieval] Computing embeddings for {len(texts)} documents (this may take a while)...")
    _emb_model = SentenceTransformer(_EMB_MODEL_NAME)
    _embeddings = _emb_model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    # Cache embeddings for next time
    try:
        with open(embeddings_path, "wb") as f:
            pickle.dump((_embeddings, _EMB_MODEL_NAME), f)
        print(f"✓ Cached embeddings to {embeddings_path}")
    except Exception as e:
        print(f"Warning: Could not cache embeddings: {e}")

def _semantic_search(query: str, k: int = 5) -> List[Tuple[int, float]]:
    if SentenceTransformer is None:
        return []
    _ensure_embeddings()
    if _emb_model is None or _embeddings is None:
        return []
    q_vec = _emb_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    # cosine similarity manually (embeddings normalized -> dot product)
    # Use numpy for faster computation
    import numpy as np
    scores = np.dot(_embeddings, q_vec).tolist()
    # Use argpartition for faster top-k (O(n) vs O(n log n))
    k_actual = min(k * 2, len(scores))  # Get more candidates for re-ranking
    top_idx = np.argpartition(scores, -k_actual)[-k_actual:]
    top_idx = top_idx[np.argsort([scores[i] for i in top_idx])[::-1]][:k]
    return [(int(i), float(scores[i])) for i in top_idx]

def _fuzzy_boost(original_query: str, candidates: List[int], k: int = 5) -> List[int]:
    """Optionally reorder candidates using rapidfuzz partial_ratio against transcription text.
    
    Uses multiple fuzzy matching strategies:
    1. Partial ratio (substring matching)
    2. Token sort ratio (word order independent)
    3. Token set ratio (word set matching)
    """
    if rf_process is None or rf_fuzz is None:
        return candidates
    assert _records is not None
    scored = []
    for i in candidates:
        text = str(_records.iloc[int(i)]["transcription"])[:400]  # limit length
        # Combine multiple fuzzy scores for better matching
        partial_score = rf_fuzz.partial_ratio(original_query, text)
        token_sort_score = rf_fuzz.token_sort_ratio(original_query, text)
        token_set_score = rf_fuzz.token_set_ratio(original_query, text)
        # Weighted average (partial is most important for typos)
        combined_score = (partial_score * 0.5 + token_sort_score * 0.3 + token_set_score * 0.2)
        scored.append((i, combined_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in scored][:k]


def search(query: str, k: int = 5) -> Dict[str, Any]:
    """Return top-k most relevant medical documents for a query.

    Returns a dict with {query, k, results:[{rank, score, content, source, specialty}]}
    Handles both MedQuAD Q&A pairs and MTSamples transcriptions.
    """
    if not query or not query.strip():
        return {"query": query, "k": k, "results": []}

    # Normalize / spell-correct
    norm_query = normalize_query(query)
    _ensure_index()
    assert _vectorizer is not None and _matrix is not None and _records is not None

    q_vec = _vectorizer.transform([norm_query])
    sims = cosine_similarity(q_vec, _matrix).ravel()
    if k <= 0:
        k = 5
    top_idx = sims.argsort()[::-1][:k]

    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_idx, start=1):
        row = _records.iloc[int(i)]
        source = row.get("source", "unknown")
        
        # Parse based on source type
        if source == "medquad":
            qa = _parse_qa(str(row["transcription"]))
            label_idx = int(row.get("label", -1)) if pd.notna(row.get("label")) else -1
            label_name = _label_to_class(label_idx)
            results.append({
                "rank": rank,
                "score": float(sims[i]),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "label": label_idx,
                "label_name": label_name,
                "source": source,
                "specialty": row.get("specialty", ""),
            })
        else:  # mtsamples or other
            # For transcriptions, just return the full text
            results.append({
                "rank": rank,
                "score": float(sims[i]),
                "question": None,
                "answer": str(row["transcription"])[:2000],  # Truncate long transcriptions
                "label": -1,
                "label_name": None,
                "source": source,
                "specialty": row.get("specialty", ""),
            })
    
    return {"query": query, "normalized_query": norm_query, "k": k, "results": results}

def hybrid_search(query: str, k: int = 5, alpha: float = 0.5, timeout: float = 2.0, use_semantic: bool = False) -> Dict[str, Any]:
    """Combine lexical (TF-IDF) and semantic embeddings across MedQuAD + MTSamples.
    alpha: weight for semantic score (0..1). Final score = (1-alpha)*lex + alpha*sem.
    Falls back to pure lexical if semantic embeddings unavailable or timeout.
    timeout: Maximum seconds to wait for semantic search before falling back to lexical-only.
    use_semantic: If False, skip semantic search entirely (faster, more reliable).
    """
    if not query:
        return {"query": query, "results": []}
    norm_query = normalize_query(query)
    _ensure_index()
    assert _vectorizer is not None and _matrix is not None and _records is not None
    
    # Lexical scores (always fast and reliable)
    lex_vec = _vectorizer.transform([norm_query])
    lex_scores = cosine_similarity(lex_vec, _matrix).ravel()
    
    # Try semantic search ONLY if explicitly enabled and embeddings are already loaded
    sem_map = {}
    semantic_enabled = False
    
    if use_semantic and SentenceTransformer is not None:
        try:
            import time
            start = time.time()
            
            # Only use semantic if embeddings are ALREADY loaded (don't wait for loading)
            if _embeddings is not None and _emb_model is not None:
                # Quick semantic search with limited k
                sem_pairs = _semantic_search(norm_query, k=min(k * 2, 20))
                sem_map = {i: s for i, s in sem_pairs}
                semantic_enabled = True
                
                if time.time() - start > timeout:
                    # Took too long, fallback to lexical
                    sem_map = {}
                    semantic_enabled = False
            # If embeddings not loaded, skip semantic (don't wait)
        except Exception as e:
            # Fallback to lexical-only on any error
            semantic_enabled = False
    
    # Combine scores (or use lexical-only if semantic failed/disabled)
    combined = []
    for i in range(len(lex_scores)):
        if i in sem_map and semantic_enabled:
            score = (1 - alpha) * lex_scores[i] + alpha * sem_map[i]
        else:
            score = lex_scores[i]  # Pure lexical if no semantic
        combined.append((i, float(score)))
    
    # Faster sorting - use numpy for large arrays, Python sort for small
    if len(combined) > 1000:
        import numpy as np
        combined_arr = np.array(combined, dtype=[('idx', int), ('score', float)])
        combined_arr = np.sort(combined_arr, order='score')[::-1]
        top = [int(item[0]) for item in combined_arr[:min(k * 2, 50)]]
    else:
        combined.sort(key=lambda x: x[1], reverse=True)
        top = [i for i,_ in combined[:min(k * 2, 50)]]
    
    # Skip fuzzy boost (it's optional and can be slow)
    # Uncomment if needed, but it adds latency
    # try:
    #     top = _fuzzy_boost(norm_query, top, k=k)
    # except:
    #     pass
    
    final = top[:k]
    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(final, start=1):
        try:
            row = _records.iloc[int(idx)]
            source = row.get("source", "unknown")
            
            # Get score safely
            score_val = 0.0
            try:
                if idx < len(combined):
                    score_val = combined[idx][1]
                elif isinstance(idx, int) and 0 <= idx < len(lex_scores):
                    score_val = float(lex_scores[idx])
            except (IndexError, TypeError):
                pass
            
            # Parse based on source type
            if source == "medquad":
                qa = _parse_qa(str(row["transcription"]))
                label_idx = int(row.get("label", -1)) if pd.notna(row.get("label")) else -1
                label_name = _label_to_class(label_idx)
                results.append({
                    "rank": rank,
                    "score": score_val,
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "label": label_idx,
                    "label_name": label_name,
                    "source": source,
                    "specialty": row.get("specialty", ""),
                })
            else:  # mtsamples or other
                results.append({
                    "rank": rank,
                    "score": score_val,
                    "question": None,
                    "answer": str(row["transcription"])[:2000],  # Truncate long transcriptions
                    "label": -1,
                    "label_name": None,
                    "source": source,
                    "specialty": row.get("specialty", ""),
                })
        except (IndexError, KeyError, ValueError, TypeError) as e:
            # Skip problematic rows instead of crashing
            logger.warning(f"Error processing row {idx}: {e}")
            continue
    return {
        "query": query,
        "normalized_query": norm_query,
        "k": k,
        "alpha": alpha,
        "semantic_enabled": semantic_enabled and SentenceTransformer is not None,
        "results": results,
    }


