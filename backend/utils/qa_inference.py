"""BERT QA inference utilities for extractive question answering.

Loads the fine-tuned BERT model and runs span prediction on context passages.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BASE_DIR / "bert-medqa-custom"

_model = None
_tokenizer = None
_device = None
_model_dir = None


def _clean_context(text: str) -> str:
    """Remove question-like prefixes and generic intro sentences from answer text.
    
    MedQuAD answers often start with:
    1. Restated questions: "What are the symptoms of X? Symptoms include..."
    2. Generic statements: "X is a disease that...", "Having X can be scary..."
    
    This function removes such prefixes to help BERT focus on actual answers.
    """
    if not text:
        return text
    
    # Only remove the first sentence if it's clearly a question or too generic
    # Don't remove all the informative content!
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if not sentences:
        return text
    
    first_sent = sentences[0].strip()
    
    # Patterns that indicate a useless first sentence
    skip_first = (
        # Direct questions
        re.match(r'^\s*(what|how|when|where|why|who|which)\b.+\?', first_sent, re.IGNORECASE)
        # Pure definitions without specifics: "X is a disease"
        or re.match(r'^\s*\w+\s+is\s+(a|an)\s+(disease|disorder|condition)\s*\.?\s*$', first_sent, re.IGNORECASE)
        # Emotional fluff: "Having X can be scary"
        or re.match(r'^\s*having\s+.+\s+(can|may)\s+be\s+(scary|frightening|overwhelming)', first_sent, re.IGNORECASE)
    )
    
    if skip_first and len(sentences) > 1:
        # Remove first sentence, keep the rest
        text = ' '.join(sentences[1:])
    
    return text.strip()


def load_model(model_dir: Optional[Path] = None) -> None:
    """Load BERT QA model and tokenizer into memory."""
    global _model, _tokenizer, _device, _model_dir
    
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    if _model is not None and _model_dir == model_dir:
        return  # already loaded
    
    logger.info(f"Loading QA model from {model_dir}")
    print(f"[QA] Loading BERT QA model from {model_dir}...", flush=True)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"[QA] Tokenizer loaded, loading model weights...", flush=True)
    _model = AutoModelForQuestionAnswering.from_pretrained(str(model_dir))
    _model.to(_device)
    _model.eval()
    _model_dir = model_dir
    logger.info(f"Model loaded on {_device}")
    print(f"[QA] Model ready on {_device}", flush=True)


def predict_answer(
    question: str,
    context: str,
    max_answer_length: int = 100,
    top_k: int = 1,
    timeout: float = 10.0,  # Timeout for QA inference
) -> List[Dict[str, Any]]:
    """Extract answer span(s) from context using fine-tuned BERT QA.
    
    Args:
        question: User question
        context: Passage/document to extract answer from
        max_answer_length: Maximum tokens in answer span
        top_k: Number of candidate answers to return
    
    Returns:
        List of dicts with keys: answer, start_logit, end_logit, confidence, start_idx, end_idx
    """
    if _model is None or _tokenizer is None:
        load_model()
    
    assert _model is not None and _tokenizer is not None
    
    # Clean context to remove question-like prefixes
    context = _clean_context(context)
    
    # Truncate context early to prevent slow processing
    max_context_chars = 800  # Limit context length
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    
    # Tokenize with truncation
    inputs = _tokenizer(
        question,
        context,
        max_length=384,  # Reduced from 512 for faster processing
        truncation="only_second",  # truncate context if too long
        padding="max_length",
        return_tensors="pt",
    )
    
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    # Run inference with error handling
    try:
        with torch.no_grad():
            outputs = _model(**inputs)
    except Exception as e:
        logger.error(f"QA inference error: {e}")
        return []  # Return empty on error
    
    start_logits = outputs.start_logits[0]  # shape: (seq_len,)
    end_logits = outputs.end_logits[0]
    
    # Find top-k answer spans (optimized - only check top candidates)
    candidates: List[Tuple[int, int, float, float, float]] = []  # (start, end, start_logit, end_logit, score)
    
    # Get top candidate start/end positions
    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)
    
    # Get top start and end positions (much faster than checking all combinations)
    seq_len = len(start_logits)
    # Only check top 20 start positions and top 20 end positions
    top_start_count = min(20, seq_len)
    top_end_count = min(20, seq_len)
    
    # Get top start positions
    top_start_scores, top_start_indices = torch.topk(start_logits, top_start_count)
    top_end_scores, top_end_indices = torch.topk(end_logits, top_end_count)
    
    # Find valid spans from top candidates only
    for start_idx in top_start_indices.cpu().tolist():
        for end_idx in top_end_indices.cpu().tolist():
            # Skip invalid spans
            if start_idx >= end_idx:
                continue
            if start_idx == 0 or end_idx == 0:  # Skip special tokens
                continue
            if end_idx - start_idx > max_answer_length:
                continue
            
            # Combined score
            score = float(start_logits[start_idx] + end_logits[end_idx])
            
            # Decode to check answer quality
            input_ids = inputs["input_ids"][0]
            answer_tokens = input_ids[start_idx : end_idx + 1]
            answer_text = _tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Skip very short answers (less than 2 words)
            if len(answer_text.split()) < 2:
                continue
            
            candidates.append((
                int(start_idx),
                int(end_idx),
                float(start_logits[start_idx]),
                float(end_logits[end_idx]),
                score,
            ))
    
    # Sort by combined score (limit to reasonable number)
    candidates.sort(key=lambda x: x[4], reverse=True)
    candidates = candidates[:top_k * 3]  # Keep top 3x candidates for final selection
    
    results: List[Dict[str, Any]] = []
    for start_idx, end_idx, start_logit, end_logit, score in candidates[:top_k]:
        # Decode answer span
        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_idx : end_idx + 1]
        answer_text = _tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Compute confidence (normalized probability)
        confidence = float(
            torch.softmax(start_logits, dim=0)[start_idx]
            * torch.softmax(end_logits, dim=0)[end_idx]
        )
        
        results.append({
            "answer": answer_text.strip(),
            "start_logit": start_logit,
            "end_logit": end_logit,
            "confidence": confidence,
            "score": score,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })
    
    return results


def batch_predict(
    question: str,
    contexts: List[str],
    max_answer_length: int = 100,
    max_contexts: int = 5,  # Limit number of contexts to process
) -> List[Dict[str, Any]]:
    """Run QA prediction on multiple contexts and return best answer.
    
    Args:
        question: User question
        contexts: List of passages to search for answer
        max_answer_length: Maximum tokens in answer span
        max_contexts: Maximum number of contexts to process (for performance)
    
    Returns:
        List of results (one per context) sorted by confidence, each with:
        - answer: extracted text
        - confidence: probability score
        - context: source passage
        - context_idx: index in input list
    """
    if not contexts:
        return []
    
    # Limit contexts for faster processing
    contexts_to_process = contexts[:max_contexts]
    
    results: List[Dict[str, Any]] = []
    
    for idx, context in enumerate(contexts_to_process):
        if not context or not context.strip():
            continue
        
        try:
            predictions = predict_answer(
                question=question,
                context=context,
                max_answer_length=max_answer_length,
                top_k=1,
            )
            
            if predictions:
                best = predictions[0]
                results.append({
                    "answer": best["answer"],
                    "confidence": best["confidence"],
                    "score": best["score"],
                    "start_logit": best["start_logit"],
                    "end_logit": best["end_logit"],
                    "context": context,
                    "context_idx": idx,
                })
        except Exception as e:
            # Skip contexts that cause errors
            logger.warning(f"Error processing context {idx}: {e}")
            continue
    
    # Sort by confidence (or score)
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    return results


def get_model_info() -> Dict[str, Any]:
    """Return information about loaded model."""
    return {
        "loaded": _model is not None,
        "device": str(_device) if _device else None,
        "model_dir": str(_model_dir) if _model_dir else None,
    }


