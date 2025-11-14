from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for relative imports
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.retriever import search as medquad_search, hybrid_search
from utils import qa_inference
from utils.safety_filter import check_safety

app = FastAPI(title="HealthQueue Medical AI API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BERT QA model is initialized lazily in qa_inference module

def _make_conversational(question: str, answer: str) -> str:
    """Wrap extracted answer span in natural conversational template.
    
    Makes raw BERT extractions feel more like a doctor's response.
    """
    q_lower = question.lower().strip()
    
    # Detect question type and choose appropriate intro
    if any(word in q_lower for word in ["what is", "what are", "what's"]):
        if answer[0].isupper():  # Already starts with proper noun/sentence
            return answer
        else:
            return f"That refers to {answer}"
    
    elif any(word in q_lower for word in ["how to", "how do", "how can"]):
        if "treat" in q_lower or "cure" in q_lower:
            return f"Treatment typically involves: {answer}"
        else:
            return f"Here's how: {answer}"
    
    elif any(word in q_lower for word in ["what causes", "why does", "what leads to"]):
        return f"The main cause is: {answer}"
    
    elif any(word in q_lower for word in ["symptom", "sign", "feel"]):
        return f"Common symptoms include: {answer}"
    
    elif "when" in q_lower or "how long" in q_lower:
        return f"Typically, {answer}"
    
    else:
        # Generic: just clean up the answer
        return answer if answer[0].isupper() else answer.capitalize()

# BERT QA model is accessed via qa_inference module


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class QARequest(BaseModel):
    question: str
    k: Optional[int] = 10  # Retrieve more passages for better results (was 5)
    alpha: Optional[float] = 0.6  # Weight semantic search higher (was 0.5)
    confidence_threshold: Optional[float] = 0.1  # Minimum confidence to return answer
    max_answer_length: Optional[int] = 150  # Longer answers (was 100)

# API Routes
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "HealthQueue Medical AI API",
        "version": "1.0",
        "models": {
            "bert_qa": "available"
        },
        "status": "operational"
    }


@app.get("/health")
async def healthcheck():
    """Lightweight health endpoint for monitoring."""
    response = {"status": "ok", "models": {}}
    
    # Check BERT QA model
    try:
        qa_info = qa_inference.get_model_info()
        response["models"]["bert_qa"] = qa_info
    except Exception as e:
        response["models"]["bert_qa"] = {"error": str(e)}
    
    return response


@app.post("/api/medquad/search")
async def medquad_search_endpoint(req: SearchRequest):
    """Semantic keyword-based search over MedQuAD Q&A (TF-IDF cosine)."""
    try:
        result = medquad_search(req.query, k=req.k or 5)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa")
async def qa_endpoint(req: QARequest):
    """Complete QA pipeline: retrieval → BERT QA → ranked answers.
    
    Flow:
    1. Safety check: filter unsafe/inappropriate queries
    2. Retrieve top-k passages using hybrid search (lexical + semantic)
    3. Run fine-tuned BERT QA on each passage to extract answer spans
    4. Rank by confidence score
    5. Return best answer + source passage + metadata
    6. If confidence below threshold, return retrieved passages instead
    """
    try:
        # Step 0: Safety check (BEFORE any expensive operations)
        is_safe, safety_reason, severity = check_safety(req.question)
        if not is_safe:
            # Return immediately - don't do retrieval/QA
            return {
                "question": req.question,
                "answer": None,
                "confidence": 0.0,
                "mode": "blocked",
                "message": safety_reason or "Query blocked by safety filter",
                "severity": severity,
            }
        
        # Add warning to response if needed
        safety_warning = safety_reason if severity == "warning" else None
        
        # Step 1: Retrieve relevant passages (only if query passed safety check)
        # Use smaller k for faster retrieval (can increase if needed)
        retrieval_k = min(req.k or 5, 10)  # Cap at 10 for performance
        # Disable semantic search by default for speed - lexical-only is fast and reliable
        retrieval_result = hybrid_search(
            query=req.question,
            k=retrieval_k,
            alpha=req.alpha or 0.5,
            timeout=2.0,
            use_semantic=False,  # Disable semantic for now - too slow, lexical works fine
        )
        
        if not retrieval_result.get("results"):
            return {
                "question": req.question,
                "answer": None,
                "confidence": 0.0,
                "mode": "no_results",
                "message": "No relevant passages found in corpus",
                "retrieval": retrieval_result,
            }
        
        # Extract passage texts (use ONLY 'answer' field from retrieved Q&A pairs)
        # IMPORTANT: Don't include the retrieved question, as BERT will extract it instead of the answer
        contexts = []
        for r in retrieval_result["results"]:
            # Use only the answer as context for QA extraction
            answer_text = r.get("answer", "").strip()
            if answer_text:
                contexts.append(answer_text)
            else:
                # Fallback: if no answer, use full text
                ctx_parts = []
                if r.get("question"):
                    ctx_parts.append(r["question"])
                contexts.append(" ".join(ctx_parts) if ctx_parts else "")
        
        # Step 2: Run BERT QA on each passage (limit to top 3 for performance)
        # Process fewer contexts to avoid timeouts
        contexts_to_process = contexts[:3]  # Only top 3 for speed
        qa_results = qa_inference.batch_predict(
            question=req.question,
            contexts=contexts_to_process,
            max_answer_length=req.max_answer_length or 100,
            max_contexts=3,  # Reduced from 5
        )
        
        if not qa_results:
            return {
                "question": req.question,
                "answer": None,
                "confidence": 0.0,
                "mode": "qa_failed",
                "message": "QA model could not extract answer from passages",
                "retrieval": retrieval_result,
            }
        
        # Step 3: Check confidence threshold
        best_answer = qa_results[0]
        confidence_threshold = req.confidence_threshold or 0.1
        
        if best_answer["confidence"] < confidence_threshold:
            # Low confidence: return retrieved passages as fallback with citations
            top_passages = []
            for idx, r in enumerate(retrieval_result["results"][:5]):  # Top 5 passages
                passage_text = r.get("answer", "") or r.get("question", "")
                if passage_text:
                    top_passages.append({
                        "text": passage_text[:500],  # Truncate for display
                        "source": r.get("label_name") or r.get("source", "unknown"),
                        "rank": r.get("rank", idx + 1),
                        "score": r.get("score", 0.0),
                    })
            
            response = {
                "question": req.question,
                "answer": None,
                "confidence": best_answer["confidence"],
                "mode": "low_confidence",
                "message": f"Answer confidence {best_answer['confidence']:.3f} below threshold {confidence_threshold}. Showing top retrieved passages instead.",
                "top_passages": top_passages,
                "retrieval": retrieval_result,
                "qa_attempts": qa_results[:3],  # show top 3 attempts
            }
            
            # Add safety warning if present
            if safety_warning:
                response["safety_warning"] = safety_warning
            
            return response
        
        # Step 4: Return successful answer with source
        source_idx = best_answer["context_idx"]
        source_passage = retrieval_result["results"][source_idx]
        
        # Wrap answer in conversational template
        raw_answer = best_answer["answer"]
        conversational_answer = _make_conversational(req.question, raw_answer)
        
        response = {
            "question": req.question,
            "answer": conversational_answer,
            "raw_answer": raw_answer,  # Keep original for debugging
            "confidence": best_answer["confidence"],
            "score": best_answer["score"],
            "mode": "success",
            "source": {
                "question": source_passage.get("question"),
                "answer": source_passage.get("answer"),
                "label": source_passage.get("label_name"),
                "retrieval_score": source_passage.get("score"),
                "retrieval_rank": source_passage.get("rank"),
                "source_type": source_passage.get("source", "unknown"),
            },
            "alternative_answers": qa_results[1:3] if len(qa_results) > 1 else [],  # top 2-3 alternatives
            "retrieval": {
                "normalized_query": retrieval_result.get("normalized_query"),
                "semantic_enabled": retrieval_result.get("semantic_enabled"),
                "total_retrieved": len(retrieval_result["results"]),
            },
        }
        
        # Add safety warning if present
        if safety_warning:
            response["safety_warning"] = safety_warning
        
        return response
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"QA pipeline error: {str(e)}\n{traceback.format_exc()}",
        )


@app.get("/api/model-info/bert-qa")
async def get_model_info():
    """Get information about BERT QA model"""
    try:
        return qa_inference.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    