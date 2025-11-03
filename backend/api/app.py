from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from models import MedAlpacaModel, BioGPTModel

app = FastAPI(title="HealthQueue Medical AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models: Dict[str, Any] = {}

def get_model(model_name: str):
    if model_name not in models:
        if model_name == "medalpaca":
            models[model_name] = MedAlpacaModel()
        elif model_name == "biogpt":
            models[model_name] = BioGPTModel()
        elif model_name == "gemini":
            class GeminiModel:
                def generate_response(self, prompt: str, **kwargs):
                    return {"summary": "Gemini API integration not implemented in backend yet."}
                def get_model_info(self):
                    return {"gemini": True, "is_initialized": True, "mode": "inference"}
            models[model_name] = GeminiModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]

def simple_payload(*, question: Optional[str]=None, answer: Optional[str]=None, model_key: str, structured: Dict[str, Any]) -> Dict[str, Any]:
    mode = structured.get("mode")
    metadata = structured.get("metadata", {})
    model_info = {"key": model_key, "name": metadata.get("model_name") or model_key, "mode": mode, "badge": metadata.get("model_name") or model_key}
    payload: Dict[str, Any] = {"model": model_info, "mode": mode, "structuredResponse": structured}
    if question is not None: payload["question"] = question
    if answer is not None: payload["answer"] = answer
    return payload

def extract_primary_answer(structured: Dict[str, Any]) -> str:
    content = structured.get("content")
    if not isinstance(content, dict): return ""
    for k in ["summary","answer","text","description"]:
        v = content.get(k)
        if isinstance(v, str) and v.strip(): return v.strip()
    for k in ["possible_causes","possible_diagnoses","recommendations","follow_up_questions","warning_signs"]:
        v = content.get(k)
        if isinstance(v, list) and v: return ", ".join(str(it) for it in v[:3])
    if "entities" in content and isinstance(content["entities"], dict):
        parts = []
        for etype, values in content["entities"].items():
            if isinstance(values, list) and values:
                parts.append(f"{etype}: {', '.join(str(v) for v in values[:3])}")
        if parts: return " | ".join(parts)
    return ""

class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None

class DiagnosisRequest(BaseModel):
    symptoms: str
    medical_history: Optional[str] = None
    model_name: str = "ensemble"
    additional_context: Optional[Dict[str, Any]] = None

class ClinicalQARequest(BaseModel):
    question: str
    context: str
    questions: Optional[List[str]] = None

class EntitiesRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class EntityExtractionRequest(BaseModel):
    text: str
    model_name: str = "pubmedbert"

class AnalysisRequest(BaseModel):
    clinical_record: str
    questions: Optional[List[str]] = None
    model_name: str = "longformer"

class PopulationHealthRequest(BaseModel):
    diagnoses: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"name":"HealthQueue Medical AI API","version":"1.0","models":{k:("initialized" if k in models else "available") for k in ["medalpaca","biogpt","gemini"]},"status":"operational"}

@app.get("/health")
async def healthcheck():
    out = {"status":"ok","models":{}}
    for name in ["medalpaca","biogpt","gemini"]:
        try:
            info = get_model(name).get_model_info()
            out["models"][name] = {"mode": info.get("mode","inference"), "initialized": info.get("is_initialized", True)}
        except Exception as e:
            out["models"][name] = {"error": str(e), "initialized": False}
    return out

@app.post("/api/medalpaca")
async def medalpaca_endpoint(request: QuestionRequest):
    model = get_model("medalpaca")
    kwargs: Dict[str, Any] = {}
    if request.max_length is not None: kwargs["max_length"] = request.max_length
    if request.temperature is not None: kwargs["temperature"] = request.temperature
    try:
        structured = model.generate_response(request.question, **kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    answer = extract_primary_answer(structured)
    return simple_payload(question=request.question, answer=answer, model_key="medalpaca", structured=structured)

@app.post("/api/biogpt")
async def biogpt_endpoint(request: QuestionRequest):
    model = get_model("biogpt")
    kwargs: Dict[str, Any] = {}
    if request.max_length is not None: kwargs["max_length"] = request.max_length
    if request.temperature is not None: kwargs["temperature"] = request.temperature
    try:
        structured = model.generate_response(request.question, **kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    answer = extract_primary_answer(structured)
    return simple_payload(question=request.question, answer=answer, model_key="biogpt", structured=structured)

# Optional additional routes preserved from your original app if needed...

if __name__ == "__main__":
    import uvicorn
    # BIOGPT_MODEL_NAME can be set to an absolute path if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
