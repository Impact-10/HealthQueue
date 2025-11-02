from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.models import MedAlpacaModel

app = FastAPI(title="HealthQueue Medical AI API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models lazily
models = {}

# Function to get or initialize model
def get_model(model_name: str):
    # Only MedAlpaca and Gemini supported
    if model_name not in models:
        if model_name == "medalpaca":
            from backend.models import MedAlpacaModel
            models[model_name] = MedAlpacaModel()
        elif model_name == "biogpt":
            from backend.models import BioGPTModel
            models[model_name] = BioGPTModel()
        elif model_name == "gemini":
            class GeminiModel:
                def generate_response(self, prompt, **kwargs):
                    return {"summary": "Gemini API integration not implemented in backend yet."}
                def get_model_info(self):
                    return {"gemini": True}
            models[model_name] = GeminiModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]


def simple_payload(
    *,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    model_key: str,
    structured: Dict[str, Any],
) -> Dict[str, Any]:
    mode = structured.get("mode")
    metadata = structured.get("metadata", {})
    model_info = {
        "key": model_key,
        "name": metadata.get("model_name") or model_key,
        "mode": mode,
        "badge": metadata.get("model_name") or model_key,
    }
    payload: Dict[str, Any] = {
        "model": model_info,
        "mode": mode,
        "structuredResponse": structured,
    }
    if question is not None:
        payload["question"] = question
    if answer is not None:
        payload["answer"] = answer
    return payload


def extract_primary_answer(structured: Dict[str, Any]) -> str:
    content = structured.get("content")
    if not isinstance(content, dict):
        return ""

    prioritized_keys = [
        "summary",
        "answer",
        "text",
        "description",
    ]
    for key in prioritized_keys:
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    list_keys = [
        "possible_causes",
        "possible_diagnoses",
        "recommendations",
        "follow_up_questions",
        "warning_signs",
    ]
    for key in list_keys:
        value = content.get(key)
        if isinstance(value, list) and value:
            return ", ".join(str(item) for item in value[:3])

    if "entities" in content and isinstance(content["entities"], dict):
        entities = content["entities"]
        parts: List[str] = []
        for etype, values in entities.items():
            if isinstance(values, list) and values:
                parts.append(f"{etype}: {', '.join(str(v) for v in values[:3])}")
        if parts:
            return " | ".join(parts)

    if "record_analysis" in content and isinstance(content["record_analysis"], list):
        entries = content["record_analysis"]
        if entries:
            first = entries[0]
            if isinstance(first, dict):
                question = first.get("question")
                result = first.get("result")
                if isinstance(result, dict):
                    answer = result.get("content")
                    if isinstance(answer, dict):
                        text = answer.get("answer")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
                if isinstance(question, str) and question.strip():
                    return question.strip()

    return ""

# Request/Response Models
class DiagnosisRequest(BaseModel):
    symptoms: str
    medical_history: Optional[str] = None
    model_name: str = "ensemble"
    additional_context: Optional[Dict[str, Any]] = None


class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None


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

# API Routes
@app.get("/")
async def root():
    """API root endpoint"""
    available_models = [
        "medalpaca",
        "biogpt",
        "longformer", 
        "pubmedbert",
        "ensemble"
    ]
    return {
        "name": "HealthQueue Medical AI API",
        "version": "1.0",
        "models": {
            name: "initialized" if name in models else "available"
            for name in available_models
        },
        "status": "operational"
    }


@app.get("/health")
async def healthcheck():
    """Lightweight health endpoint for monitoring."""
    response = {"status": "ok", "models": {}}
    for name in ["medalpaca", "biogpt", "longformer", "pubmedbert", "ensemble"]:
        model = get_model(name)
        info = model.get_model_info()
        response["models"][name] = {
            "mode": info.get("mode"),
            "initialized": info.get("is_initialized"),
        }
    return response


@app.post("/api/medalpaca")
async def medalpaca_endpoint(request: QuestionRequest):
    """Direct question/answer endpoint backed by MedAlpaca-7B."""

    model = get_model("medalpaca")
    kwargs: Dict[str, Any] = {}
    if request.max_length is not None:
        kwargs["max_length"] = request.max_length
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    try:
        structured = model.generate_response(request.question, **kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = extract_primary_answer(structured)
    return simple_payload(
        question=request.question,
        answer=answer,
        model_key="medalpaca",
        structured=structured,
    )


@app.post("/api/biogpt")
async def biogpt_endpoint(request: QuestionRequest):
    """Biomedical differential diagnosis endpoint backed by BioGPT."""

    model = get_model("biogpt")
    kwargs: Dict[str, Any] = {}
    if request.max_length is not None:
        kwargs["max_length"] = request.max_length
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    try:
        structured = model.generate_response(
            request.question,
            **kwargs,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = extract_primary_answer(structured)
    return simple_payload(
        question=request.question,
        answer=answer,
        model_key="biogpt",
        structured=structured,
    )


@app.post("/api/clinical_longformer")
async def clinical_longformer_endpoint(request: ClinicalQARequest):
    """Clinical QA endpoint backed by Clinical-Longformer."""

    model = get_model("longformer")
    try:
        if request.questions:
            structured = model.analyze_clinical_record(request.context, request.questions)
        else:
            structured = model.answer_question(request.context, request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = extract_primary_answer(structured)
    return simple_payload(
        question=request.question,
        answer=answer,
        model_key="longformer",
        structured=structured,
    )


@app.post("/api/pubmedbert")
async def pubmedbert_endpoint(request: EntitiesRequest):
    """Biomedical entity extraction endpoint backed by PubMedBERT."""

    model = get_model("pubmedbert")
    threshold = request.threshold if request.threshold is not None else 0.5
    try:
        structured = model.extract_entities(request.text, threshold=threshold)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = extract_primary_answer(structured)
    response = simple_payload(
        question=request.text,
        answer=answer,
        model_key="pubmedbert",
        structured=structured,
    )
    response["entities"] = structured.get("content", {}).get("entities", {})
    response["threshold"] = threshold
    return response

@app.post("/api/diagnosis")
async def get_diagnosis(request: DiagnosisRequest):
    """Get medical diagnosis from specified model"""
    try:
        model = get_model(request.model_name)
        
        if request.model_name == "ensemble":
            result = model.get_ensemble_diagnosis(
                request.symptoms,
                request.medical_history,
                **(request.additional_context or {})
            )
        elif request.model_name == "biogpt":
            result = model.generate_diagnosis(
                request.symptoms,
                request.medical_history,
                **(request.additional_context or {})
            )
        else:
            result = model.generate_response(
                f"Symptoms: {request.symptoms}\n" +
                f"Medical History: {request.medical_history or 'None'}",
                **(request.additional_context or {})
            )

        return result

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/entities")
async def extract_entities(request: EntityExtractionRequest):
    """Extract medical entities from text"""
    try:
        if request.model_name not in ["pubmedbert", "ensemble"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model for entity extraction"
            )

        model = get_model(request.model_name)
        if request.model_name == "pubmedbert":
            result = model.extract_entities(request.text)
        else:
            result = model.models["pubmedbert"].extract_entities(request.text)

        return result

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/analyze")
async def analyze_clinical_record(request: AnalysisRequest):
    """Analyze clinical records"""
    try:
        if request.model_name not in ["longformer", "ensemble"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model for clinical analysis"
            )

        model = get_model(request.model_name)

        if request.model_name == "longformer":
            if request.questions:
                result = model.analyze_clinical_record(
                    request.clinical_record,
                    request.questions,
                )
            else:
                result = model.extract_clinical_insights(request.clinical_record)
        else:
            result = model.models["longformer"].extract_clinical_insights(
                request.clinical_record
            )

        return result

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/population-health")
async def analyze_population_health(request: PopulationHealthRequest):
    """Analyze population health trends"""
    try:
        ensemble = get_model("ensemble")
        result = ensemble.analyze_population_health(request.diagnoses)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-info/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    try:
        model = get_model(model_name)
        return model.get_model_info()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)