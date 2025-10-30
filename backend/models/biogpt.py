"""BioGPT lightweight wrapper with fallback-first behaviour."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from .medalpaca import MedAlpacaModel

from .base import BaseMedicalModel
from config import ensure_allowed_model_name, get_model_cache_dir, with_cache_dir


DIAGNOSIS_PROMPT = (
    "You are a biomedical AI assistant specialised in differential diagnosis.\n"
    "Return a JSON object with keys: summary, possible_diagnoses (array of strings),\n"
    "rationale (array of strings), recommended_tests (array of strings),\n"
    "next_steps (array of strings), warning_signs (array of strings).\n"
    "Do not include extra commentary.\n\n"
    "Patient symptoms and history:\n{symptoms}\n\n"
    "Provide evidence-based reasoning and stay concise."
)

FALLBACK_LIBRARY: List[Dict[str, Any]] = [
    {
        "keywords": {"chest", "pain"},
        "content": {
            "summary": "Chest pain evaluation requires urgent risk stratification.",
            "possible_diagnoses": [
                "Acute coronary syndrome",
                "Gastroesophageal reflux",
                "Musculoskeletal strain",
            ],
            "rationale": [
                "Cardiac causes must be excluded rapidly",
                "Reflux can mimic cardiac discomfort",
                "Chest wall strain common with movement-related pain",
            ],
            "recommended_tests": [
                "Electrocardiogram (ECG)",
                "Cardiac enzymes (troponin)",
                "Chest X-ray if respiratory symptoms",
            ],
            "next_steps": [
                "Seek emergency care if pain is severe or accompanied by dyspnea",
                "Avoid exertion until evaluated",
                "Track triggers, duration, associated symptoms",
            ],
            "warning_signs": [
                "Crushing chest pain lasting >15 minutes",
                "Radiation to arm/jaw, diaphoresis",
                "Shortness of breath, syncope, palpitations",
            ],
        },
    }
]

DEFAULT_FALLBACK: Dict[str, Any] = {
    "summary": "General biomedical assessment guidance.",
    "possible_diagnoses": ["Additional clinical detail required"],
    "rationale": ["Insufficient structured information provided"],
    "recommended_tests": [
        "Collect vital signs",
        "Review recent lab/imaging data",
    ],
    "next_steps": [
        "Document symptom onset, severity, and modifiers",
        "Consult an appropriate clinical specialist",
    ],
    "warning_signs": [
        "Rapid deterioration, severe pain, or neurological deficits",
        "Signs of sepsis (fever, tachycardia, hypotension)",
    ],
}


class BioGPTModel(BaseMedicalModel):
    """Biomedical text generation with configurable inference mode."""

    def __init__(self) -> None:


        local_only = os.getenv("BIOGPT_LOCAL_ONLY", "0") == "1"
        load_kwargs = with_cache_dir({
            "trust_remote_code": True,
            "local_files_only": local_only,
        })

        cache_dir = get_model_cache_dir()
        if cache_dir:



            # Use MedAlpacaModel for all BioGPT requests
            BioGPTModel = MedAlpacaModel