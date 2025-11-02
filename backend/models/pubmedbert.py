"""PubMedBERT wrapper with fallback keyword-based extraction."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .base import BaseMedicalModel
from ..config import ensure_allowed_model_name, get_model_cache_dir, with_cache_dir

ENTITY_TYPES = [
    "DISEASE",
    "DRUG",
    "SYMPTOM",
    "TREATMENT",
    "TEST",
    "ANATOMY",
    "PROCEDURE",
    "DOSAGE",
    "OTHER",
]

KEYWORD_LIBRARY: Dict[str, List[str]] = {
    "SYMPTOM": ["pain", "fever", "nausea", "fatigue", "cough", "headache"],
    "DISEASE": ["diabetes", "hypertension", "asthma", "covid", "flu"],
    "DRUG": ["aspirin", "ibuprofen", "insulin", "metformin", "amoxicillin"],
    "TREATMENT": ["therapy", "surgery", "rehabilitation", "physiotherapy"],
    "TEST": ["mri", "ct", "x-ray", "blood test", "cbc"],
    "ANATOMY": ["heart", "lung", "liver", "kidney", "brain"],
    "PROCEDURE": ["biopsy", "injection", "transplant", "intubation"],
    "DOSAGE": ["mg", "ml", "dose", "tablet"],
}


class PubMedBERTModel(BaseMedicalModel):
    """Biomedical entity extractor with optional heavy model loading."""

    def __init__(self) -> None:
        requested_name = os.getenv(
            "PUBMEDBERT_MODEL_NAME",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        )
        self.model_name = ensure_allowed_model_name("pubmedbert", requested_name)
        super().__init__(
            "pubmedbert",
            max_length=int(os.getenv("PUBMEDBERT_MAX_LENGTH", "512")),
        )

    def _load_resources(self) -> None:
        local_only = os.getenv("PUBMEDBERT_LOCAL_ONLY", "0") == "1"
        load_kwargs = with_cache_dir({"local_files_only": local_only})

        cache_dir = get_model_cache_dir()
        if cache_dir:
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(ENTITY_TYPES),
            **load_kwargs,
        ).to(self.device)

    def extract_entities(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        if self.use_fallback():
            self.require_inference_disabled()
            entities = self._fallback_entities(text)
            return self.build_response(
                content={"entities": entities},
                metadata={"threshold": threshold, "text_length": len(text)},
                warnings=[
                    "PubMedBERT running in fallback mode; set PUBMEDBERT_MODE=inference to enable model loading."
                ],
            )

        if not self.ensure_model_loaded():
            self.require_inference_disabled()
            entities = self._fallback_entities(text)
            return self.build_response(
                content={"entities": entities},
                metadata={"threshold": threshold, "text_length": len(text)},
                warnings=["Model load failed; using fallback"],
            )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        outputs = self._model(**inputs)
        predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        tokens = self._tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        entities: Dict[str, List[str]] = {etype: [] for etype in ENTITY_TYPES}
        current = {"type": None, "text": []}

        for token, pred in zip(tokens, predictions[0]):
            if token.startswith("##"):
                if current["type"]:
                    current["text"].append(token[2:])
                continue

            max_pred = float(np.max(pred))
            if max_pred > threshold:
                entity_type = ENTITY_TYPES[int(np.argmax(pred))]
                if current["type"] == entity_type:
                    current["text"].append(token)
                else:
                    if current["type"]:
                        entities[current["type"]].append("".join(current["text"]))
                    current = {"type": entity_type, "text": [token]}
            else:
                if current["type"]:
                    entities[current["type"]].append("".join(current["text"]))
                    current = {"type": None, "text": []}

        if current["type"]:
            entities[current["type"]].append("".join(current["text"]))

        for etype in entities:
            entities[etype] = [val.strip() for val in entities[etype] if val.strip()]

        return self.build_response(
            content={"entities": entities},
            metadata={
                "model_name": self.model_name,
                "threshold": threshold,
                "text_length": len(text),
            },
        )

    def classify_medical_text(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        response = self.extract_entities(text, threshold=threshold)
        entities = response.get("content", {}).get("entities", {})
        entity_counts = {etype: len(entities.get(etype, [])) for etype in ENTITY_TYPES}
        primary = max(entity_counts, key=entity_counts.get) if any(entity_counts.values()) else "UNKNOWN"

        response.setdefault("content", {})["classification"] = {
            "primary_category": primary,
            "entity_distribution": entity_counts,
        }
        return response

    def batch_process(self, texts: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        return [self.extract_entities(text, threshold=threshold) for text in texts]

    def _fallback_entities(self, text: str) -> Dict[str, List[str]]:
        normalized = text.lower()
        entities = {etype: [] for etype in ENTITY_TYPES}
        for etype, keywords in KEYWORD_LIBRARY.items():
            hits = sorted({kw for kw in keywords if kw in normalized})
            if hits:
                entities[etype] = hits
        medication_matches = re.findall(r"\b\d+\s?(mg|ml|mcg)\b", normalized)
        if medication_matches:
            entities["DOSAGE"].extend(sorted(set(medication_matches)))
        return entities

    def get_model_info(self) -> Dict[str, Any]:  # type: ignore[override]
        info = super().get_model_info()
        info.update(
            {
                "model_name": self.model_name,
                "entity_types": ENTITY_TYPES,
            }
        )
        return info