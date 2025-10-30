"""Clinical-Longformer wrapper with lightweight defaults."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from transformers import LongformerForQuestionAnswering, LongformerTokenizer

from .base import BaseMedicalModel
from config import ensure_allowed_model_name, get_model_cache_dir, get_timeout_seconds, with_cache_dir


FALLBACK_QUESTIONS = [
    "What are the main symptoms?",
    "What is the diagnosis?",
    "What medications were prescribed?",
    "What are the treatment recommendations?",
    "Are there any contraindications?",
    "What is the follow-up plan?",
]


def _extract_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for piece in text.replace("\n", " ").split('. '):
        clean = piece.strip()
        if clean:
            sentences.append(clean.rstrip('.'))
    return sentences


class ClinicalLongformerModel(BaseMedicalModel):
    """Handles long clinical records with optional heavy model loading."""

    def __init__(self) -> None:
        requested_name = os.getenv("LONGFORMER_MODEL_NAME", "yikuan8/Clinical-Longformer")
        self.model_name = ensure_allowed_model_name("longformer", requested_name)
        device_override = os.getenv("LONGFORMER_DEVICE")
        super().__init__(
            "longformer",
            device=device_override or "cpu",
            max_length=int(os.getenv("LONGFORMER_MAX_LENGTH", "4096")),
        )
        self._timeout_seconds = get_timeout_seconds("longformer", default=45)

    def _load_resources(self) -> None:
        local_only = os.getenv("LONGFORMER_LOCAL_ONLY", "0") == "1"
        load_kwargs = with_cache_dir({"local_files_only": local_only})

        cache_dir = get_model_cache_dir()
        if cache_dir:
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        self._tokenizer = LongformerTokenizer.from_pretrained(self.model_name, **load_kwargs)
        self._model = LongformerForQuestionAnswering.from_pretrained(
            self.model_name,
            **load_kwargs,
        ).to(self.device)

    def answer_question(self, context: str, question: str, **_: Any) -> Dict[str, Any]:
        if self.use_fallback():
            self.require_inference_disabled()
            return self.build_response(
                content=self._fallback_answer(context, question),
                metadata={"question": question, "mode": "fallback"},
                warnings=[
                    "Clinical-Longformer running in fallback mode; set LONGFORMER_MODE=inference to enable QA model."
                ],
            )

        if not self.ensure_model_loaded():
            self.require_inference_disabled()
            return self.build_response(
                content=self._fallback_answer(context, question),
                metadata={"question": question, "mode": "fallback"},
                warnings=["Model load failed; using fallback"],
            )

        inputs = self._tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self._model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        answer_tokens = inputs.input_ids[0][answer_start : answer_end + 1]
        answer_text = self._tokenizer.decode(answer_tokens).strip()

        start_scores = torch.softmax(outputs.start_logits, dim=1)
        end_scores = torch.softmax(outputs.end_logits, dim=1)
        confidence = float((start_scores.max() + end_scores.max()) / 2)

        return self.build_response(
            content={
                "answer": answer_text,
                "confidence": confidence,
            },
            metadata={
                "model_name": self.model_name,
                "answer_start": int(answer_start),
                "answer_end": int(answer_end),
            },
        )

    def analyze_clinical_record(self, record: str, questions: List[str]) -> Dict[str, Any]:
        responses = [
            {
                "question": q,
                "result": self.answer_question(record, q),
            }
            for q in questions
        ]
        return self.build_response(
            content={
                "record_analysis": responses,
            },
            metadata={
                "record_length": len(record),
                "questions_analyzed": len(questions),
            },
        )

    def extract_clinical_insights(self, record: str) -> Dict[str, Any]:
        return self.analyze_clinical_record(record, FALLBACK_QUESTIONS)

    def _fallback_answer(self, context: str, question: str) -> Dict[str, Any]:
        sentences = _extract_sentences(context)
        lower_question = question.lower()

        def find_snippets(keywords: List[str], limit: int = 2) -> List[str]:
            matched: List[str] = []
            for sentence in sentences:
                lower_sentence = sentence.lower()
                if all(keyword in lower_sentence for keyword in keywords):
                    matched.append(sentence)
                if len(matched) >= limit:
                    break
            return matched

        if "symptom" in lower_question:
            return {"answer": find_snippets(["symptom"]), "confidence": 0.1}
        if "diagnosis" in lower_question:
            return {"answer": find_snippets(["diagnosis"]), "confidence": 0.1}
        if "medication" in lower_question or "drug" in lower_question:
            return {"answer": find_snippets(["medication"]), "confidence": 0.1}
        if "follow-up" in lower_question or "follow up" in lower_question:
            return {"answer": find_snippets(["follow"], limit=1), "confidence": 0.1}
        if "treatment" in lower_question or "recommend" in lower_question:
            return {"answer": find_snippets(["treat"], limit=2), "confidence": 0.1}

        snippet = sentences[0] if sentences else "No clinical details available."
        return {"answer": snippet, "confidence": 0.05}

    def get_model_info(self) -> Dict[str, Any]:  # type: ignore[override]
        info = super().get_model_info()
        info.update(
            {
                "model_name": self.model_name,
                "timeout_seconds": self._timeout_seconds,
                "question_count": len(FALLBACK_QUESTIONS),
            }
        )
        return info