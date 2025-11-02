"""MedAlpaca lightweight wrapper with configurable fallback/inference modes."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseMedicalModel
from ..config import ensure_allowed_model_name, get_model_cache_dir, get_timeout_seconds, with_cache_dir

# Simple symptom heuristics used by the fallback pathway.  Each entry
# contains a set of keywords and the structured response we want to surface.
FALLBACK_LIBRARY: List[Dict[str, Any]] = [
    {
        "keywords": {"headache", "nausea"},
        "content": {
            "summary": "Symptoms of headache accompanied by nausea.",
            "possible_causes": [
                "Migraine",
                "Tension headache",
                "Dehydration",
            ],
            "follow_up_questions": [
                "How intense is the headache on a 0-10 scale?",
                "Do you notice light or sound sensitivity?",
                "Have you experienced recent stress, illness, or injury?",
            ],
            "recommendations": [
                "Rest in a quiet, dark environment",
                "Stay hydrated throughout the day",
                "Consider over-the-counter pain relief if appropriate",
            ],
            "warning_signs": [
                "Sudden or worst-ever headache",
                "Neurological symptoms such as confusion or weakness",
                "High fever or head trauma",
            ],
        },
    }
]

DEFAULT_FALLBACK_RESPONSE: Dict[str, Any] = {
    "summary": "General medical triage guidance.",
    "possible_causes": [
        "Insufficient information for specific causes",
    ],
    "follow_up_questions": [
        "What symptoms are you experiencing and for how long?",
        "Do you have relevant medical history or recent changes?",
    ],
    "recommendations": [
        "Document symptom frequency, severity, and triggers",
        "Maintain hydration and balanced nutrition",
        "Consult a licensed clinician if symptoms persist or worsen",
    ],
    "warning_signs": [
        "Severe pain, high fever, or difficulty breathing",
        "New neurological deficits or sudden onset symptoms",
    ],
}

RESPONSE_TEMPLATE = (
    "You are a helpful, concise medical AI assistant.\n"
    "When a user describes symptoms, always suggest the most likely diagnosis or possible causes based on the information provided.\n"
    "Be clear that this is not a confirmed diagnosis, but your best assessment based on the symptoms.\n"
    "Always remind the user to consult a healthcare professional for confirmation and treatment.\n\n"
    "Answer the following patient question in plain English, providing clear, practical advice.\n\n"
    "Clinical conversation context begins below:\n{context}\n\n"
    "Patient prompt:\n{prompt}\n"
)


class MedAlpacaModel(BaseMedicalModel):
    """MedAlpaca medical triage model with lightweight defaults."""

    def __init__(self) -> None:
        # Use distilgpt2 as the default and only model for minimal RAM usage
        requested_name = os.getenv("MEDALPACA_MODEL_NAME", "distilgpt2")
        self.model_name = requested_name
        super().__init__(
            "medalpaca",
            device="cpu",  # keep CPU-only by default for broad compatibility
            max_length=int(os.getenv("MEDALPACA_MAX_LENGTH", "256")),
        )
        self._conversation_history: List[Dict[str, str]] = []
        self._timeout_seconds = get_timeout_seconds("medalpaca", default=30)
        self._pad_token_id = None

    def _load_resources(self) -> None:
        """Load tokenizer/model when inference mode is enabled."""

        local_only = os.getenv("MEDALPACA_LOCAL_ONLY", "0") == "1"
        load_kwargs = with_cache_dir({
            "local_files_only": local_only,
        })

        cache_dir = get_model_cache_dir()
        if cache_dir:
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.max_length,
            trust_remote_code=True,
            **load_kwargs,
        )
        pad_token_id = self._tokenizer.eos_token_id or self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **load_kwargs,
        ).to(self.device)
        self._pad_token_id = pad_token_id

    def generate_response(
        self,
        prompt: str,
        *,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return a structured medical assessment for the provided prompt."""

        if self.use_fallback():
            self.require_inference_disabled()
            content = self._fallback_assessment(prompt)
            warnings = [
                "Running in fallback mode; enable MEDALPACA_MODE=inference for live model."
            ]
            return self.build_response(
                content=content,
                metadata={"model_name": f"{self.model_name} (demo)"},
                warnings=warnings,
            )

        if not self.ensure_model_loaded():
            self.require_inference_disabled()
            # ensure_model_loaded may switch us back to fallback on failure
            content = self._fallback_assessment(prompt)
            warnings = [
                "Model load failed; using fallback content.",
            ]
            return self.build_response(
                content=content,
                metadata={"model_name": self.model_name, "mode": "fallback"},
                warnings=warnings,
            )

        formatted_prompt = self._build_prompt(prompt)
        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        generation = self._model.generate(
            inputs["input_ids"],
            max_length=max_length or self.max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            num_beams=4,
            repetition_penalty=1.15,
            pad_token_id=getattr(self, "_pad_token_id", self._tokenizer.eos_token_id),
        )
        raw_text = self._tokenizer.decode(generation[0], skip_special_tokens=True)
        content = self._parse_assessment(raw_text)

        self._append_history("user", prompt)
        self._append_history("assistant", json.dumps(content))

        return self.build_response(
            content=content,
            metadata={
                "model_name": self.model_name,
                "temperature": temperature,
                "max_length": max_length or self.max_length,
            },
            extra={
                "conversation": {
                    "message_count": len(self._conversation_history),
                    "recent": self._conversation_history[-6:],
                }
            },
        )

    def _append_history(self, role: str, content: str) -> None:
        self._conversation_history.append({"role": role, "content": content})
        # Keep history short for memory reasons
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

    def _fallback_assessment(self, prompt: str) -> Dict[str, Any]:
        text = prompt.lower()
        words = set(text.replace(",", " ").replace(".", " ").split())
        for entry in FALLBACK_LIBRARY:
            if entry["keywords"].issubset(words):
                return entry["content"]
        return DEFAULT_FALLBACK_RESPONSE

    def _build_prompt(self, prompt: str) -> str:
        context_lines = []
        for message in self._conversation_history[-4:]:
            role = "Assistant" if message["role"] == "assistant" else "Patient"
            context_lines.append(f"{role}: {message['content']}")
        context_block = "\n".join(context_lines) if context_lines else "(no prior exchanges)"
        return RESPONSE_TEMPLATE.format(context=context_block, prompt=prompt)

    def _parse_assessment(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return {
                    "summary": parsed.get("summary", ""),
                    "possible_causes": parsed.get("possible_causes", []),
                    "follow_up_questions": parsed.get("follow_up_questions", []),
                    "recommendations": parsed.get("recommendations", []),
                    "warning_signs": parsed.get("warning_signs", []),
                    "raw_text": raw_text,
                }
        except json.JSONDecodeError:
            pass

        # Fallback if the model did not return valid JSON
        sections = self._extract_sections_from_text(raw_text)
        sections["raw_text"] = raw_text
        return sections

    @staticmethod
    def _extract_sections_from_text(raw_text: str) -> Dict[str, Any]:
        sections = {
            "summary": "",
            "possible_causes": [],
            "follow_up_questions": [],
            "recommendations": [],
            "warning_signs": [],
        }
        current_key = "summary"
        for line in raw_text.splitlines():
            clean = line.strip(" -:\t")
            lower = clean.lower()
            if "cause" in lower:
                current_key = "possible_causes"
                continue
            if "question" in lower:
                current_key = "follow_up_questions"
                continue
            if "recommend" in lower:
                current_key = "recommendations"
                continue
            if "warning" in lower or "seek" in lower:
                current_key = "warning_signs"
                continue
            if not clean:
                continue
            if current_key == "summary":
                sections["summary"] += (clean + " ")
            else:
                sections[current_key].append(clean)
        sections["summary"] = sections["summary"].strip()
        return sections

    def get_model_info(self) -> Dict[str, Any]:  # type: ignore[override]
        base_info = super().get_model_info()
        base_info.update(
            {
                "model_name": self.model_name,
                "conversation_messages": len(self._conversation_history),
                "timeout_seconds": self._timeout_seconds,
            }
        )
        return base_info