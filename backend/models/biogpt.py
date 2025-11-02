"""BioGPT lightweight wrapper with fallback-first behaviour."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseMedicalModel
from ..config import ensure_allowed_model_name, get_model_cache_dir, get_timeout_seconds, with_cache_dir


DIAGNOSIS_PROMPT = (
    "You are a biomedical AI assistant specialised in diabetology.\n"
    "Return a JSON object with keys: summary, possible_diagnoses (array of strings),\n"
    "rationale (array of strings), recommended_tests (array of strings),\n"
    "next_steps (array of strings), warning_signs (array of strings).\n"
    "Do not include extra commentary.\n\n"
    "Patient symptoms and history:\n{symptoms}\n\n"
    "Provide evidence-based reasoning and stay concise."
)

RESPONSE_TEMPLATE = (
    "You are a helpful, concise medical AI assistant.\n"
    "When a user describes symptoms, always suggest the most likely diagnosis or possible causes based on the information provided.\n"
    "Be clear that this is not a confirmed diagnosis, but your best assessment based on the symptoms.\n"
    "Always remind the user to consult a healthcare professional for confirmation and treatment.\n\n"
    "Answer the following patient question in plain English, providing clear, practical advice.\n\n"
    "Clinical conversation context begins below:\n{context}\n\n"
    "Patient prompt:\n{prompt}\n"
)

FALLBACK_LIBRARY: List[Dict[str, Any]] = [
    {
        "keywords": {"increased", "thirst", "urination"},
        "content": {
            "summary": "Symptoms suggestive of diabetes mellitus.",
            "possible_diagnoses": [
                "Type 1 diabetes",
                "Type 2 diabetes",
                "Gestational diabetes",
            ],
            "rationale": [
                "Polyuria and polydipsia are classic signs of hyperglycemia",
                "Differential diagnosis based on age, onset, and risk factors",
                "Gestational diabetes if pregnant",
            ],
            "recommended_tests": [
                "Fasting blood glucose",
                "HbA1c",
                "Oral glucose tolerance test",
            ],
            "next_steps": [
                "Consult endocrinologist for evaluation",
                "Monitor blood glucose if diabetic",
                "Lifestyle modifications",
            ],
            "warning_signs": [
                "Diabetic ketoacidosis (nausea, vomiting, fruity breath)",
                "Hyperglycemic hyperosmolar state",
                "Severe dehydration or altered mental status",
            ],
        },
    }
]

DEFAULT_FALLBACK: Dict[str, Any] = {
    "summary": "General diabetology assessment guidance.",
    "possible_diagnoses": ["Additional clinical detail required"],
    "rationale": ["Insufficient structured information provided"],
    "recommended_tests": [
        "Blood glucose testing",
        "HbA1c measurement",
    ],
    "next_steps": [
        "Document symptom onset, severity, and modifiers",
        "Consult a diabetologist or endocrinologist",
    ],
    "warning_signs": [
        "Severe hyperglycemia, hypoglycemia, or ketosis",
        "Signs of complications like neuropathy or retinopathy",
    ],
}


class BioGPTModel(BaseMedicalModel):
    """Biomedical text generation with configurable inference mode, fine-tuned for diabetology."""

    def __init__(self) -> None:
        requested_name = os.getenv("BIOGPT_MODEL_NAME", "./biogpt-diabetology-custom")
        self.model_name = requested_name
        super().__init__(
            "biogpt",
            device="cpu",
            max_length=int(os.getenv("BIOGPT_MAX_LENGTH", "512")),
        )
        self._timeout_seconds = get_timeout_seconds("biogpt", default=30)
        self._pad_token_id = None

    def _load_resources(self) -> None:
        """Load tokenizer/model when inference mode is enabled."""
        local_only = os.getenv("BIOGPT_LOCAL_ONLY", "0") == "1"
        load_kwargs = with_cache_dir({"local_files_only": local_only})

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
        temperature: float = 0.65,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return a structured diabetology assessment for the provided prompt."""
        if self.use_fallback():
            self.require_inference_disabled()
            content = self._fallback_assessment(prompt)
            warnings = ["Running in fallback mode; enable BIOGPT_MODE=inference for live model."]
            # Ensure non-empty summary
            if not content.get("summary") or not str(content["summary"]).strip():
                content["summary"] = (
                    "Consider hydration, balanced meals, and reviewing glucose control; "
                    "seek care if symptoms persist or red flags appear."
                )
            return self.build_response(
                content=content,
                metadata={"model_name": f"{self.model_name} (demo)"},
                warnings=warnings,
            )

        if not self.ensure_model_loaded():
            self.require_inference_disabled()
            content = self._fallback_assessment(prompt)
            warnings = ["Model load failed; using fallback content."]
            if not content.get("summary") or not str(content["summary"]).strip():
                content["summary"] = (
                    "Consider hydration, balanced meals, and reviewing glucose control; "
                    "seek care if symptoms persist or red flags appear."
                )
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
            max_length=self.max_length - 128,  # Reserve space for generation
        )
        # Attention mask and device placement
        inputs["attention_mask"] = (inputs["input_ids"] != self._tokenizer.pad_token_id).long()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Safer generation and continuation-only decoding
        generation = self._model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=112,        # was 96
    temperature=0.6,           # add sampling
    do_sample=True,            # was False
    top_p=0.85,
    num_beams=1,
    repetition_penalty=1.18,   # slightly lower to avoid crushing variation
    pad_token_id=getattr(self, "_pad_token_id", self._tokenizer.eos_token_id),
)


        prompt_len = inputs["input_ids"].shape[1]
        continuation_ids = generation[0][prompt_len:]
        raw_text = self._tokenizer.decode(continuation_ids, skip_special_tokens=True)

        # Marker for user intent (diabetes or not)
        user_low = prompt.strip().lower()
        is_diabetes_user = any(
            a in user_low for a in ("diabetes", "sugar", "glucose", "a1c", "insulin", "thirst", "urination", "hypergly", "hypogly")
        )
        content = self._parse_assessment(raw_text + f"\n<__DIAB_USER__:{int(is_diabetes_user)}>", prompt)

        # Clean and soften summary; ensure non-empty
        if "summary" in content and isinstance(content["summary"], str):
            content["summary"] = self._soften_summary(self._clean_summary(content["summary"]))
        if not content.get("summary") or not str(content["summary"]).strip():
            content["summary"] = (
                "Consider hydration, balanced meals, and reviewing glucose control; "
                "seek care if symptoms persist or red flags appear."
            )

        return self.build_response(
            content=content,
            metadata={
                "model_name": self.model_name,
                "temperature": temperature,
                "max_length": max_length or self.max_length,
            },
        )

    def _append_history(self, role: str, content: str) -> None:
        # History disabled in this diabetology wrapper
        pass

    def _fallback_assessment(self, prompt: str) -> Dict[str, Any]:
        text = prompt.lower()
        words = set(text.replace(",", " ").replace(".", " ").split())
        for entry in FALLBACK_LIBRARY:
            if entry["keywords"].issubset(words):
                return entry["content"]
        return DEFAULT_FALLBACK

    def _build_prompt(self, prompt: str) -> str:
        return DIAGNOSIS_PROMPT.format(symptoms=prompt)

    # Robust JSON extraction with diabetology-anchors safeguard
    def _parse_assessment(self, raw_text: str, prompt: str) -> Dict[str, Any]:
        text = raw_text.strip()

        # Extract user-intent marker and remove it
        is_diabetes_user = False
        if "<__DIAB_USER__:" in text:
            try:
                tag = text.rsplit("<__DIAB_USER__:", 1)[1]
                is_diabetes_user = tag.split(">", 1)[0].startswith("1")
            except Exception:
                pass
            text = text.split("<__DIAB_USER__:", 1)[0].strip()

        # Trim leaked roles before JSON if present
        first_role = min(
            [i for i in [text.find("Assistant:"), text.find("User:"), text.find("Patient:")] if i != -1],
            default=-1,
        )
        if first_role != -1:
            if "Assistant:" in text:
                idx = text.find("Assistant:")
                text = text[idx + len("Assistant:") :].strip()
            else:
                text = text[first_role:].split(":", 1)[-1].strip()

        # Extract outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return {
                        "summary": str(data.get("summary", "")).strip(),
                        "possible_diagnoses": data.get("possible_diagnoses", []),
                        "rationale": data.get("rationale", []),
                        "recommended_tests": data.get("recommended_tests", []),
                        "next_steps": data.get("next_steps", []),
                        "warning_signs": data.get("warning_signs", []),
                        "raw_text": raw_text,
                    }
            except Exception:
                pass

        # Heuristic section extraction if no valid JSON
        sections = self._extract_sections_from_text(text)
        if not sections.get("summary"):
            trimmed = " ".join(text.split())
            sections["summary"] = trimmed[:220] + ("…" if len(trimmed) > 220 else "")

        # Intent-aware logic
        diabetes_anchors = ("diabetes", "sugar", "glucose", "a1c", "insulin", "thirst", "urination", "polyuria", "nocturia", "hypergly", "hypogly")
        summary_low = (sections.get("summary") or "").lower()
        user_low = (prompt or "").lower()

        is_diabetes_user = any(a in user_low for a in diabetes_anchors)

        # Only steer to the generic diabetology line if user intent is diabetes AND summary lacks any diabetic anchors
        if is_diabetes_user and not any(a in summary_low for a in diabetes_anchors):
            sections["summary"] = (
                "High blood sugar with fatigue could relate to sleep quality, dehydration, medication effects, "
                "or suboptimal glucose control. Consider consistent meals, hydration, light activity, and "
                "follow up on A1c and medication timing with your clinician."
            )

        # Only inject the wound default if the USER mentioned wounds/ulcers/skin, not otherwise
        wound_terms = ("wound", "ulcer", "heal", "healing", "skin", "itch", "dry")
        if any(t in user_low for t in wound_terms) and not any(t in summary_low for t in wound_terms):
            sections["summary"] = (
                "Slow-healing wounds can be related to high blood sugar, infection, or circulation issues. "
                "Keep wounds clean, avoid pressure, and seek prompt clinical evaluation if redness, warmth, swelling, or fever appear."
            )

        # Blend symptom hint
        hint = self._symptom_hint(prompt or "")
        if hint and hint.lower() not in ((sections["summary"] or "").lower()):
            base = sections["summary"] or ""
            sections["summary"] = (base + (" " if base else "") + hint).strip()

        # Final non-empty guard with neutral triage line
        if not sections.get("summary") or not str(sections["summary"]).strip():
            sections["summary"] = (
                "Consider hydration, balanced meals, and reviewing glucose control; "
                "track symptoms and seek care if red flags like chest pain, fever, or worsening occur."
            )

        sections["raw_text"] = raw_text
        return sections

    @staticmethod
    def _extract_sections_from_text(raw_text: str) -> Dict[str, Any]:
        sections = {
            "summary": "",
            "possible_diagnoses": [],
            "rationale": [],
            "recommended_tests": [],
            "next_steps": [],
            "warning_signs": [],
        }
        current_key = "summary"
        for line in raw_text.splitlines():
            clean = line.strip(" -:\t")
            lower = clean.lower()
            if "diagnos" in lower:
                current_key = "possible_diagnoses"
                continue
            if "rational" in lower:
                current_key = "rationale"
                continue
            if "test" in lower:
                current_key = "recommended_tests"
                continue
            if "next" in lower or "step" in lower:
                current_key = "next_steps"
                continue
            if "warning" in lower or "sign" in lower:
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

    def _clean_summary(self, s: str) -> str:
        txt = " ".join(s.split())
        drops = [
    "image credit", "shutterstock", "associated press", "copyright",
    "all rights reserved", "terms of use", "privacy policy", "editor's note",
    "contact us", "open-access article", "distributed under the terms",
    "for reprints", "press release", "standards editor", "news organization",
    "www/", "http://", "https://", "documentation", "support information",
    "assistant diagnostics", "assistant diagnostication", "assistant diagnology",
    "assistant diagnosals", "docs/", "adoption.org", "index.php",
    "additional resources", "practical guide", "managing diabetes", "applet",
    "smartcard", "card system", "topics are available", "this article",
    "assistant-testimonials", "assistant-", "testimonials",
    "documentation & documentation resources", "assistant diagnostic team",
    "keyword:", "key points", "avoid unnecessary complications", "reduce risk",
    "improve adherence", "additional reporting", "assistant professor",
    "dr john", "wojc", "congestive pulmonary", "coronary arteries",
    "kidney failure", "liver problems", "high blood pressure", "excessive stress",
    "avoiding the use of medication", "cardiovascular diseases", "strokes",
    "cancer", "hypertension", "stroke", "heart disease", "etc.", "avoid using medications",
    "maintaining healthy habits"
]

        low = txt.lower()
        for d in drops:
            if d in low:
                parts = [p.strip() for p in txt.split(".") if p.strip()]
                parts = [p for p in parts if d not in p.lower()]
                txt = ". ".join(parts[:3])
                if txt and not txt.endswith("."):
                    txt += "."
                break
        sentences = [p.strip() for p in txt.split(".") if p.strip()]
        txt = ". ".join(sentences[:3])
        if txt and not txt.endswith("."):
            txt += "."
        return txt

    def _symptom_hint(self, user_text: str) -> str:
        t = (user_text or "").lower()
        if "urinate" in t or "nocturia" in t or "night" in t and "urinate" in t:
            return "Nighttime urination can reflect evening fluids, diuretics, sleep apnea, or elevated glucose; reduce late fluids and discuss timing of medicines."
        if "itch" in t or "dry skin" in t or "pruritus" in t:
            return "Itchy, dry skin is common with dehydration and high sugars; moisturize after bathing, use gentle cleansers, and hydrate."
        if "thirst" in t or "polydipsia" in t:
            return "Excess thirst with frequent urination suggests high glucose; check fasting/bedtime sugars and review A1c."
        return ""

    def _soften_summary(self, s: str) -> str:
        txt = s.strip()
        replacements = [
            ("is most consistent with", "could be consistent with"),
            ("is consistent with", "may be consistent with"),
            ("is suggestive of", "could suggest"),
            ("This is not a confirmed diagnosis", "This isn’t a diagnosis"),
        ]
        for a, b in replacements:
            txt = txt.replace(a, b)
            txt = txt.replace(a.capitalize(), b.capitalize())
        low = txt.lower()
        if "retinopath" in low and "vision" not in low:
            txt += " If you also notice blurred vision, floaters, or vision changes, consider getting your eyes checked soon."
        if "diabetes" in low and ("thirst" not in low and "urination" not in low):
            txt += " Watch for increased thirst, frequent urination, fatigue, or slow-healing wounds."
        return " ".join(txt.split())

    def get_model_info(self) -> Dict[str, Any]:  # type: ignore[override]
        base_info = super().get_model_info()
        base_info.update(
            {
                "model_name": self.model_name,
                "timeout_seconds": self._timeout_seconds,
            }
        )
        return base_info
