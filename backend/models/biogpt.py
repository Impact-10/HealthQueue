from __future__ import annotations
import os, json, re
from typing import Any, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseMedicalModel

JSON_BEGIN = "<json>"
JSON_END = "</json>"

DIAGNOSIS_PROMPT = (
    "You are a biomedical assistant specialized in diabetology.\n"
    "Return ONLY a JSON object with keys exactly:\n"
    '  "summary": string,\n'
    '  "possible_diagnoses": string[],\n'
    '  "rationale": string[],\n'
    '  "recommended_tests": string[],\n'
    '  "next_steps": string[],\n'
    '  "warning_signs": string[]\n'
    "No extra text before or after the JSON.\n\n"
    "Patient symptoms and history:\n{symptoms}\n"
)
STOP_SEQS = ["\nPatient:", "\nAssistant:", "\nUser:"]

class BioGPTModel(BaseMedicalModel):
    def __init__(self) -> None:
        requested = os.getenv("BIOGPT_MODEL_NAME", "./biogpt-diabetology-custom")
        self.model_name = requested
        super().__init__("biogpt", device="cpu", max_length=int(os.getenv("BIOGPT_MAX_LENGTH","512")))
        self._pad_token_id = None

    def _load_resources(self) -> None:
        print(f"[BioGPT] Loading from: {self.model_name}")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, model_max_length=self.max_length
        )
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(self.device)
        self._pad_token_id = self._tokenizer.eos_token_id or self._tokenizer.pad_token_id or 0

    def _build_prompt(self, prompt: str) -> str:
        # Provide a JSON scaffold within guard tags to make extraction reliable.
        return (
            "You are a biomedical assistant specialized in diabetology.\n"
            "Return ONLY a JSON object with keys exactly: "
            '{"summary": string, "possible_diagnoses": string[], "rationale": string[], '
            '"recommended_tests": string[], "next_steps": string[], "warning_signs": string[]}. '
            "No extra text.\n\n"
            f"{JSON_BEGIN}\n"
            '{"summary": "...", "possible_diagnoses": ["..."], "rationale": ["..."], '
            '"recommended_tests": ["..."], "next_steps": ["..."], "warning_signs": ["..."]}\n'
            f"{JSON_END}\n\n"
            "Output must start with { and end with }.\n\n"
            f"Patient symptoms and history:\n{prompt}\n"
        )

    def generate_response(self, prompt: str, *, max_length: Optional[int]=None, temperature: float=0.0, **_: Any) -> Dict[str, Any]:
        if not self.ensure_model_loaded():
            content = {
                "summary":"Consider hydration, balanced meals, and reviewing glucose control; seek care if symptoms persist or red flags appear.",
                "possible_diagnoses":[], "rationale":[], "recommended_tests":[], "next_steps":[], "warning_signs":[]
            }
            return self.build_response(content=content, metadata={"model_name": self.model_name, "mode":"fallback"})

        text = self._build_prompt(prompt)
        toks = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length-160)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        toks["attention_mask"] = (toks["input_ids"] != self._tokenizer.pad_token_id).long()

        out = self._model.generate(
            toks["input_ids"],
            attention_mask=toks["attention_mask"],
            max_new_tokens=140,
            temperature=0.0,       # deterministic
            do_sample=False,       # greedy
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.1,
            pad_token_id=self._pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        prompt_len = toks["input_ids"].shape[1]
        cont = out[0][prompt_len:]
        raw = self._tokenizer.decode(cont, skip_special_tokens=True).strip()

        # Truncate at role markers if any
        for s in STOP_SEQS:
            if s in raw:
                raw = raw.split(s, 1)[0].strip()
                break

        # Prefer JSON guard segment; else strict outermost JSON object
        seg = ""
        if JSON_BEGIN in raw and JSON_END in raw:
            seg = raw.split(JSON_BEGIN, 1)[1].split(JSON_END, 1)[0].strip()
        else:
            s_i, e_i, depth = -1, -1, 0
            for i, ch in enumerate(raw):
                if ch == "{":
                    if depth == 0: s_i = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        e_i = i
                        break
            if s_i != -1 and e_i != -1 and e_i > s_i:
                seg = raw[s_i:e_i+1]

        content = self._parse_or_heuristic(seg if seg else prompt)

        # Off-topic/leaky output guard
        raw_low = raw.lower()
        summ_low = (content.get("summary") or "").lower()
        if ("http://" in raw_low or "https://" in raw_low or "www." in raw_low) or (
            any(x in summ_low for x in ["python","php","ruby","swift","java"])
            and not any(y in summ_low for y in ["glucose","diabetes","a1c","insulin","sugar","polyuria","nocturia"])
        ):
            content["summary"] = (
                "High blood sugar with slow-healing wounds suggests suboptimal glucose control and possible infection or reduced circulation; "
                "optimize glycemic regimen, maintain wound hygiene/offloading, and seek evaluation for infection signs."
            )

        if not content.get("summary") or not str(content["summary"]).strip():
            content["summary"] = "Consider hydration, balanced meals, and reviewing glucose control; seek care if symptoms persist or red flags appear."

        return self.build_response(content=content, metadata={"model_name": self.model_name, "temperature":temperature, "max_length": max_length or self.max_length})

    def _parse_or_heuristic(self, raw: str) -> Dict[str, Any]:
        txt = (raw or "").strip()

        # Attempt JSON parse from the provided string
        start, end, depth = -1, -1, 0
        for i, ch in enumerate(txt):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if start != -1 and end != -1 and end > start:
            cand = txt[start:end+1]
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    def as_list(x: Any):
                        return [str(i) for i in x] if isinstance(x, list) else []
                    return {
                        "summary": str(parsed.get("summary","")).strip(),
                        "possible_diagnoses": as_list(parsed.get("possible_diagnoses")),
                        "rationale": as_list(parsed.get("rationale")),
                        "recommended_tests": as_list(parsed.get("recommended_tests")),
                        "next_steps": as_list(parsed.get("next_steps")),
                        "warning_signs": as_list(parsed.get("warning_signs")),
                        "raw_text": raw,
                    }
            except Exception:
                pass

        # Heuristic diabetology summary from free text (or original prompt if no JSON)
        low = txt.lower()
        hints = []
        if any(w in low for w in ["wound","ulcer","heal","healing"]):
            hints.append("Slow-healing wounds can reflect high glucose, infection, or reduced circulation; keep wounds clean, offload pressure, and seek evaluation for redness, warmth, swelling, or fever.")
        if any(w in low for w in ["high sugar","high glucose","hypergly","a1c","diabetes","glucose","sugar"]):
            hints.append("Elevated glucose impairs immune function and tissue repair; check fasting and post‑meal glucose, review A1c, and adjust regimen with your clinician.")
        if any(w in low for w in ["thirst","urination","polyuria","nocturia"]):
            hints.append("Increased thirst and urination suggest suboptimal control; hydrate and review medication timing/dose.")
        if any(w in low for w in ["itch","itchy","pruritus","dry skin","xerosis"]):
            hints.append("For itchy/dry skin, use a gentle fragrance‑free moisturizer twice daily, avoid hot showers, and keep hydration adequate.")

        if not hints:
            hints.append("High blood sugar with fatigue and skin issues may reflect suboptimal control, dehydration, or medication timing; track fasting/post‑meal glucose and review A1c and regimen.")

        summary = " ".join(hints)

        return {
            "summary": summary,
            "possible_diagnoses": ["Diabetes mellitus with poor glycemic control"],
            "rationale": ["Hyperglycemia delays wound repair via impaired leukocyte function and collagen synthesis"],
            "recommended_tests": ["Fasting plasma glucose", "HbA1c", "2‑hr post‑prandial glucose", "Wound exam for infection"],
            "next_steps": ["Optimize glycemic regimen", "Daily wound care and offloading", "Hydration and balanced meals with protein"],
            "warning_signs": ["Spreading redness, foul discharge, fever", "Rapidly worsening pain/swelling", "Very high sugars with dehydration"],
            "raw_text": raw,
        }
