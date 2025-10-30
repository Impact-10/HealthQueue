"""Quick test for the fine-tuned model.

Loads from ./medalpaca-custom-diagnosis and answers a few sample questions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT.parent / "medalpaca-custom-diagnosis"


SAMPLES: List[str] = [
    "I have a throbbing headache with nausea and light sensitivity for three days. What could this be?",
    "I feel burning chest discomfort after meals and it worsens when I lie down. Any advice?",
    "I have a cough, runny nose, sore throat, and low fever for a week. What is likely?",
]


def format_prompt(question: str) -> str:
    return (
        "You are a helpful medical assistant. Provide concise, evidence-based guidance.\n"
        f"Patient: {question}\nAssistant:"
    )


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {MODEL_DIR}. Train it first.")

    print(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for q in SAMPLES:
        prompt = format_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_length=int(os.getenv("GEN_MAX_LENGTH", "256")),
            temperature=float(os.getenv("GEN_TEMPERATURE", "0.7")),
            do_sample=True,
            top_p=0.9,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\nQuestion:", q)
        print("Answer:")
        print(text.split("Assistant:", 1)[-1].strip())


if __name__ == "__main__":
    main()


