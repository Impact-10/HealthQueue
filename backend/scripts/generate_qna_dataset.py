"""Generate a synthetic medical diagnosis Q/A dataset.

Outputs:
- Plain JSONL at backend/data/med_qna.jsonl
- Plain JSON at backend/data/med_qna.json
- CSV at backend/data/med_qna.csv

Record format per line (JSON/JSONL):
{"question": "...", "answer": "..."}

No external APIs required. Designed to be lightweight and deterministic.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _join_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


@dataclass(frozen=True)
class SymptomPattern:
    name: str
    templates: Tuple[str, ...]


CONDITIONS: Dict[str, Dict[str, List[str]]] = {
    "migraine": {
        "symptoms": [
            "throbbing headache",
            "nausea",
            "light sensitivity",
            "sound sensitivity",
            "visual aura",
        ],
        "advice": [
            "rest in a dark, quiet room",
            "use over-the-counter analgesics if appropriate",
            "hydrate well",
            "track triggers like sleep changes or certain foods",
        ],
        "flags": [
            "sudden worst headache",
            "fever with neck stiffness",
            "neurologic deficits (weakness, confusion)",
        ],
    },
    "tension headache": {
        "symptoms": [
            "band-like pressure headache",
            "mild to moderate pain",
            "neck or scalp tightness",
        ],
        "advice": [
            "practice stress reduction and posture care",
            "consider gentle stretching or heat",
            "short-term analgesics can help",
        ],
        "flags": [
            "sudden severe pain",
            "headache after injury",
            "new headache over age 50",
        ],
    },
    "viral upper respiratory infection": {
        "symptoms": [
            "runny nose",
            "sore throat",
            "cough",
            "low-grade fever",
            "fatigue",
        ],
        "advice": [
            "rest and fluids",
            "honey or throat lozenges for cough",
            "acetaminophen or ibuprofen for fever",
        ],
        "flags": [
            "shortness of breath",
            "high persistent fever",
            "chest pain",
        ],
    },
    "acute sinusitis": {
        "symptoms": [
            "facial pressure",
            "nasal congestion",
            "thick nasal discharge",
            "dental pain",
        ],
        "advice": [
            "saline irrigation",
            "steam inhalation",
            "decongestants for short-term relief",
        ],
        "flags": [
            "eye swelling",
            "severe headache",
            "fever over 39°C",
        ],
    },
    "gerd": {
        "symptoms": [
            "burning chest discomfort after meals",
            "acidic taste",
            "worse when lying down",
        ],
        "advice": [
            "avoid late meals, caffeine, alcohol, and spicy foods",
            "elevate head of bed",
            "trial of antacids or H2 blockers",
        ],
        "flags": [
            "trouble swallowing",
            "unintentional weight loss",
            "chest pain with exertion",
        ],
    },
    "asthma": {
        "symptoms": [
            "wheezing",
            "episodic shortness of breath",
            "chest tightness",
            "cough worse at night",
        ],
        "advice": [
            "use prescribed rescue inhaler if available",
            "avoid triggers like smoke or allergens",
            "seek evaluation for controller therapy",
        ],
        "flags": [
            "speaking in single words",
            "lips or face turning blue",
            "no relief with rescue inhaler",
        ],
    },
    "urinary tract infection": {
        "symptoms": [
            "burning urination",
            "increased urinary frequency",
            "urgency",
            "lower abdominal discomfort",
        ],
        "advice": [
            "hydrate and avoid bladder irritants",
            "seek urine testing for diagnosis",
            "prompt antibiotics when confirmed",
        ],
        "flags": [
            "fever or flank pain",
            "vomiting",
            "pregnancy with symptoms",
        ],
    },
    "type 2 diabetes concern": {
        "symptoms": [
            "increased thirst",
            "frequent urination",
            "fatigue",
            "blurred vision",
        ],
        "advice": [
            "screen with fasting glucose or HbA1c",
            "increase physical activity and optimize diet",
            "discuss risk factors with clinician",
        ],
        "flags": [
            "confusion or drowsiness",
            "nausea with abdominal pain",
            "rapid breathing",
        ],
    },
    "iron deficiency anemia": {
        "symptoms": [
            "fatigue",
            "pale skin",
            "shortness of breath on exertion",
            "brittle nails",
        ],
        "advice": [
            "check CBC and ferritin",
            "increase dietary iron and vitamin C",
            "evaluate for blood loss",
        ],
        "flags": [
            "chest pain",
            "black stools",
            "fainting",
        ],
    },
    "allergic rhinitis": {
        "symptoms": [
            "sneezing",
            "itchy watery eyes",
            "runny or stuffy nose",
        ],
        "advice": [
            "avoid allergens and use saline rinses",
            "trial non-sedating antihistamines",
            "consider intranasal corticosteroids",
        ],
        "flags": [
            "wheezing or severe shortness of breath",
            "facial swelling",
        ],
    },
}


SYMPTOM_PATTERNS: Tuple[SymptomPattern, ...] = (
    SymptomPattern(
        "duration_intensity",
        (
            "For the past {duration}, I've had {symptom} rated about {severity}/10.",
            "{duration} of {symptom} around {severity}/10 intensity.",
            "I've been experiencing {symptom} for {duration}, roughly {severity}/10.",
        ),
    ),
    SymptomPattern(
        "trigger_timing",
        (
            "It gets worse {timing}, especially after {trigger}.",
            "Worse {timing}; {trigger} seems to precipitate it.",
            "{timing} makes it worse, often following {trigger}.",
        ),
    ),
    SymptomPattern(
        "associated",
        (
            "I also notice {assoc1} and {assoc2}.",
            "Associated symptoms include {assoc1} and {assoc2}.",
            "Additionally, there is {assoc1} with occasional {assoc2}.",
        ),
    ),
)


DURATIONS = [
    "two days",
    "a week",
    "three weeks",
    "about a month",
    "several hours each day",
]

SEVERITIES = ["3", "5", "6", "7", "8"]
TIMINGS = ["at night", "in the morning", "with exertion", "after meals", "when lying down"]
TRIGGERS = [
    "stress",
    "cold air exposure",
    "large meals",
    "missing sleep",
    "seasonal changes",
]


def _build_question(condition_key: str) -> str:
    cdata = CONDITIONS[condition_key]
    primary = random.choice(cdata["symptoms"])  # main symptom
    assoc = random.sample([s for s in cdata["symptoms"] if s != primary], k=min(2, max(0, len(cdata["symptoms"]) - 1)))
    assoc1, assoc2 = (assoc + ["fatigue", "mild fever"])[:2]
    parts: List[str] = []
    parts.append(
        random.choice(SYMPTOM_PATTERNS[0].templates).format(
            duration=random.choice(DURATIONS), symptom=primary, severity=random.choice(SEVERITIES)
        )
    )
    parts.append(
        random.choice(SYMPTOM_PATTERNS[1].templates).format(
            timing=random.choice(TIMINGS), trigger=random.choice(TRIGGERS)
        )
    )
    parts.append(
        random.choice(SYMPTOM_PATTERNS[2].templates).format(assoc1=assoc1, assoc2=assoc2)
    )
    question = " ".join(parts)
    question = re.sub(r"\s+", " ", question).strip()
    return _sentence_case(question)


def _build_answer(condition_key: str) -> str:
    cdata = CONDITIONS[condition_key]
    causes = _join_list([condition_key])
    recs = _join_list(cdata["advice"][:3])
    flags = _join_list(cdata["flags"][:2])
    answer = (
        f"Your description is most consistent with {causes} based on the pattern of symptoms. "
        f"Initial self-care includes {recs}. "
        f"If you develop {flags} or symptoms worsen, seek urgent medical care. "
        f"This is not a confirmed diagnosis; a clinician can evaluate and test if needed."
    )
    return _sentence_case(answer)


def generate_pairs(n: int = 1000, seed: int = 42) -> List[Dict[str, str]]:
    random.seed(seed)
    keys = list(CONDITIONS.keys())
    pairs: List[Dict[str, str]] = []
    # Round-robin conditions to ensure distribution
    for i in range(n):
        key = keys[i % len(keys)]
        q = _build_question(key)
        a = _build_answer(key)
        pairs.append({"question": q, "answer": a})
    return pairs


def save_all_formats(pairs: List[Dict[str, str]]) -> None:
    jsonl_path = DATA_DIR / "med_qna.jsonl"
    json_path = DATA_DIR / "med_qna.json"
    csv_path = DATA_DIR / "med_qna.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    pd.DataFrame(pairs).to_csv(csv_path, index=False)


def main() -> None:
    num = int(os.getenv("NUM_SAMPLES", "1000"))
    pairs = generate_pairs(num)
    save_all_formats(pairs)
    print(f"Wrote {len(pairs)} records to {DATA_DIR}")


if __name__ == "__main__":
    main()


