"""Spell & medical abbreviation normalization utilities.

Usage:
    from utils.spell_norm import normalize_query
    cleaned = normalize_query("pt c/o sob & htn")

This module uses SymSpell for fast spell correction plus a simple
medical synonym / abbreviation expansion map. It loads lazily to avoid
startup cost if not used.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import List

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:  # graceful fallback if dependency not installed yet
    SymSpell = None  # type: ignore
    Verbosity = None  # type: ignore

# Comprehensive medical abbreviation expansion
_ABBREV_MAP = {
    "bp": "blood pressure",
    "sob": "shortness of breath",
    "htn": "hypertension",
    "dm": "diabetes",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "pt": "patient",
    "c/o": "complains of",
    "hx": "history",
    "dx": "diagnosis",
    "fx": "fracture",
    "w/u": "workup",
    "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure",
    "afib": "atrial fibrillation",
    "uti": "urinary tract infection",
    "pneumonia": "pneumonia",
    "asthma": "asthma",
    "gerd": "gastroesophageal reflux disease",
    "ibd": "inflammatory bowel disease",
    "ibs": "irritable bowel syndrome",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "ards": "acute respiratory distress syndrome",
    "sepsis": "sepsis",
    "mrsa": "methicillin resistant staphylococcus aureus",
    "vte": "venous thromboembolism",
}

_DICTIONARY_PATH = Path(__file__).resolve().parents[1] / "data" / "spell_dictionary.txt"


def _ensure_dictionary(sym) -> None:
    """Load or create a minimal dictionary for SymSpell.
    If no file exists, we bootstrap with abbreviation expansions & common words.
    """
    if _DICTIONARY_PATH.exists():
        sym.load_dictionary(str(_DICTIONARY_PATH), term_index=0, count_index=1)
        return
    # Create comprehensive medical dictionary
    common_terms = [
        # Basic medical terms
        "patient", "pain", "pressure", "chest", "heart", "diabetes", "hypertension",
        "infarction", "therapy", "treatment", "assessment", "blood", "shortness",
        "breath", "disease", "infection", "injury", "fever", "nausea", "cough",
        # Symptoms
        "headache", "dizziness", "fatigue", "weakness", "vomiting", "diarrhea",
        "constipation", "rash", "itching", "swelling", "inflammation", "bleeding",
        # Conditions
        "pneumonia", "asthma", "bronchitis", "pneumonia", "tuberculosis", "covid",
        "cancer", "tumor", "malignancy", "benign", "metastasis", "chemotherapy",
        "radiation", "surgery", "procedure", "operation", "anesthesia", "antibiotic",
        # Cardiovascular
        "cardiac", "arrhythmia", "tachycardia", "bradycardia", "angina", "ischemia",
        "thrombosis", "embolism", "aneurysm", "atherosclerosis", "cholesterol",
        # Respiratory
        "respiratory", "pulmonary", "ventilation", "oxygen", "asthma", "copd",
        "emphysema", "bronchitis", "pneumothorax", "pleural", "effusion",
        # Gastrointestinal
        "gastrointestinal", "stomach", "intestine", "liver", "pancreas", "gallbladder",
        "ulcer", "gastritis", "hepatitis", "cirrhosis", "jaundice", "appendicitis",
        # Neurological
        "neurological", "brain", "stroke", "seizure", "epilepsy", "migraine",
        "dementia", "alzheimer", "parkinson", "multiple sclerosis", "neuropathy",
        # Musculoskeletal
        "muscle", "bone", "joint", "arthritis", "osteoporosis", "fracture",
        "sprain", "strain", "tendon", "ligament", "cartilage", "spine", "vertebra",
        # Endocrine
        "endocrine", "thyroid", "hormone", "insulin", "glucose", "metabolism",
        "obesity", "weight", "diabetes", "hypoglycemia", "hyperglycemia",
        # Renal
        "kidney", "renal", "nephritis", "dialysis", "urine", "urinary", "bladder",
        # Mental health
        "depression", "anxiety", "stress", "mental", "psychiatric", "therapy",
        "counseling", "medication", "antidepressant", "antipsychotic",
        # Medications
        "medication", "drug", "prescription", "dosage", "dose", "side effect",
        "adverse", "contraindication", "allergy", "allergic", "reaction",
        # Diagnostic
        "diagnosis", "symptom", "sign", "test", "laboratory", "imaging", "xray",
        "mri", "ct scan", "ultrasound", "biopsy", "pathology", "radiology",
        # Treatment
        "treatment", "therapy", "medication", "surgery", "procedure", "rehabilitation",
        "physical therapy", "occupational therapy", "prevention", "vaccination",
        # Common phrases
        "shortness of breath", "complains of", "blood pressure", "coronary artery disease",
        "myocardial infarction", "chronic obstructive pulmonary disease",
    ] + list(set(_ABBREV_MAP.values()))
    with open(_DICTIONARY_PATH, "w", encoding="utf-8") as f:
        for w in common_terms:
            f.write(f"{w}\t1\n")
    sym.load_dictionary(str(_DICTIONARY_PATH), term_index=0, count_index=1)


@lru_cache(maxsize=1)
def _get_symspell():  # -> Optional[SymSpell]
    if SymSpell is None:
        return None
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    _ensure_dictionary(sym)
    return sym


def _expand_abbreviations(tokens: List[str]) -> List[str]:
    expanded: List[str] = []
    for t in tokens:
        key = t.lower()
        if key in _ABBREV_MAP:
            expanded.extend(_ABBREV_MAP[key].split())
        else:
            expanded.append(t)
    return expanded


# Common medical typos mapping (handled before SymSpell for reliability)
_MEDICAL_TYPOS = {
    "diabetis": "diabetes",
    "diabeties": "diabetes",
    "diabetus": "diabetes",
    "symptons": "symptoms",
    "symptomes": "symptoms",
    "symptomps": "symptoms",
    "pnemonia": "pneumonia",
    "pnuemonia": "pneumonia",
    "pneumoniae": "pneumonia",
    "hypertensoin": "hypertension",
    "hypertention": "hypertension",
    "hypertenssion": "hypertension",
    "hypertensio": "hypertension",
    "asthmae": "asthma",
    "bronchitis": "bronchitis",
    "bronchitits": "bronchitis",
    "cardiac": "cardiac",
    "cardic": "cardiac",
    "arrhythmia": "arrhythmia",
    "arrythmia": "arrhythmia",
    "arrythmias": "arrhythmias",
    "tachycardia": "tachycardia",
    "tachicardia": "tachycardia",
    "bradycardia": "bradycardia",
    "bradicardia": "bradycardia",
    "angina": "angina",
    "anginaa": "angina",
    "ischemia": "ischemia",
    "ischaemia": "ischemia",
    "thrombosis": "thrombosis",
    "thrombosiss": "thrombosis",
    "embolism": "embolism",
    "embolizm": "embolism",
    "aneurysm": "aneurysm",
    "aneurism": "aneurysm",
    "atherosclerosis": "atherosclerosis",
    "atherosclerosiss": "atherosclerosis",
    "cholesterol": "cholesterol",
    "cholesteral": "cholesterol",
    "respiratory": "respiratory",
    "respiratry": "respiratory",
    "pulmonary": "pulmonary",
    "pulmonry": "pulmonary",
    "ventilation": "ventilation",
    "ventilaton": "ventilation",
    "oxygen": "oxygen",
    "oxygn": "oxygen",
    "copd": "copd",  # abbreviation, will be expanded
    "pneumothorax": "pneumothorax",
    "pneumothorx": "pneumothorax",
    "pleural": "pleural",
    "pleurial": "pleural",
    "effusion": "effusion",
    "effuson": "effusion",
    "gastrointestinal": "gastrointestinal",
    "gastrointestnal": "gastrointestinal",
    "stomach": "stomach",
    "stomache": "stomach",
    "intestine": "intestine",
    "intestin": "intestine",
    "liver": "liver",
    "livr": "liver",
    "pancreas": "pancreas",
    "pancreass": "pancreas",
    "gallbladder": "gallbladder",
    "gallblader": "gallbladder",
    "ulcer": "ulcer",
    "ulcerr": "ulcer",
    "gastritis": "gastritis",
    "gastritiss": "gastritis",
    "hepatitis": "hepatitis",
    "hepatitiss": "hepatitis",
    "cirrhosis": "cirrhosis",
    "cirrhosiss": "cirrhosis",
    "jaundice": "jaundice",
    "jaundic": "jaundice",
    "appendicitis": "appendicitis",
    "appendicitiss": "appendicitis",
    "neurological": "neurological",
    "neurologcal": "neurological",
    "brain": "brain",
    "braine": "brain",
    "stroke": "stroke",
    "stroek": "stroke",
    "seizure": "seizure",
    "seizur": "seizure",
    "epilepsy": "epilepsy",
    "epilepsie": "epilepsy",
    "migraine": "migraine",
    "migrane": "migraine",
    "dementia": "dementia",
    "dementa": "dementia",
    "alzheimer": "alzheimer",
    "alzheimers": "alzheimer",
    "parkinson": "parkinson",
    "parkinsons": "parkinson",
    "multiple": "multiple",
    "multple": "multiple",
    "sclerosis": "sclerosis",
    "sclerosiss": "sclerosis",
    "neuropathy": "neuropathy",
    "neuropaty": "neuropathy",
    "muscle": "muscle",
    "muscl": "muscle",
    "bone": "bone",
    "bon": "bone",
    "joint": "joint",
    "joit": "joint",
    "arthritis": "arthritis",
    "arthritiss": "arthritis",
    "osteoporosis": "osteoporosis",
    "osteoporosiss": "osteoporosis",
    "fracture": "fracture",
    "fractur": "fracture",
    "sprain": "sprain",
    "spraen": "sprain",
    "strain": "strain",
    "straen": "strain",
    "tendon": "tendon",
    "tendn": "tendon",
    "ligament": "ligament",
    "ligamnt": "ligament",
    "cartilage": "cartilage",
    "cartilag": "cartilage",
    "spine": "spine",
    "spien": "spine",
    "vertebra": "vertebra",
    "vertebrae": "vertebra",
    "endocrine": "endocrine",
    "endocine": "endocrine",
    "thyroid": "thyroid",
    "thyroi": "thyroid",
    "hormone": "hormone",
    "hormon": "hormone",
    "insulin": "insulin",
    "insuln": "insulin",
    "glucose": "glucose",
    "glucos": "glucose",
    "metabolism": "metabolism",
    "metabolis": "metabolism",
    "obesity": "obesity",
    "obesit": "obesity",
    "weight": "weight",
    "weght": "weight",
    "hypoglycemia": "hypoglycemia",
    "hypoglycemi": "hypoglycemia",
    "hyperglycemia": "hyperglycemia",
    "hyperglycemi": "hyperglycemia",
    "kidney": "kidney",
    "kidne": "kidney",
    "renal": "renal",
    "renl": "renal",
    "nephritis": "nephritis",
    "nephritiss": "nephritis",
    "dialysis": "dialysis",
    "dialysiss": "dialysis",
    "urine": "urine",
    "urin": "urine",
    "urinary": "urinary",
    "urinry": "urinary",
    "bladder": "bladder",
    "blader": "bladder",
    "depression": "depression",
    "depresion": "depression",
    "anxiety": "anxiety",
    "anxiet": "anxiety",
    "stress": "stress",
    "stres": "stress",
    "mental": "mental",
    "mentl": "mental",
    "psychiatric": "psychiatric",
    "psychiatrc": "psychiatric",
    "therapy": "therapy",
    "therapi": "therapy",
    "counseling": "counseling",
    "counsling": "counseling",
    "medication": "medication",
    "medicaton": "medication",
    "antidepressant": "antidepressant",
    "antidepresant": "antidepressant",
    "antipsychotic": "antipsychotic",
    "antipsychotc": "antipsychotic",
    "drug": "drug",
    "dru": "drug",
    "prescription": "prescription",
    "prescripton": "prescription",
    "dosage": "dosage",
    "dosag": "dosage",
    "dose": "dose",
    "dos": "dose",
    "side": "side",
    "sde": "side",
    "effect": "effect",
    "efect": "effect",
    "adverse": "adverse",
    "advers": "adverse",
    "contraindication": "contraindication",
    "contraindicton": "contraindication",
    "allergy": "allergy",
    "alergy": "allergy",
    "allergic": "allergic",
    "alergic": "allergic",
    "reaction": "reaction",
    "reacton": "reaction",
    "diagnosis": "diagnosis",
    "diagnosiss": "diagnosis",
    "symptom": "symptom",
    "symptm": "symptom",
    "sign": "sign",
    "sgn": "sign",
    "test": "test",
    "tst": "test",
    "laboratory": "laboratory",
    "laboratry": "laboratory",
    "imaging": "imaging",
    "imagin": "imaging",
    "xray": "xray",
    "x-ray": "xray",
    "mri": "mri",
    "ct": "ct",
    "scan": "scan",
    "scn": "scan",
    "ultrasound": "ultrasound",
    "ultrasond": "ultrasound",
    "biopsy": "biopsy",
    "biopsi": "biopsy",
    "pathology": "pathology",
    "patholgy": "pathology",
    "radiology": "radiology",
    "radiolgy": "radiology",
    "treatment": "treatment",
    "treatmnt": "treatment",
    "surgery": "surgery",
    "surgry": "surgery",
    "procedure": "procedure",
    "procedur": "procedure",
    "rehabilitation": "rehabilitation",
    "rehabilitaton": "rehabilitation",
    "physical": "physical",
    "physcal": "physical",
    "occupational": "occupational",
    "occupatonal": "occupational",
    "prevention": "prevention",
    "preventon": "prevention",
    "vaccination": "vaccination",
    "vaccinaton": "vaccination",
}


def normalize_query(query: str) -> str:
    """Apply abbreviation expansion + common typo fixes + spell correction (best suggestion)."""
    if not query:
        return query
    import re
    tokens = query.strip().split()
    
    # Step 1: Fix common medical typos first (before abbreviation expansion)
    # Strip punctuation for lookup, but preserve it in the output
    corrected_tokens = []
    for tok in tokens:
        # Extract word (strip punctuation) for lookup
        word_match = re.match(r'^([^\w]*)(\w+)([^\w]*)$', tok)
        if word_match:
            prefix, word, suffix = word_match.groups()
            key = word.lower()
            if key in _MEDICAL_TYPOS:
                # Replace the word but keep prefix and suffix
                corrected_tokens.append(prefix + _MEDICAL_TYPOS[key] + suffix)
            else:
                corrected_tokens.append(tok)
        else:
            # No word characters, keep as-is
            corrected_tokens.append(tok)
    
    # Step 2: Expand abbreviations
    tokens = _expand_abbreviations(corrected_tokens)
    
    # Step 3: Run SymSpell for any remaining corrections
    sym = _get_symspell()
    if sym is None:  # dependency missing; just return expanded tokens
        return " ".join(tokens)
    final_corrected: List[str] = []
    for tok in tokens:
        suggestions = sym.lookup(tok, Verbosity.TOP, max_edit_distance=2)
        if suggestions:
            final_corrected.append(suggestions[0].term)
        else:
            final_corrected.append(tok)
    return " ".join(final_corrected)


if __name__ == "__main__":
    examples = [
        "pt c/o sob & htn", "bp high chest pian", "hx of mi and dm", "fever and couhg",
    ]
    for e in examples:
        print(e, "->", normalize_query(e))


