"""Safety filter for medical QA system.

Blocks unsafe, inappropriate, or out-of-scope queries before processing.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

# Blocked patterns: explicit content, violence, illegal activities
BLOCKED_PATTERNS = [
    r'\b(kill|murder|suicide|self.?harm|cutting|overdose)\b',
    r'\b(bomb|weapon|gun|knife|poison)\b',
    r'\b(illegal|drug|dealer|prescription.?abuse)\b',
    r'\b(explicit|porn|sexual|nude)\b',
    r'\b(hack|virus|malware|exploit)\b',
]

# Out-of-scope patterns: non-medical topics
OUT_OF_SCOPE_PATTERNS = [
    r'\b(sports|football|basketball|game|score|player)\b',
    r'\b(movie|film|actor|celebrity|entertainment)\b',
    r'\b(coding|programming|python|javascript|code)\b',
    r'\b(politics|election|president|government|vote)\b',
    r'\b(stock|market|invest|trading|finance|money)\b',
    r'\b(cooking|recipe|food|restaurant)\b',  # unless health-related
    r'\b(travel|vacation|hotel|flight)\b',
]

# Medical emergency keywords (should be flagged but not blocked)
EMERGENCY_KEYWORDS = [
    r'\b(chest.?pain|heart.?attack|stroke|severe|emergency|911|ambulance)\b',
    r'\b(difficulty.?breathing|can.?t.?breathe|choking)\b',
    r'\b(unconscious|passed.?out|seizure|convulsion)\b',
    r'\b(severe.?bleeding|heavy.?bleeding|uncontrolled)\b',
]

# Medical diagnosis requests (should be redirected)
DIAGNOSIS_PATTERNS = [
    r'\b(diagnose|diagnosis|what.?do.?i.?have|what.?is.?wrong)\b',
    r'\b(am.?i.?sick|do.?i.?have|test.?results|lab.?results)\b',
]


def check_safety(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if query is safe and appropriate.
    
    Args:
        query: User's question
        
    Returns:
        Tuple of (is_safe, reason, severity)
        - is_safe: True if query should be processed
        - reason: Explanation if blocked
        - severity: 'blocked', 'warning', or None
    """
    if not query or not query.strip():
        return False, "Empty query", "blocked"
    
    query_lower = query.lower().strip()
    
    # Check for blocked patterns (explicit, violence, illegal)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return False, "This query contains inappropriate content and cannot be processed.", "blocked"
    
    # Check for out-of-scope patterns (non-medical topics)
    out_of_scope_matches = []
    strong_non_medical = False
    
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            out_of_scope_matches.append(pattern)
            # Strong non-medical indicators (sports, coding, entertainment)
            if any(p in pattern for p in ['sports', 'game', 'score', 'coding', 'programming', 'python', 'movie', 'film']):
                strong_non_medical = True
    
    # Block immediately if strong non-medical indicator OR multiple matches
    # This prevents slow retrieval/QA on clearly non-medical queries
    if strong_non_medical or len(out_of_scope_matches) >= 2:
        return False, "This question isn't related to health or wellness. Please ask a health-related question so I can help you.", "blocked"
    
    # Also check if query has NO health-related keywords and has out-of-scope matches
    if len(out_of_scope_matches) >= 1 and not is_health_related(query):
        return False, "This question isn't related to health or wellness. Please ask a health-related question so I can help you.", "blocked"
    
    # Check for emergency keywords (warn but allow)
    emergency_found = False
    for pattern in EMERGENCY_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            emergency_found = True
            break
    
    # Check for diagnosis requests (warn but allow)
    diagnosis_request = False
    for pattern in DIAGNOSIS_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            diagnosis_request = True
            break
    
    # Return warnings if needed
    if emergency_found:
        return True, "If this is a medical emergency, please call 911 or go to the nearest emergency room immediately.", "warning"
    
    if diagnosis_request:
        return True, "I cannot provide medical diagnoses. Please consult a healthcare professional for diagnosis and treatment.", "warning"
    
    return True, None, None


def is_health_related(query: str) -> bool:
    """Quick check if query is health-related."""
    health_keywords = [
        r'\b(health|medical|symptom|disease|condition|illness|pain|treatment|medication|doctor|hospital)\b',
        r'\b(diabetes|hypertension|asthma|pneumonia|fever|cough|headache|nausea|vomiting)\b',
        r'\b(wellness|fitness|nutrition|diet|exercise|mental.?health|therapy)\b',
    ]
    
    query_lower = query.lower()
    for pattern in health_keywords:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    
    return False

