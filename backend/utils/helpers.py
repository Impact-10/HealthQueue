"""Utility functions for the HealthQueue backend"""

import json
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

def preprocess_medical_text(text: str) -> str:
    """Preprocess medical text for model input"""
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Add medical context markers
    text = f"[MEDICAL CONTEXT] {text}"
    
    return text

def postprocess_model_output(
    output: str,
    model_name: str,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """Postprocess model outputs for consistent format"""
    try:
        # Basic cleaning
        output = output.strip()
        
        # Add metadata
        result = {
            "text": output,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "confidence": np.random.uniform(0.7, 0.99)  # Placeholder
        }
        
        # Filter low confidence results
        if result["confidence"] < confidence_threshold:
            result["warning"] = "Low confidence prediction"
        
        return result
    
    except Exception as e:
        return {
            "error": str(e),
            "text": output,
            "model": model_name
        }

def validate_medical_entities(
    entities: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Validate extracted medical entities"""
    valid_categories = {
        "DISEASE", "DRUG", "SYMPTOM", "TREATMENT",
        "TEST", "ANATOMY", "PROCEDURE"
    }
    
    return {
        k: v for k, v in entities.items()
        if k in valid_categories
    }

def format_diagnosis_response(
    diagnosis: Dict[str, Any],
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Format diagnosis response for API output"""
    response = {
        "diagnosis": diagnosis.get("text", ""),
        "confidence": diagnosis.get("confidence", 0.0),
        "timestamp": diagnosis.get("timestamp", datetime.now().isoformat())
    }
    
    if include_metadata:
        response["metadata"] = {
            "model": diagnosis.get("model", "unknown"),
            "processing_time": diagnosis.get("processing_time", 0),
            "warnings": diagnosis.get("warnings", [])
        }
    
    return response

def analyze_population_trends(
    diagnoses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze trends in population health data"""
    if not diagnoses:
        return {"error": "No diagnoses provided"}
    
    try:
        # Count conditions
        conditions = {}
        severities = {"Low": 0, "Medium": 0, "High": 0}
        
        for diagnosis in diagnoses:
            # Count severity
            severity = diagnosis.get("severity", "Unknown")
            if severity in severities:
                severities[severity] += 1
            
            # Extract and count conditions
            if "diagnosis_text" in diagnosis:
                # Simple keyword extraction (replace with better NLP)
                words = diagnosis["diagnosis_text"].split()
                for word in words:
                    if word in conditions:
                        conditions[word] += 1
                    else:
                        conditions[word] = 1
        
        # Get top conditions
        sorted_conditions = sorted(
            conditions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "top_conditions": sorted_conditions,
            "severity_distribution": severities,
            "total_records": len(diagnoses),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {"error": str(e)}

def generate_health_alert(
    trend_data: Dict[str, Any],
    threshold: float = 0.7
) -> Optional[Dict[str, Any]]:
    """Generate health alerts based on trend analysis"""
    try:
        alerts = []
        severity_dist = trend_data.get("severity_distribution", {})
        total = sum(severity_dist.values())
        
        if total == 0:
            return None
        
        # Check for high severity cases
        high_ratio = severity_dist.get("High", 0) / total
        if high_ratio > threshold:
            alerts.append({
                "level": "HIGH",
                "message": "Unusually high number of severe cases detected",
                "ratio": high_ratio
            })
        
        # Check for condition clusters
        top_conditions = trend_data.get("top_conditions", [])
        if top_conditions:
            top_condition_count = top_conditions[0][1]
            if top_condition_count / total > threshold:
                alerts.append({
                    "level": "MEDIUM",
                    "message": f"Potential cluster of {top_conditions[0][0]} detected",
                    "ratio": top_condition_count / total
                })
        
        if alerts:
            return {
                "alerts": alerts,
                "timestamp": datetime.now().isoformat(),
                "data_points": total
            }
        
        return None
    
    except Exception as e:
        return {"error": str(e)}

def calculate_facility_load(
    diagnoses: List[Dict[str, Any]],
    facility_capacity: Dict[str, int]
) -> Dict[str, Any]:
    """Calculate healthcare facility load based on diagnoses"""
    try:
        facility_load = {
            facility: {"current": 0, "capacity": capacity}
            for facility, capacity in facility_capacity.items()
        }
        
        # Calculate load based on severity
        for diagnosis in diagnoses:
            severity = diagnosis.get("severity", "Low")
            if severity == "High":
                facility_load["emergency"]["current"] += 1
            elif severity == "Medium":
                facility_load["urgent_care"]["current"] += 1
            else:
                facility_load["general"]["current"] += 1
        
        # Calculate utilization percentages
        for facility in facility_load:
            current = facility_load[facility]["current"]
            capacity = facility_load[facility]["capacity"]
            facility_load[facility]["utilization"] = (current / capacity) * 100
        
        return facility_load
    
    except Exception as e:
        return {"error": str(e)}