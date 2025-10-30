from typing import List, Dict, Any, Optional
import numpy as np
from .medalpaca import MedAlpacaModel
from .biogpt import BioGPTModel
from .clinical_longformer import ClinicalLongformerModel
from .pubmedbert import PubMedBERTModel


class EnsembleModel:
    """Ensemble model combining multiple medical AI models"""
    
    def __init__(self):
        self.models = {
            "medalpaca": MedAlpacaModel(),
            "biogpt": BioGPTModel(),
            "longformer": ClinicalLongformerModel(),
            "pubmedbert": PubMedBERTModel()
        }
        self.weights = {
            "medalpaca": 0.3,
            "biogpt": 0.3,
            "longformer": 0.2,
            "pubmedbert": 0.2
        }

    def get_ensemble_diagnosis(
        self,
        symptoms: str,
        medical_history: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get diagnosis from all models and ensemble the results"""
        try:
            results = {}
            
            # Get diagnosis from MedAlpaca
            medalpaca_input = {
                "symptoms": symptoms,
                "history": medical_history or "None",
            }
            results["medalpaca"] = self.models["medalpaca"].generate_response(
                f"Symptoms: {medalpaca_input['symptoms']}\nMedical History: {medalpaca_input['history']}"
            )

            # Get diagnosis from BioGPT
            results["biogpt"] = self.models["biogpt"].generate_diagnosis(
                symptoms, medical_history
            )

            # Get insights from Clinical-Longformer
            context = f"Symptoms: {symptoms}\nMedical History: {medical_history or 'None'}"
            results["longformer"] = self.models["longformer"].extract_clinical_insights(context)

            # Get entity analysis from PubMedBERT
            results["pubmedbert"] = self.models["pubmedbert"].extract_entities(
                context
            )

            # Ensemble the results
            ensemble_result = self._combine_results(results)

            return {
                "model": "ensemble",
                "individual_results": results,
                "ensemble_result": ensemble_result,
                "metadata": {
                    "weights": self.weights,
                    "models_used": list(self.models.keys())
                }
            }

        except RuntimeError:
            raise
        except Exception as e:
            return {
                "error": str(e),
                "model": "ensemble",
                "message": "Error in ensemble diagnosis generation."
            }

    def _combine_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        combined: Dict[str, Any] = {
            "summaries": [],
            "possible_causes": [],
            "recommendations": [],
            "warning_signs": [],
            "entities": {},
            "record_analysis": [],
            "modes": {},
        }

        for name, response in results.items():
            combined["modes"][name] = response.get("mode")
            content = response.get("content", {}) or {}

            if name == "medalpaca":
                combined["summaries"].append(content.get("summary", ""))
                combined["possible_causes"].extend(content.get("possible_causes", []))
                combined["recommendations"].extend(content.get("recommendations", []))
                combined["warning_signs"].extend(content.get("warning_signs", []))
            elif name == "biogpt":
                combined["summaries"].append(content.get("summary", ""))
                combined.setdefault("differential", []).extend(content.get("possible_diagnoses", []))
                combined["recommendations"].extend(content.get("next_steps", []))
                combined["warning_signs"].extend(content.get("warning_signs", []))
            elif name == "longformer":
                combined["record_analysis"] = content.get("record_analysis", [])
            elif name == "pubmedbert":
                entities = content.get("entities", {})
                for etype, values in entities.items():
                    combined["entities"].setdefault(etype, set()).update(values)

        combined["entities"] = {
            etype: sorted(values) for etype, values in combined["entities"].items()
        }
        combined["primary_entities"] = self._get_primary_entities(combined["entities"])
        combined["summaries"] = [s for s in combined["summaries"] if s]
        combined["possible_causes"] = sorted(set(combined["possible_causes"]))
        combined["recommendations"] = sorted(set(combined["recommendations"]))
        combined["warning_signs"] = sorted(set(combined["warning_signs"]))

        return combined

    def _get_primary_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Get the most significant entities from each category"""
        primary = {}
        for category, items in entities.items():
            unique_items = list(items)
            if unique_items:
                primary[category] = unique_items[:3]
        return primary

    def analyze_population_health(
        self,
        diagnoses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze population health trends from multiple diagnoses"""
        try:
            # Extract entities across all diagnoses
            all_entities = {
                "DISEASE": [],
                "SYMPTOM": [],
                "TREATMENT": []
            }

            for diagnosis in diagnoses:
                if "ensemble_result" in diagnosis:
                    ent_map = diagnosis["ensemble_result"].get("entities", {})
                    for entity_type in all_entities:
                        if entity_type in ent_map:
                            all_entities[entity_type].extend(ent_map[entity_type])

            # Calculate trends
            trends = {}
            for entity_type, entities in all_entities.items():
                if entities:
                    # Get frequency distribution
                    unique, counts = np.unique(entities, return_counts=True)
                    freq_dist = dict(zip(unique, counts))
                    
                    # Sort by frequency
                    sorted_items = sorted(
                        freq_dist.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    trends[entity_type] = {
                        "top_items": sorted_items[:5],
                        "total_occurrences": len(entities),
                        "unique_count": len(unique)
                    }

            return {
                "trends": trends,
                "sample_size": len(diagnoses),
                "metadata": {
                    "analysis_timestamp": "2025-10-28",  # Add actual timestamp
                    "confidence": 0.85
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": "Error analyzing population health trends"
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models in the ensemble"""
        return {
            "name": "Medical AI Ensemble",
            "mode": "aggregate",
            "version": "1.0",
            "models": {
                name: model.get_model_info()
                for name, model in self.models.items()
            },
            "weights": self.weights,
            "capabilities": [
                "Multi-model diagnosis",
                "Entity extraction",
                "Clinical analysis",
                "Population health trends"
            ]
        }