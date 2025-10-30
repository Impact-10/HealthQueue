"""Script to generate synthetic medical datasets for model testing and evaluation"""

import json
import random
from typing import List, Dict, Any
import pandas as pd
from faker import Faker
import google.generativeai as genai
from datetime import datetime, timedelta

# Configure Faker
fake = Faker()

# Configure Google Generative AI
genai.configure(api_key='your-api-key-here')

class DatasetGenerator:
    """Generate synthetic medical datasets"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Common medical conditions and symptoms
        self.conditions = [
            "Hypertension", "Diabetes Type 2", "Asthma", "Migraine",
            "Anxiety", "Depression", "GERD", "Arthritis"
        ]
        self.symptoms = [
            "headache", "fever", "cough", "fatigue",
            "shortness of breath", "chest pain", "nausea",
            "dizziness", "joint pain"
        ]

    def generate_patient_profile(self) -> Dict[str, Any]:
        """Generate a synthetic patient profile"""
        return {
            "patient_id": fake.uuid4(),
            "age": random.randint(18, 85),
            "gender": random.choice(["Male", "Female"]),
            "blood_type": random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
            "weight_kg": round(random.uniform(50, 100), 1),
            "height_cm": random.randint(150, 190),
            "existing_conditions": random.sample(self.conditions, random.randint(0, 3))
        }

    async def generate_symptom_query(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a synthetic symptom query"""
        # Create context for Gemini
        prompt = f"""
        Generate a realistic medical symptom description for a patient with the following profile:
        Age: {profile['age']}
        Gender: {profile['gender']}
        Existing Conditions: {', '.join(profile['existing_conditions'])}

        Format the response as a first-person description from the patient's perspective.
        Include:
        1. Main symptoms
        2. Duration
        3. Severity
        4. Any relevant history
        """

        response = await self.model.generate_content(prompt)
        
        return {
            "query_id": fake.uuid4(),
            "patient_id": profile["patient_id"],
            "timestamp": datetime.now().isoformat(),
            "symptoms_description": response.text,
            "primary_symptoms": random.sample(self.symptoms, random.randint(2, 4))
        }

    async def generate_diagnosis(self, query: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a synthetic diagnosis"""
        prompt = f"""
        Generate a medical diagnosis based on:
        
        Patient Profile:
        - Age: {profile['age']}
        - Gender: {profile['gender']}
        - Existing Conditions: {', '.join(profile['existing_conditions'])}
        
        Symptoms:
        {query['symptoms_description']}
        
        Provide:
        1. Primary diagnosis
        2. Differential diagnoses
        3. Recommended tests
        4. Treatment plan
        5. Follow-up recommendations
        """

        response = await self.model.generate_content(prompt)
        
        return {
            "diagnosis_id": fake.uuid4(),
            "query_id": query["query_id"],
            "patient_id": profile["patient_id"],
            "timestamp": datetime.now().isoformat(),
            "diagnosis_text": response.text,
            "severity": random.choice(["Low", "Medium", "High"]),
            "confidence": round(random.uniform(0.7, 0.99), 2)
        }

    async def generate_conversation(self, query: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a synthetic medical conversation"""
        prompt = f"""
        Generate a realistic medical conversation between a doctor and a patient with:
        
        Patient's Description:
        {query['symptoms_description']}
        
        Include:
        1. Doctor's questions
        2. Patient's responses
        3. Follow-up questions
        4. Final recommendations
        
        Format as a list of alternating messages.
        """

        response = await self.model.generate_content(prompt)
        
        return {
            "conversation_id": fake.uuid4(),
            "query_id": query["query_id"],
            "patient_id": profile["patient_id"],
            "timestamp": datetime.now().isoformat(),
            "messages": response.text.split("\n"),
            "duration_minutes": random.randint(5, 30)
        }

    def save_dataset(self, dataset: Dict[str, List[Dict]], format: str = "json"):
        """Save generated dataset to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            with open(f"medical_dataset_{timestamp}.json", "w") as f:
                json.dump(dataset, f, indent=2)
        
        elif format == "csv":
            for key, data in dataset.items():
                df = pd.DataFrame(data)
                df.to_csv(f"{key}_{timestamp}.csv", index=False)

async def generate_complete_dataset(num_samples: int = 500) -> Dict[str, List[Dict]]:
    """Generate a complete synthetic medical dataset"""
    generator = DatasetGenerator()
    dataset = {
        "profiles": [],
        "queries": [],
        "diagnoses": [],
        "conversations": []
    }

    for _ in range(num_samples):
        # Generate profile
        profile = generator.generate_patient_profile()
        dataset["profiles"].append(profile)

        # Generate query
        query = await generator.generate_symptom_query(profile)
        dataset["queries"].append(query)

        # Generate diagnosis
        diagnosis = await generator.generate_diagnosis(query, profile)
        dataset["diagnoses"].append(diagnosis)

        # Generate conversation
        conversation = await generator.generate_conversation(query, profile)
        dataset["conversations"].append(conversation)

    return dataset

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Generating synthetic medical dataset...")
        dataset = await generate_complete_dataset(500)
        
        generator = DatasetGenerator()
        generator.save_dataset(dataset, format="json")
        generator.save_dataset(dataset, format="csv")
        
        print("Dataset generation complete!")

    asyncio.run(main())