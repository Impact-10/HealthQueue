import os
import sys
import json

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for testing
os.environ["BIOGPT_MODE"] = "inference"

# Import directly from the models module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.biogpt import BioGPTModel

# Test the BioGPT model
model = BioGPTModel()

prompts = [
    "i have high blood sugar and feel tired",
    "i have nocturia",
    "i have itchy skin",
    "i have thirst",
    "i have a headache",
    ""
]

for prompt in prompts:
    print(f"\nTesting BioGPT with prompt: '{prompt}'")

    response = model.generate_response(prompt)

    print("\n=== BioGPT Response ===\n")
    print("Content:", json.dumps(response.get("content", {}), indent=2))
    print("Metadata:", json.dumps(response.get("metadata", {}), indent=2))
    print("Warnings:", response.get("warnings", []))
