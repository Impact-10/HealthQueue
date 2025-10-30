"""Script to download and setup required models"""

import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path("./model_cache")
MODEL_CACHE_DIR.mkdir(exist_ok=True)

def download_model(model_name: str, model_type: str = "base"):
    """Download and cache a model"""
    try:
        logger.info(f"Downloading {model_name}...")
        
        if model_type == "longformer":
            tokenizer = LongformerTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR
            )
            model = LongformerForQuestionAnswering.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR
            )
            if model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=MODEL_CACHE_DIR,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=MODEL_CACHE_DIR
                )
        
        logger.info(f"Successfully downloaded {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        return False

def main():
    """Download all required models"""
    models = [
        ("medalpaca/medalpaca-7b", "causal"),
        ("microsoft/biogpt", "causal"),
        ("yikuan8/Clinical-Longformer", "longformer"),
        ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "base")
    ]
    
    success = True
    for model_name, model_type in models:
        if not download_model(model_name, model_type):
            success = False
    
    if success:
        logger.info("All models downloaded successfully!")
    else:
        logger.warning("Some models failed to download. Check logs for details.")

if __name__ == "__main__":
    main()