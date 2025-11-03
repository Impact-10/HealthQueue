"""Backend startup script"""

import os
import logging
from dotenv import load_dotenv
import uvicorn
from api.app import app
from utils.download_models import main as download_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup():
    """Setup environment and download models"""
    # Load environment variables
    load_dotenv()

    # Download models only if not in fallback mode
    if os.getenv("MEDALPACA_MODE") != "fallback":
        download_models()

def main():
    """Start the backend server"""
    try:
        setup()
        logger.info("Starting server...")
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")

if __name__ == "__main__":
    main()