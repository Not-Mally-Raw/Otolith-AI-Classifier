"""
Simplified version of the Otolith AI Classifier
This script provides a minimal FastAPI application for otolith classification
"""

import torch
from fastapi import FastAPI, UploadFile, File
from model import OtolithResNet50
import uvicorn

# Create FastAPI app
app = FastAPI(title="Otolith AI Classifier", description="Classifies otolith images to identify fish species")

# Dummy class list (can be extended later)
SPECIES_LIST = [
    "Lutjanus argentimaculatus",  # Mangrove red snapper
    "Lutjanus johnii",            # Golden snapper
    "Scomberomorus commerson",    # Narrow-barred Spanish mackerel
    "Scomberomorus guttatus",     # Indo-Pacific king mackerel
    "Rastrelliger kanagurta"      # Indian mackerel
]

# Initialize model (create dummy if real one fails to load)
try:
    model = OtolithResNet50(num_classes=len(SPECIES_LIST))
    print("Model initialized successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.post("/classify")
async def classify_otolith(file: UploadFile = File(...)):
    """
    Classify an otolith image
    """
    try:
        # For demonstration, return fixed result
        species = SPECIES_LIST[0]
        confidence = 0.95
        
        return {
            "species": species,
            "confidence": confidence,
            "status": "success",
            "message": "Classification successful (demo mode)"
        }
    except Exception as e:
        return {
            "species": "Unknown",
            "confidence": 0.0,
            "status": "error",
            "message": f"Error during classification: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """
    Service health check
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "species_supported": len(SPECIES_LIST)
    }

@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "service": "Otolith AI Classifier",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/classify", "method": "POST", "description": "Classify an otolith image"},
            {"path": "/health", "method": "GET", "description": "Service health check"}
        ]
    }

if __name__ == "__main__":
    print("Starting Otolith AI Classifier Service...")
    uvicorn.run(app, host="0.0.0.0", port=8002)