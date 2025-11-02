# FastAPI route for classification with REAL H5 models
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import tempfile
import os
from PIL import Image
from real_model_classifier import OtolithRealClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Otolith Wild vs Hatchery Classifier",
    description="Real AI classification using trained H5 models from SEANOE dataset",
    version="2.0.0"
)

# Initialize the real classifier
try:
    classifier = OtolithRealClassifier()
    logger.info("✅ Real otolith classifier initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize classifier: {e}")
    classifier = None

@app.get("/")
async def root():
    return {
        "message": "Otolith Wild vs Hatchery Classifier - ML Project #2",
        "status": "operational" if classifier else "error",
        "models_loaded": classifier.get_model_info() if classifier else {},
        "description": "Real AI classification using trained neural networks"
    }

@app.post("/classify")
async def classify_otolith(file: UploadFile = File(...)):
    """
    Classify otolith image as wild or hatchery origin
    Uses real trained H5 models from SEANOE dataset
    """
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not initialized")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Classify using real models
            result = classifier.classify_wild_vs_hatchery(tmp_file_path)
            
            # Add metadata
            result["filename"] = file.filename
            result["file_size_bytes"] = len(image_data)
            result["model_version"] = "SEANOE_H5_v1.0"
            
            logger.info(f"✅ Classified {file.filename}: {result['primary_prediction']['origin']}")
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        logger.error(f"❌ Classification error for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not initialized")
    
    return classifier.get_model_info()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if classifier else "unhealthy",
        "models_available": classifier is not None,
        "service": "ML_Project_2_Otolith_Classifier"
    }
