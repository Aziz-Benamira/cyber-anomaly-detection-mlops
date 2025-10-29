"""
FastAPI Application for MoE Model Serving
=========================================

Endpoints:
- GET  /health          - Health check
- POST /predict         - Single prediction
- POST /predict_batch   - Batch predictions
- GET  /metrics         - Prometheus metrics
- GET  /model_info      - Model metadata
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
import mlflow.pyfunc
from pathlib import Path
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MoE Cybersecurity API",
    description="Mixture-of-Experts model for network anomaly detection",
    version="1.0.0"
)

# Global model variable
model = None
model_info = {}

# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    handle_missing: bool = True
    missing_strategy: str = "zero"
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "Flow Duration": 1200,
                    "Total Fwd Packets": 250,
                    "Flow IAT Mean": 4.8,
                    "SYN Flag Count": 250,
                    "ACK Flag Count": 0
                },
                "handle_missing": True,
                "missing_strategy": "zero"
            }
        }

class BatchPredictionRequest(BaseModel):
    features: List[Dict[str, float]]
    handle_missing: bool = True
    missing_strategy: str = "zero"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    gating_weights: Optional[Dict[str, float]] = None
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_time_ms: float
    average_time_ms: float

# ============================================================================
# Startup Event - Load Model
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, model_info
    
    logger.info("Loading model...")
    
    # Always use local inference class (handles MoE properly)
    model_path = Path("models/weights/cicids_moe_best.pt")
    
    if model_path.exists():
        logger.info(f"Loading from local file: {model_path}")
        
        # Import inference module
        from src.serving.inference import MoEInference
        model = MoEInference('CICIDS', str(model_path))
        model_info = {
            "source": "local_inference",
            "model_path": str(model_path),
            "stage": "Production",
            "note": "Using MoEInference class for proper MoE handling"
        }
        logger.info("✅ Model loaded successfully")
    else:
        logger.error(f"❌ Model not found at: {model_path}")
        logger.error("Please train the model first or check the path.")
        model_info = {"source": "none", "error": f"Model not found at {model_path}"}

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_info
    }

# ============================================================================
# Model Info
# ============================================================================

@app.get("/model_info")
async def get_model_info():
    """Get model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "moe-cybersecurity-cicids",
        "architecture": "MoE (FT-Transformer + 1D-CNN + Gating)",
        "dataset": "CICIDS",
        "expected_features": 72,
        **model_info
    }

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single sample prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Use MoEInference class
        result = model.predict(
            request.features,
            handle_missing=request.handle_missing,
            missing_strategy=request.missing_strategy
        )
        
        response = {
            **result,
            "inference_time_ms": (time.time() - start_time) * 1000
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        predictions = []
        
        for features in request.features:
            pred_start = time.time()
            
            # Use MoEInference class
            result = model.predict(
                features,
                handle_missing=request.handle_missing,
                missing_strategy=request.missing_strategy
            )
            pred_response = {
                **result,
                "inference_time_ms": (time.time() - pred_start) * 1000
            }
            
            predictions.append(pred_response)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": predictions,
            "total_time_ms": total_time,
            "average_time_ms": total_time / len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Prometheus Metrics (Simple Counter)
# ============================================================================

# Simple in-memory metrics
metrics = {
    "predictions_total": 0,
    "predictions_attack": 0,
    "predictions_normal": 0,
    "errors_total": 0
}

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-style metrics."""
    return {
        "# TYPE predictions_total counter": metrics["predictions_total"],
        "# TYPE predictions_attack counter": metrics["predictions_attack"],
        "# TYPE predictions_normal counter": metrics["predictions_normal"],
        "# TYPE errors_total counter": metrics["errors_total"]
    }

# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "MoE Cybersecurity API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict_batch",
            "info": "/model_info",
            "metrics": "/metrics"
        }
    }

# ============================================================================
# Run with: uvicorn src.serving.api:app --reload --port 8000
# ============================================================================
