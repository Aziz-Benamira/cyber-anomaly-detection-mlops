"""
Enhanced API with Prometheus Metrics
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
import json
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.serving.inference import MoEInference

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ============================================================================
# Prometheus Metrics
# ============================================================================
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total API requests',
    ['endpoint', 'method']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['prediction']
)

PREDICTION_TIME = Histogram(
    'prediction_duration_seconds',
    'Time spent making predictions',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

CONFIDENCE_GAUGE = Gauge(
    'prediction_confidence',
    'Confidence of last prediction',
    ['prediction']
)

EXPERT_WEIGHT_GAUGE = Gauge(
    'expert_gating_weight',
    'Expert gating weights',
    ['expert']
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether model is loaded (1=loaded, 0=not loaded)'
)

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="MoE Cybersecurity Detection API",
    description="Network anomaly detection using Mixture-of-Experts model",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Load Model
# ============================================================================
print("🔄 Loading MoE model...")
try:
    model_path = "models/weights/cicids_moe_best.pt"
    
    inference = MoEInference(
        dataset='CICIDS',
        model_path=model_path,
        device='cpu'
    )
    MODEL_LOADED.set(1)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    MODEL_LOADED.set(0)
    inference = None

# ============================================================================
# Pydantic Models
# ============================================================================
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    gating_weights: Dict[str, float]
    inference_time_ms: float

class BatchPredictionRequest(BaseModel):
    samples: List[Dict[str, float]]

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    REQUEST_COUNT.labels(endpoint='/', method='GET').inc()
    return {
        "name": "MoE Cybersecurity Detection API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict_batch",
            "model_info": "/model_info",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(endpoint='/health', method='GET').inc()
    return {
        "status": "healthy",
        "model_loaded": inference is not None,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    REQUEST_COUNT.labels(endpoint='/predict', method='POST').inc()
    
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    with PREDICTION_TIME.time():
        # Enable handle_missing to work with preprocessed features
        result = inference.predict(request.features, handle_missing=True)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Update metrics
    PREDICTION_COUNT.labels(prediction=result['prediction']).inc()
    CONFIDENCE_GAUGE.labels(prediction=result['prediction']).set(result['confidence'])
    
    for expert, weight in result['gating_weights'].items():
        expert_name = expert.replace('_', ' ').title()
        EXPERT_WEIGHT_GAUGE.labels(expert=expert_name).set(weight)
    
    return {
        "prediction": result['prediction'],
        "confidence": result['confidence'],
        "probabilities": result['probabilities'],
        "gating_weights": {k.replace('_', ' ').title(): v for k, v in result['gating_weights'].items()},
        "inference_time_ms": inference_time
    }

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    REQUEST_COUNT.labels(endpoint='/predict_batch', method='POST').inc()
    
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for sample in request.samples:
        result = inference.predict(sample)
        results.append(result)
        PREDICTION_COUNT.labels(prediction=result['prediction']).inc()
    
    return {"predictions": results, "count": len(results)}

@app.get("/model_info")
async def model_info():
    REQUEST_COUNT.labels(endpoint='/model_info', method='GET').inc()
    
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "MoE (Mixture-of-Experts)",
        "architecture": {
            "tabular_expert": "FT-Transformer (47 features)",
            "temporal_expert": "1D-CNN (25 features)",
            "gating": "Dynamic weighting",
            "classifier": "Dense layers"
        },
        "performance": {
            "f1_score": 0.9835,
            "precision": 0.9716,
            "recall": 0.9956,
            "auc_pr": 0.9991
        },
        "parameters": 733237,
        "input_features": 72
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
