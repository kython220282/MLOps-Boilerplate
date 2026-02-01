"""FastAPI server for model serving."""
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import time

from ml_service.applications.inference import InferenceService
from ml_service.monitoring.model_monitor import ModelMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Service API",
    description="Production ML model serving API",
    version="0.1.0"
)

# Global variables (will be initialized on startup)
inference_service: Optional[InferenceService] = None
monitor: Optional[ModelMonitor] = None
model_name = "default_model"
model_version = "v1"


# Request/Response models
class PredictionRequest(BaseModel):
    """Prediction request schema."""
    features: Dict[str, Any] = Field(..., description="Input features")
    return_probabilities: bool = Field(default=False, description="Return prediction probabilities")


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: Any
    probabilities: Optional[List[float]] = None
    model_name: str
    model_version: str
    timestamp: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global inference_service, monitor
    
    try:
        # Load model (path should come from config/environment)
        model_path = "models/model.joblib"
        inference_service = InferenceService(model_path)
        
        # Initialize monitoring
        monitor = ModelMonitor()
        
        logger.info("ML Service API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Service API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inference_service is not None else "unhealthy",
        model_loaded=inference_service is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Make a prediction.
    
    Args:
        request: Prediction request
        background_tasks: Background tasks for async logging
        
    Returns:
        Prediction response
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Make prediction
        prediction = inference_service.predict(request.features)
        
        # Get probabilities if requested
        probabilities = None
        if request.return_probabilities:
            try:
                probabilities = inference_service.predict_proba(request.features).tolist()
            except:
                pass
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # ms
        
        # Log prediction asynchronously
        if monitor:
            background_tasks.add_task(
                monitor.log_prediction,
                model_name=model_name,
                model_version=model_version,
                input_data=request.features,
                prediction=prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                latency=latency / 1000
            )
        
        return PredictionResponse(
            prediction=prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            probabilities=probabilities,
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(features_list: List[Dict[str, Any]]):
    """
    Make batch predictions.
    
    Args:
        features_list: List of feature dictionaries
        
    Returns:
        List of predictions
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(features_list)
        predictions = inference_service.predict(df)
        
        return {
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    if monitor:
        metrics = monitor.get_metrics_prometheus()
        return Response(content=metrics, media_type="text/plain")
    else:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")


@app.get("/stats")
async def get_stats(hours: int = 24):
    """
    Get prediction statistics.
    
    Args:
        hours: Time window in hours
        
    Returns:
        Statistics dictionary
    """
    if monitor:
        stats = monitor.get_prediction_stats(model_name, time_window_hours=hours)
        return stats
    else:
        raise HTTPException(status_code=503, detail="Monitoring not initialized")


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "ml_service.applications.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
