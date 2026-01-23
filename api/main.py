"""
FastAPI service for Customer Purchase Prediction.

This service provides REST API endpoints for predicting customer purchase likelihood
based on browsing behavior data.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import sys

# Add src directory to path for importing local modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import PurchasePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Purchase Prediction API",
    description="ML service for predicting customer purchase likelihood based on e-commerce browsing behavior",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global predictor instance
predictor = None


class CustomerData(BaseModel):
    """Input model for single customer prediction."""
    customer_id: int = Field(..., gt=0, description="Unique customer identifier")
    time_spent: float = Field(..., ge=0, description="Minutes spent on website")
    pages_viewed: int = Field(..., ge=0, description="Number of pages viewed")
    basket_value: float = Field(..., ge=0, description="Value of items in basket")
    device_type: str = Field(..., description="Device type: Mobile, Desktop, Tablet, Unknown")
    customer_type: str = Field(..., description="Customer type: New, Returning")

    @validator('device_type')
    def validate_device_type(cls, v):
        allowed = ['Mobile', 'Desktop', 'Tablet', 'Unknown']
        if v not in allowed:
            raise ValueError(f'device_type must be one of: {allowed}')
        return v

    @validator('customer_type')
    def validate_customer_type(cls, v):
        allowed = ['New', 'Returning']
        if v not in allowed:
            raise ValueError(f'customer_type must be one of: {allowed}')
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    customer_id: int
    purchase_prediction: int = Field(..., description="Predicted purchase (0=No, 1=Yes)")
    purchase_probability: float = Field(..., ge=0, le=1, description="Probability of purchase")
    prediction_confidence: str = Field(..., description="Confidence level: low, medium, high")


class BatchPredictionRequest(BaseModel):
    """Input model for batch predictions."""
    customers: List[CustomerData] = Field(..., min_items=1, max_items=1000,
                                         description="List of customer data (max 1000)")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_customers: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    version: str = Field(..., description="API version")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")


@app.on_event("startup")
async def startup_event():
    """Initialize the ML predictor on startup."""
    global predictor
    try:
        predictor = PurchasePredictor()
        logger.info("ML predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML predictor: {e}")
        raise RuntimeError(f"Service initialization failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.

    Returns:
        Health status, model loading status, and basic information
    """
    try:
        model_info = None
        model_loaded = False

        if predictor:
            model_loaded = True
            try:
                model_info = predictor.get_model_info()
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")

        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            version="1.0.0",
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_customer(customer: CustomerData):
    """
    Predict purchase likelihood for a single customer.

    Args:
        customer: Customer browsing behavior data

    Returns:
        Prediction result with probability and confidence
    """
    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not available"
            )

        # Convert Pydantic model to dict for prediction
        customer_dict = customer.dict()

        # Make prediction
        result = predictor.predict(customer_dict, return_probabilities=True)

        # Convert result to response model
        response = PredictionResponse(**result)

        logger.info(f"Prediction made for customer {customer.customer_id}: {response.purchase_prediction}")
        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed for customer {customer.customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to internal error"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_customers(request: BatchPredictionRequest):
    """
    Predict purchase likelihood for multiple customers.

    Args:
        request: Batch prediction request with list of customers

    Returns:
        Batch prediction results
    """
    import time
    start_time = time.time()

    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not available"
            )

        # Convert Pydantic models to list of dicts
        customers_data = [customer.dict() for customer in request.customers]

        # Make batch predictions
        predictions = predictor.predict(customers_data, return_probabilities=True)

        # Convert results to response models
        prediction_responses = [
            PredictionResponse(**pred) for pred in predictions
        ]

        processing_time = time.time() - start_time

        logger.info(f"Batch prediction completed for {len(prediction_responses)} customers in {processing_time:.2f}s")

        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_customers=len(prediction_responses),
            processing_time_seconds=round(processing_time, 3)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed due to internal error"
        )


@app.get("/")
async def root():
    """
    Root endpoint with basic API information.

    Returns:
        Welcome message and available endpoints
    """
    return {
        "message": "Customer Purchase Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health - Service health check",
            "predict": "/predict - Single customer prediction",
            "batch_predict": "/predict/batch - Batch customer predictions",
            "docs": "/docs - Interactive API documentation"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )