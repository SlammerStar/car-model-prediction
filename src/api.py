"""
FastAPI Backend for Car Price Prediction
========================================

REST API providing car price prediction endpoints.

Endpoints:
    POST /predict  - Predict the price of a single car
    GET  /health   - Health check
    GET  /brands   - Get available brands
    GET  /models/{brand} - Get models for a specific brand
"""

import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from src.prediction import predict_price
from src.utils import (
    PIPELINE_PATH,
    BRAND_FILE_MAP,
    load_model,
    logger,
)
from src.data_processing import load_and_merge_datasets

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DRIVEIQ - AI Valuation Engine API",
    description=(
        "Predict premium used car prices across multiple brands. "
        "Built with scikit-learn pipelines, featuring advanced ML models and market adaptations."
    ),
    version="1.0.0",
    contact={
        "name": "Pratham Nigam",
    },
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline at startup
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Load the ML pipeline on application startup."""
    global pipeline
    try:
        pipeline = load_model(PIPELINE_PATH)
        logger.info("Pipeline loaded successfully at startup.")
    except FileNotFoundError:
        logger.warning(
            "Model not found. Run 'python -m src.prediction' first. "
            "API will return errors until model is available."
        )


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class CarInput(BaseModel):
    """Input schema for car price prediction."""

    brand: str = Field(..., description="Car brand name")
    model: str = Field(..., description="Car model name")
    year: int = Field(..., ge=1990, le=2026, description="Manufacturing year")
    transmission: str = Field(..., description="Transmission type")
    mileage: int = Field(..., ge=0, description="Odometer reading in miles")
    fuelType: str = Field(..., description="Fuel type")
    mpg: float = Field(..., ge=0, description="Miles per gallon")
    engineSize: float = Field(..., ge=0, description="Engine size in litres")

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand": "BMW",
                "model": "X5",
                "year": 2019,
                "transmission": "Automatic",
                "mileage": 45000,
                "fuelType": "Diesel",
                "mpg": 52.3,
                "engineSize": 2.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for car price prediction."""

    predicted_price: str = Field(..., description="Predicted price in INR")
    predicted_price_raw: float = Field(..., description="Raw predicted price in INR")
    price_range: str = Field(..., description="Estimated price range")
    original_price: str = Field(..., description="Simulated original price")
    depreciation_percent: str = Field(..., description="Estimated depreciation percentage")
    confidence: str = Field(..., description="Prediction confidence score")
    recommendations: list = Field(default=[], description="Similar vehicle recommendations")
    input_summary: dict = Field(..., description="Summary of input features used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DRIVEIQ - AI Valuation Engine API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check if the API and model are healthy."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict car price",
    description="Predict the price of a used car in Indian Rupees (₹).",
)
async def predict(car: CarInput):
    """
    Predict the price of a used car.

    Returns the predicted price in Indian Lakhs notation.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first using 'python -m src.prediction'.",
        )

    try:
        result = predict_price(
            brand=car.brand,
            model=car.model,
            year=car.year,
            transmission=car.transmission,
            mileage=car.mileage,
            fuel_type=car.fuelType,
            mpg=car.mpg,
            engine_size=car.engineSize,
            pipeline=pipeline,
        )

        return PredictionResponse(
            predicted_price=result["predicted_price"],
            predicted_price_raw=result["predicted_price_raw"],
            price_range=result["price_range"],
            original_price=result["original_price"],
            depreciation_percent=result["depreciation_percent"],
            confidence=result["confidence"],
            recommendations=result.get("recommendations", []),
            input_summary=result["input_summary"],
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/brands", tags=["Data"])
async def get_brands():
    """Get the list of available car brands."""
    return {"brands": list(BRAND_FILE_MAP.keys())}


@app.get("/models/{brand}", tags=["Data"])
async def get_models_for_brand(brand: str):
    """Get available car models for a specific brand."""
    try:
        df = load_and_merge_datasets()
        brand_df = df[df["brand"].str.lower() == brand.lower()]
        if brand_df.empty:
            raise HTTPException(
                status_code=404, detail=f"Brand '{brand}' not found."
            )
        models = sorted(brand_df["model"].unique().tolist())
        return {"brand": brand, "models": models}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
