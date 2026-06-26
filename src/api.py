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

from src.prediction import predict_price
from src.utils import (
    PIPELINE_PATH,
    load_model,
    logger,
)

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
    version="2.0.0",
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

# Global instances
pipeline = None
knowledge_engine = None


@app.on_event("startup")
async def startup_event():
    """Load the ML pipeline and Knowledge Engine on application startup."""
    global pipeline
    global knowledge_engine

    # Load ML pipeline
    try:
        pipeline = load_model(PIPELINE_PATH)
        logger.info("Pipeline loaded successfully at startup.")
    except FileNotFoundError:
        logger.warning(
            "Model not found. Run 'python -m src.prediction' first. "
            "API will return errors until model is available."
        )

    # Load Knowledge Engine
    try:
        from src.data_processing import (
            load_and_merge_datasets,
            clean_data,
            convert_price_to_inr,
            create_features,
        )
        from src.knowledge_engine import VehicleKnowledgeEngine

        raw_df = load_and_merge_datasets()
        clean_df = clean_data(raw_df)
        conv_df = convert_price_to_inr(clean_df)
        final_df = create_features(conv_df)

        knowledge_engine = VehicleKnowledgeEngine(final_df)
        logger.info("Vehicle Knowledge Engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Knowledge Engine: {e}")


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class CarInput(BaseModel):
    """Input schema for car price prediction."""

    brand: str = Field(..., description="Car brand name")
    model: str = Field(..., description="Car base model name (from knowledge engine)")
    variant: str = Field(
        default="Standard", description="Car variant (from knowledge engine)"
    )
    year: int = Field(..., ge=1990, le=2030, description="Manufacturing year")
    transmission: str = Field(..., description="Transmission type (Manual/Automatic)")
    mileage: int = Field(..., ge=0, description="Kilometers driven")
    fuelType: str = Field(..., description="Fuel type (Petrol/Diesel/CNG/LPG/Electric)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand": "Maruti",
                "model": "Swift",
                "variant": "Dzire VDI",
                "year": 2018,
                "transmission": "Manual",
                "mileage": 45000,
                "fuelType": "Diesel",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for car price prediction."""

    predicted_price: str = Field(..., description="Predicted price in INR")
    predicted_price_raw: float = Field(..., description="Raw predicted price in INR")
    price_range: str = Field(..., description="Estimated price range")
    original_price: str = Field(..., description="Simulated original price")
    depreciation_percent: str = Field(
        ..., description="Estimated depreciation percentage"
    )
    confidence: str = Field(..., description="Prediction confidence score")
    recommendations: list = Field(
        default=[], description="Similar vehicle recommendations"
    )
    input_summary: dict = Field(..., description="Summary of input features used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    knowledge_engine_loaded: bool
    version: str = "2.0.0"


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
    """Check if the API, model, and knowledge engine are healthy."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        knowledge_engine_loaded=knowledge_engine is not None,
        version="2.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict car price",
    description="Predict the price of a used car in Indian Rupees (₹) using the Vehicle Knowledge Engine.",
)
async def predict(car: CarInput):
    """
    Predict the price of a used car.

    Returns the predicted price in Indian Lakhs notation.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )
    if knowledge_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge Engine not loaded.",
        )

    try:
        # Retrieve specs from knowledge engine automatically
        specs = knowledge_engine.get_specs(
            brand=car.brand,
            base_model=car.model,
            year=car.year,
            variant=car.variant,
            fuel=car.fuelType,
            transmission=car.transmission,
        )

        if not specs:
            raise HTTPException(
                status_code=400,
                detail="Invalid vehicle configuration. Combination does not exist in the Indian market.",
            )

        full_model = knowledge_engine.get_full_model_string(car.model, car.variant)

        result = predict_price(
            brand=car.brand,
            model=full_model,
            year=car.year,
            transmission=car.transmission,
            mileage=car.mileage,
            fuel_type=car.fuelType,
            mpg=specs["mileage"],
            engine_size=specs["engineSize"],
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/knowledge/brands", tags=["Knowledge Engine"])
async def get_brands():
    """Get the list of all valid car brands."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge Engine offline")
    return {"brands": knowledge_engine.get_brands()}


@app.get("/knowledge/models", tags=["Knowledge Engine"])
async def get_models(brand: str):
    """Get available base models for a specific brand."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge Engine offline")
    models = knowledge_engine.get_models(brand)
    if not models:
        raise HTTPException(status_code=404, detail=f"Brand '{brand}' not found.")
    return {"brand": brand, "models": models}


@app.get("/knowledge/years", tags=["Knowledge Engine"])
async def get_years(brand: str, model: str):
    """Get available manufacturing years for a brand and model."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge Engine offline")
    years = knowledge_engine.get_years(brand, model)
    if not years:
        raise HTTPException(status_code=404, detail="Configuration not found.")
    return {"years": years}


@app.get("/knowledge/variants", tags=["Knowledge Engine"])
async def get_variants(brand: str, model: str, year: int):
    """Get available variants for a specific configuration."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge Engine offline")
    return {"variants": knowledge_engine.get_variants(brand, model, year)}


@app.get("/knowledge/specs", tags=["Knowledge Engine"])
async def get_specs(
    brand: str, model: str, year: int, variant: str, fuel: str, transmission: str
):
    """Get inferred specifications for a complete configuration."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge Engine offline")
    specs = knowledge_engine.get_specs(brand, model, year, variant, fuel, transmission)
    if not specs:
        raise HTTPException(status_code=404, detail="Configuration not found.")
    return {"specs": specs}


@app.get("/knowledge/similar", tags=["Knowledge Engine"])
async def get_similar(brand: str, model: str, year: int, variant: str):
    """Get similar alternative configurations for a vehicle."""
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge Engine offline")
    similar = knowledge_engine.get_similar_configurations(brand, model, year, variant)
    return {"similar_configurations": similar}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
