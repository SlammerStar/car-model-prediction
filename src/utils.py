"""
Utility functions and constants for the Car Price Prediction System.

This module contains shared constants, helper functions for price formatting,
model persistence, and path management used across the entire application.
"""

import os
import joblib
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("car_price_predictor")

# ---------------------------------------------------------------------------
# Project Paths (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "Datasets"  # legacy path fallback
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

# Create directories if they don't exist
for d in [MODELS_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCHANGE_RATE = 115  # GBP to INR conversion rate
CURRENT_YEAR = 2026  # Reference year for car_age calculation
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Brand-to-filename mapping
BRAND_FILE_MAP = {
    "Audi": "audi.csv",
    "BMW": "bmw.csv",
    "Ford": "ford.csv",
    "Hyundai": "hyundai.csv",
    "Mercedes": "mercedes.csv",
    "Skoda": "skoda.csv",
    "Toyota": "toyota.csv",
    "Volkswagen": "vw.csv",
}

# Feature column definitions
CATEGORICAL_FEATURES = ["brand", "model", "transmission", "fuelType"]
NUMERICAL_FEATURES = ["year", "car_age", "mileage", "mpg", "engineSize", "km_per_year", "premium_brand_flag"]
TARGET_COLUMN = "price_inr"

# Model save path
MODEL_PATH = MODELS_DIR / "model.pkl"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
RECOMMENDER_PATH = MODELS_DIR / "recommender.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Indian Market Calibration
# UK used cars are typically cheaper. These multipliers adjust the base converted price
# to reflect the heavy taxation and resale value trends in the Indian market.
INDIAN_MARKET_MULTIPLIERS = {
    "Audi": 2.8,
    "BMW": 2.8,
    "Mercedes": 2.9,
    "Toyota": 1.8,
    "Ford": 1.4,
    "Hyundai": 1.3,
    "Volkswagen": 1.6,
    "Skoda": 1.6,
}

PREMIUM_BRANDS = ["Audi", "BMW", "Mercedes"]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def format_price_inr(price: float) -> str:
    """
    Format a price in INR to the Indian Lakhs notation.

    Args:
        price: Price in Indian Rupees (raw number).

    Returns:
        Formatted string like '₹8.75 Lakhs' or '₹1.23 Crores'.

    Examples:
        >>> format_price_inr(875000)
        '₹8.75 Lakhs'
        >>> format_price_inr(12400000)
        '₹1.24 Crores'
    """
    if price >= 1_00_00_000:  # 1 Crore
        return f"₹{price / 1_00_00_000:.2f} Crores"
    else:
        return f"₹{price / 1_00_000:.2f} Lakhs"


def save_model(model, path: Path = MODEL_PATH) -> None:
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained scikit-learn model or pipeline.
        path: File path to save the model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH):
    """
    Load a trained model from disk.

    Args:
        path: File path to load the model from.

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run training first."
        )
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def get_data_dir() -> Path:
    """
    Determine the correct data directory, checking both new and legacy paths.

    Returns:
        Path to the directory containing the CSV files.
    """
    if DATA_DIR.exists() and any(DATA_DIR.glob("*.csv")):
        return DATA_DIR
    elif DATASETS_DIR.exists() and any(DATASETS_DIR.glob("*.csv")):
        return DATASETS_DIR
    else:
        raise FileNotFoundError(
            "No data directory found. Ensure CSV files are in 'data/' or 'Datasets/'."
        )
