"""
Test Suite for the Car Price Prediction System
===============================================

Tests for utility functions, data preprocessing, feature engineering,
and prediction logic.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# Tests: src/utils.py
# ============================================================================
class TestUtils:
    """Tests for utility functions."""

    def test_format_price_inr_lakhs(self):
        """Test price formatting in Lakhs."""
        from src.utils import format_price_inr

        result = format_price_inr(875000)
        assert "₹" in result
        assert "Lakhs" in result
        assert "8.75" in result

    def test_format_price_inr_crores(self):
        """Test price formatting in Crores."""
        from src.utils import format_price_inr

        result = format_price_inr(1_50_00_000)
        assert "Crores" in result
        assert "1.50" in result

    def test_format_price_inr_zero(self):
        """Test formatting of zero price."""
        from src.utils import format_price_inr

        result = format_price_inr(0)
        assert "₹" in result
        assert "0.00" in result

    def test_project_paths_exist(self):
        """Test that project path constants are valid."""
        from src.utils import PROJECT_ROOT, MODELS_DIR, IMAGES_DIR

        assert PROJECT_ROOT.exists()
        assert MODELS_DIR.exists()
        assert IMAGES_DIR.exists()

    def test_constants(self):
        """Test that key constants are defined correctly."""
        from src.utils import EXCHANGE_RATE, CURRENT_YEAR, RANDOM_STATE

        assert EXCHANGE_RATE == 115
        assert CURRENT_YEAR == 2026
        assert RANDOM_STATE == 42

    def test_brand_file_map(self):
        """Test that all brands have file mappings."""
        from src.utils import BRAND_FILE_MAP

        expected_brands = [
            "Audi", "BMW", "Ford", "Hyundai",
            "Mercedes", "Skoda", "Toyota", "Volkswagen",
        ]
        for brand in expected_brands:
            assert brand in BRAND_FILE_MAP


# ============================================================================
# Tests: src/data_preprocessing.py
# ============================================================================
class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    def test_load_and_merge_datasets(self):
        """Test that datasets load and merge successfully."""
        from src.data_processing import load_and_merge_datasets

        df = load_and_merge_datasets()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "brand" in df.columns
        assert "price" in df.columns

    def test_clean_data(self):
        """Test data cleaning removes duplicates and handles missing values."""
        from src.data_processing import load_and_merge_datasets, clean_data

        raw = load_and_merge_datasets()
        cleaned = clean_data(raw)
        assert len(cleaned) <= len(raw)
        assert cleaned["price"].min() > 0

    def test_convert_price_to_inr(self):
        """Test GBP to INR conversion."""
        from src.data_processing import convert_price_to_inr

        df = pd.DataFrame({"price": [100, 200, 500], "brand": ["Ford", "Toyota", "BMW"]})
        result = convert_price_to_inr(df)
        assert "price_inr" in result.columns
        assert result["price_inr"].iloc[0] == 100 * 115 * 1.4
        assert result["price_inr"].iloc[1] == 200 * 115 * 1.8

    def test_prepare_data(self):
        """Test the full data preparation pipeline."""
        from src.data_processing import prepare_data

        df = prepare_data()
        assert "price_inr" in df.columns
        assert "brand" in df.columns
        assert len(df) > 1000  # Should have substantial data


# ============================================================================
# Tests: src/feature_engineering.py
# ============================================================================
class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_create_features(self):
        """Test car_age feature creation."""
        from src.data_processing import create_features

        df = pd.DataFrame({
            "year": [2020, 2015, 2010],
            "brand": ["BMW", "Audi", "Ford"],
            "model": ["X5", "A3", "Focus"],
            "transmission": ["Automatic", "Manual", "Manual"],
            "fuelType": ["Diesel", "Petrol", "Petrol"],
            "price_inr": [1000000, 500000, 300000],
            "mileage": [10000, 20000, 50000],
        })
        result = create_features(df)
        assert "car_age" in result.columns
        assert result["car_age"].iloc[0] == 2026 - 2020  # 6
        assert result["car_age"].iloc[1] == 2026 - 2015  # 11

    def test_get_feature_target_split(self):
        """Test feature-target splitting."""
        from src.data_processing import get_feature_target_split

        df = pd.DataFrame({
            "brand": ["BMW"],
            "model": ["X5"],
            "year": [2020],
            "car_age": [6],
            "transmission": ["Automatic"],
            "mileage": [30000],
            "fuelType": ["Diesel"],
            "mpg": [50.0],
            "engineSize": [2.0],
            "price_inr": [1500000],
        })
        X, y = get_feature_target_split(df)
        assert "price_inr" not in X.columns
        assert len(y) == 1

    def test_build_preprocessor(self):
        """Test preprocessor construction."""
        from src.data_processing import build_preprocessor

        X = pd.DataFrame({
            "brand": ["BMW", "Audi"],
            "model": ["X5", "A3"],
            "transmission": ["Auto", "Manual"],
            "fuelType": ["Diesel", "Petrol"],
            "year": [2020, 2019],
            "car_age": [6, 7],
            "mileage": [30000, 40000],
            "mpg": [50.0, 55.0],
            "engineSize": [2.0, 1.5],
        })
        preprocessor = build_preprocessor(X)
        assert preprocessor is not None


# ============================================================================
# Tests: src/predict.py
# ============================================================================
class TestPredict:
    """Tests for prediction functions."""

    def test_create_input_dataframe(self):
        """Test input DataFrame creation."""
        from src.prediction import create_input_dataframe

        df = create_input_dataframe(
            brand="BMW", model="X5", year=2019,
            transmission="Automatic", mileage=45000,
            fuel_type="Diesel", mpg=52.3, engine_size=2.0,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["brand"].iloc[0] == "BMW"
        assert df["car_age"].iloc[0] == 2026 - 2019


# ============================================================================
# Tests: API
# ============================================================================
class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_api_imports(self):
        """Test that the API module imports correctly."""
        from src.api import app
        assert app is not None

    def test_api_root(self):
        """Test the root endpoint."""
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_api_health(self):
        """Test the health check endpoint."""
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_api_brands(self):
        """Test the brands endpoint."""
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/brands")
        assert response.status_code == 200
        assert "brands" in response.json()
        assert len(response.json()["brands"]) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
