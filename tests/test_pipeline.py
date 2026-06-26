"""
Test Suite for the DRIVEIQ Car Price Prediction System
======================================================

Tests for utility functions, data preprocessing, feature engineering,
and prediction logic.

Updated for DRIVEIQ 2.0 (Indian market datasets).
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd


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
        from src.utils import CURRENT_YEAR, RANDOM_STATE

        assert CURRENT_YEAR >= 2025  # Dynamic year
        assert RANDOM_STATE == 42

    def test_premium_brands(self):
        """Test that premium brands include Indian market luxury brands."""
        from src.utils import PREMIUM_BRANDS

        assert "Audi" in PREMIUM_BRANDS
        assert "BMW" in PREMIUM_BRANDS
        assert "Mercedes" in PREMIUM_BRANDS
        assert "Jaguar" in PREMIUM_BRANDS


# ============================================================================
# Tests: src/data_processing.py
# ============================================================================
class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    def test_load_and_merge_datasets(self):
        """Test that Indian datasets load and merge successfully."""
        from src.data_processing import load_and_merge_datasets

        df = load_and_merge_datasets()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 10000  # Should have 14K+ raw records
        assert "brand" in df.columns
        assert "selling_price" in df.columns
        assert "source" in df.columns
        # Verify Indian market brands present
        brands = df["brand"].unique()
        assert "Maruti" in brands
        assert "Hyundai" in brands

    def test_clean_data(self):
        """Test data cleaning removes duplicates and handles missing values."""
        from src.data_processing import load_and_merge_datasets, clean_data

        raw = load_and_merge_datasets()
        cleaned = clean_data(raw)
        assert len(cleaned) < len(raw)
        assert cleaned["selling_price"].min() >= 20000
        assert cleaned["km_driven"].min() >= 100

    def test_prices_are_native_inr(self):
        """Test that prices are in native INR (no conversion needed)."""
        from src.data_processing import load_and_merge_datasets, clean_data

        df = clean_data(load_and_merge_datasets())
        # Indian used car prices: most between 50K and 2 Crore
        assert df["selling_price"].median() > 100_000
        assert df["selling_price"].median() < 50_00_000

    def test_prepare_data(self):
        """Test the full data preparation pipeline."""
        from src.data_processing import prepare_data

        df = prepare_data()
        assert "price_inr" in df.columns
        assert "brand" in df.columns
        assert len(df) > 5000


# ============================================================================
# Tests: src/data_processing.py - Feature Engineering
# ============================================================================
class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_create_features(self):
        """Test car_age feature creation."""
        from src.data_processing import create_features
        from src.utils import CURRENT_YEAR

        df = pd.DataFrame(
            {
                "year": [2020, 2015, 2010],
                "brand": ["Maruti", "Hyundai", "Honda"],
                "model": ["Swift", "i20", "City"],
                "transmission": ["Manual", "Manual", "Automatic"],
                "fuel_type": ["Petrol", "Diesel", "Petrol"],
                "selling_price": [500000, 350000, 250000],
                "km_driven": [30000, 60000, 90000],
                "mileage_kmpl": [22.0, 20.0, 18.0],
                "engine_cc": [1197, 1396, 1497],
                "max_power_bhp": [83.0, 90.0, 117.0],
                "seats": [5, 5, 5],
            }
        )
        result = create_features(df)
        assert "car_age" in result.columns
        assert result["car_age"].iloc[0] == CURRENT_YEAR - 2020
        assert result["car_age"].iloc[1] == CURRENT_YEAR - 2015
        # Backward compatibility
        assert "price_inr" in result.columns
        assert "mileage" in result.columns
        assert "mpg" in result.columns
        assert "engineSize" in result.columns
        assert "fuelType" in result.columns

    def test_get_feature_target_split(self):
        """Test feature-target splitting."""
        from src.data_processing import get_feature_target_split

        df = pd.DataFrame(
            {
                "brand": ["Maruti"],
                "model": ["Swift"],
                "year": [2020],
                "car_age": [6],
                "transmission": ["Manual"],
                "mileage": [30000],
                "fuelType": ["Petrol"],
                "mpg": [22.0],
                "engineSize": [1.2],
                "price_inr": [500000],
                "km_per_year": [5000],
                "premium_brand_flag": [0],
            }
        )
        X, y = get_feature_target_split(df)
        assert "price_inr" not in X.columns
        assert len(y) == 1

    def test_build_preprocessor(self):
        """Test preprocessor construction."""
        from src.data_processing import build_preprocessor

        X = pd.DataFrame(
            {
                "brand": ["Maruti", "Hyundai"],
                "model": ["Swift", "i20"],
                "transmission": ["Manual", "Automatic"],
                "fuelType": ["Petrol", "Diesel"],
                "year": [2020, 2019],
                "car_age": [6, 7],
                "mileage": [30000, 40000],
                "mpg": [22.0, 20.0],
                "engineSize": [1.2, 1.4],
            }
        )
        preprocessor = build_preprocessor(X)
        assert preprocessor is not None


# ============================================================================
# Tests: Parsing Utilities
# ============================================================================
class TestParsing:
    """Tests for data parsing utilities."""

    def test_parse_brand_model(self):
        """Test brand/model extraction from combined name."""
        from src.data_processing import parse_brand_model

        brand, model, base_model, variant = parse_brand_model("Maruti Swift Dzire VDI")
        assert brand == "Maruti"
        assert "Swift" in model
        assert base_model == "Swift"
        assert variant == "Dzire VDI"

    def test_parse_brand_model_land_rover(self):
        """Test multi-word brand parsing."""
        from src.data_processing import parse_brand_model

        brand, model, base_model, variant = parse_brand_model(
            "Land Rover Range Rover 3.0"
        )
        assert brand == "Land Rover"
        assert base_model == "Range"
        assert variant == "Rover 3.0"

    def test_parse_mileage_kmpl(self):
        """Test fuel efficiency parsing."""
        from src.data_processing import parse_mileage_kmpl

        assert parse_mileage_kmpl("23.4 kmpl") == 23.4
        assert parse_mileage_kmpl("26.6 km/kg") == 26.6
        assert parse_mileage_kmpl(None) is None

    def test_parse_numeric_with_unit(self):
        """Test numeric parsing with units."""
        from src.data_processing import parse_numeric_with_unit

        assert parse_numeric_with_unit("1248 CC", "CC") == 1248.0
        assert parse_numeric_with_unit("74 bhp", "bhp") == 74.0
        assert parse_numeric_with_unit(None) is None


# ============================================================================
# Tests: src/predict.py
# ============================================================================
class TestPredict:
    """Tests for prediction functions."""

    def test_create_input_dataframe(self):
        """Test input DataFrame creation."""
        from src.prediction import create_input_dataframe

        df = create_input_dataframe(
            brand="Maruti",
            model="Swift",
            year=2019,
            transmission="Manual",
            mileage=45000,
            fuel_type="Petrol",
            mpg=22.0,
            engine_size=1.2,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["brand"].iloc[0] == "Maruti"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
