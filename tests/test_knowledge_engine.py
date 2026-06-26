import pytest
import pandas as pd
from src.knowledge_engine import VehicleKnowledgeEngine


@pytest.fixture
def mock_df():
    """Create a mock dataframe resembling the cleaned dataset."""
    return pd.DataFrame(
        [
            {
                "brand": "Maruti",
                "base_model": "Swift",
                "variant": "Dzire VDI",
                "year": 2018,
                "fuelType": "Diesel",
                "transmission": "Manual",
                "engineSize": 1.2,
                "max_power_bhp": 74.0,
                "mpg": 23.4,
                "seats": 5.0,
            },
            {
                "brand": "Maruti",
                "base_model": "Swift",
                "variant": "VXI",
                "year": 2019,
                "fuelType": "Petrol",
                "transmission": "Manual",
                "engineSize": 1.2,
                "max_power_bhp": 82.0,
                "mpg": 21.2,
                "seats": 5.0,
            },
            {
                "brand": "Audi",
                "base_model": "A4",
                "variant": "2.0 TDI",
                "year": 2017,
                "fuelType": "Diesel",
                "transmission": "Automatic",
                "engineSize": 2.0,
                "max_power_bhp": 174.3,
                "mpg": 17.1,
                "seats": 5.0,
            },
        ]
    )


def test_knowledge_engine_initialization(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    assert len(engine.kb) == 3
    assert "confidence_count" in engine.kb.columns


def test_knowledge_engine_get_brands(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    brands = engine.get_brands()
    assert brands == ["Audi", "Maruti"]


def test_knowledge_engine_get_models(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    assert engine.get_models("Maruti") == ["Swift"]
    assert engine.get_models("Audi") == ["A4"]
    assert engine.get_models("Toyota") == []


def test_knowledge_engine_get_years(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    years = engine.get_years("Maruti", "Swift")
    assert years == [2019, 2018]  # Sorted descending


def test_knowledge_engine_get_variants(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    variants = engine.get_variants("Maruti", "Swift", 2018)
    assert variants == ["Dzire VDI"]


def test_knowledge_engine_get_fuel_types(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    fuels = engine.get_fuel_types("Maruti", "Swift", 2019, "VXI")
    assert fuels == ["Petrol"]


def test_knowledge_engine_get_transmissions(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    transmissions = engine.get_transmissions("Audi", "A4", 2017, "2.0 TDI", "Diesel")
    assert transmissions == ["Automatic"]


def test_knowledge_engine_get_specs(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    specs = engine.get_specs("Maruti", "Swift", 2018, "Dzire VDI", "Diesel", "Manual")
    assert specs is not None
    assert specs["engineSize"] == 1.2
    assert specs["max_power_bhp"] == 74.0
    assert specs["mileage"] == 23.4
    assert specs["seats"] == 5
    assert specs["confidence_count"] == 1
    assert "confidence_level" in specs


def test_knowledge_engine_get_specs_not_found(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    specs = engine.get_specs("Maruti", "Swift", 2018, "Dzire VDI", "Petrol", "Manual")
    assert specs is None


def test_get_full_model_string(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    assert engine.get_full_model_string("Swift", "Dzire VDI") == "Swift Dzire VDI"
    assert engine.get_full_model_string("Swift", "Standard") == "Swift"


def test_get_variant_label(mock_df):
    engine = VehicleKnowledgeEngine(mock_df)
    label = engine.get_variant_label("Maruti", "Swift", 2018, "Dzire VDI")
    assert label == "Dzire VDI (1.2L Diesel Manual)"


def test_get_similar_configurations(mock_df):
    # Add another 2018 Swift variant to the mock dataframe for similarity testing
    mock_df.loc[3] = {
        "brand": "Maruti",
        "base_model": "Swift",
        "variant": "ZXI",
        "year": 2018,
        "fuelType": "Petrol",
        "transmission": "Automatic",
        "engineSize": 1.2,
        "max_power_bhp": 82.0,
        "mpg": 21.2,
        "seats": 5.0,
    }
    engine = VehicleKnowledgeEngine(mock_df)
    similar = engine.get_similar_configurations("Maruti", "Swift", 2018, "Dzire VDI")
    assert len(similar) == 1
    assert similar[0]["variant"] == "ZXI"
