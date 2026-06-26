import pytest
import pandas as pd
import json
from src.valuation_intelligence import ValuationIntelligenceEngine


class MockPipeline:
    def predict(self, X):
        return [500000.0]


@pytest.fixture
def valuation_engine(tmp_path):
    config = {
        "market_position_thresholds": {
            "undervalued_max": -0.05,
            "fairly_priced_max": 0.05,
            "slightly_overpriced_max": 0.15,
        },
        "confidence_thresholds": {"high_min": 85, "medium_min": 60},
        "risk_thresholds": {
            "ownership_risk": {"high_min": 0.7, "medium_min": 0.4},
            "market_risk": {"high_min": 0.6, "medium_min": 0.3},
            "maintenance_risk": {"high_min": 0.7, "medium_min": 0.4},
        },
        "recommendation_matrix": {
            "Excellent Buy": {
                "market_position": ["Undervalued"],
                "confidence": ["High"],
                "max_ownership_risk": "Low",
            },
            "Fair Purchase": {
                "market_position": ["Fairly Priced"],
                "confidence": ["Medium"],
                "max_ownership_risk": "Medium",
            },
        },
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    return ValuationIntelligenceEngine(config_path=str(config_file))


def test_calculate_confidence(valuation_engine):
    # Old car, rare config
    X = pd.DataFrame(
        [
            {
                "car_age": 15,
                "configuration_scarcity_score": 0.9,
                "market_liquidity_score": 0.2,
            }
        ]
    )
    score = valuation_engine._calculate_confidence(X, 500000.0)
    assert score < 100.0


def test_get_market_position(valuation_engine):
    X = pd.DataFrame([{"brand_resale_retention_score": 1.1}])  # Offset -0.1
    assert valuation_engine._get_market_position(X) == "Undervalued"

    X = pd.DataFrame([{"brand_resale_retention_score": 0.9}])  # Offset 0.1
    assert valuation_engine._get_market_position(X) == "Slightly Overpriced"


def test_generate_valuation_report(valuation_engine):
    pipeline = MockPipeline()
    input_data = pd.DataFrame([{"brand": "Test", "model": "Car"}])
    X = pd.DataFrame(
        [
            {
                "car_age": 5,
                "brand_resale_retention_score": 1.1,
                "owner_risk_score": 0.1,
                "market_stability_score": 0.9,
                "age_based_depreciation": 0.1,
                "configuration_scarcity_score": 0.1,
                "market_liquidity_score": 0.9,
            }
        ]
    )

    report = valuation_engine.generate_valuation_report(pipeline, input_data, X)

    assert report["estimated_market_value_raw"] == 500000.0
    assert report["market_position"] == "Undervalued"
    assert report["recommendation"] == "Excellent Buy"
    assert report["confidence"]["label"] == "High"
    assert "ownership risk" in [k.lower() for k in report["risk_assessment"].keys()]
    assert isinstance(report["ai_summary"], str)
