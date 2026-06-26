import pytest
import pandas as pd
from src.explanation_engine import ExplanationEngine, BaseExplanationProvider


class MockExplanationProvider(BaseExplanationProvider):
    def explain(self, model, X: pd.DataFrame):
        base_value = 500000.0
        contributions = {
            "car_age": -50000.0,
            "engine_performance_score": 30000.0,
            "owner_risk_score": -10000.0,
            "brand_popularity": 25000.0,
        }
        return base_value, contributions


@pytest.fixture
def explanation_engine():
    config = {
        "feature_translation_dictionary": {
            "car_age": "Vehicle Age",
            "engine_performance_score": "Engine Performance",
        }
    }
    return ExplanationEngine(MockExplanationProvider(), config)


def test_explain_prediction(explanation_engine):
    X = pd.DataFrame(
        [
            {
                "car_age": 5,
                "engine_performance_score": 0.8,
                "owner_risk_score": 0.5,
                "brand_popularity": 0.9,
            }
        ]
    )
    result = explanation_engine.explain_prediction(None, X)

    assert result["base_value"] == 500000.0

    # Check translations and directions
    assert len(result["top_negative_factors"]) == 2
    assert result["top_negative_factors"][0]["feature"] == "Vehicle Age"
    assert (
        "reduces the market value" in result["top_negative_factors"][0]["explanation"]
    )

    assert len(result["top_positive_factors"]) == 2
    assert result["top_positive_factors"][0]["feature"] == "Engine Performance"
    assert (
        "increases the market value" in result["top_positive_factors"][0]["explanation"]
    )
