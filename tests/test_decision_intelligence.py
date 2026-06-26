from src.decision_intelligence.recommendation_engine import RecommendationEngine
from src.decision_intelligence.negotiation_engine import NegotiationEngine
from src.decision_intelligence.forecast_engine import ForecastEngine
from src.decision_intelligence.ownership_engine import OwnershipEngine
from src.decision_intelligence.buyer_insights_engine import BuyerInsightsEngine
from src.decision_intelligence.decision_engine import DecisionIntelligenceEngine


def test_recommendation_engine():
    engine = RecommendationEngine()
    val_report = {
        "estimated_market_value_raw": 500000,
        "confidence": {"score": 80},
        "market_position": "Fairly Priced",
        "risk_assessment": {"Ownership Risk": "Low", "Maintenance Risk": "Low"},
    }

    # Test deal score with good asking price
    score = engine.calculate_deal_score(val_report, 450000)
    assert score > 50  # Should get a boost for being underpriced

    # Test deal score with high asking price
    score_high = engine.calculate_deal_score(val_report, 600000)
    assert score_high < score

    # Test Recommendation
    rec = engine.generate_recommendation(90)
    assert rec == "Excellent Buy"


def test_negotiation_engine():
    engine = NegotiationEngine()
    val_report = {"estimated_market_value_raw": 500000}

    res = engine.calculate_negotiation(val_report, None)
    assert res["is_available"] is False

    res2 = engine.calculate_negotiation(val_report, 550000)
    assert res2["is_available"] is True
    assert res2["negotiation_difficulty"] == "Moderate"


def test_forecast_engine():
    engine = ForecastEngine()
    val_report = {"estimated_market_value_raw": 100000}
    res = engine.process(val_report)
    assert "forecast_timeline" in res
    assert len(res["forecast_timeline"]) == 5
    assert res["forecast_timeline"][0]["year"] == 1


def test_ownership_engine():
    engine = OwnershipEngine()
    val_report = {"estimated_market_value_raw": 500000}
    input_sum = {"car_age": 5, "engineSize": 1.2, "mpg": 15.0}

    res = engine.process(val_report, input_sum)
    assert "total_5y" in res
    assert "insurance_5y" in res


def test_buyer_insights_engine():
    engine = BuyerInsightsEngine()
    val_report = {"risk_assessment": {"Market Risk": "Low", "Maintenance Risk": "Low"}}
    input_sum = {"mpg": 22.0, "engineSize": 1.0}

    res = engine.process(val_report, input_sum)
    insights = [i["insight"] for i in res["insights"]]
    assert "Excellent Commuter" in insights
    assert "Strong Resale Value" in insights
    assert "Low Maintenance" in insights


def test_decision_engine_orchestration():
    engine = DecisionIntelligenceEngine()
    val_report = {
        "estimated_market_value_raw": 500000,
        "confidence": {"score": 80},
        "market_position": "Fairly Priced",
        "risk_assessment": {"Ownership Risk": "Low", "Maintenance Risk": "Low"},
    }
    input_sum = {"car_age": 5, "engineSize": 1.2, "mpg": 15.0}

    report = engine.generate_decision_report(val_report, input_sum, asking_price=None)

    assert "deal_score" in report
    assert "final_recommendation" in report
    assert "forecast_timeline" in report
    assert "total_5y" in report
    assert "insights" in report
    assert report["negotiation_assistant"]["is_available"] is False
