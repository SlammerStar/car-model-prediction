import json
from typing import Dict, Any, Optional
import pandas as pd

from src.utils import format_price_inr
from src.knowledge_engine import VehicleKnowledgeEngine
from src.explanation_engine import ExplanationEngine


class ValuationIntelligenceEngine:
    def __init__(
        self,
        config_path: str = "src/valuation_config.json",
        explanation_engine: Optional[ExplanationEngine] = None,
        knowledge_engine: Optional[VehicleKnowledgeEngine] = None,
    ):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.explanation_engine = explanation_engine
        self.knowledge_engine = knowledge_engine

    def _calculate_confidence(
        self, input_features: pd.DataFrame, predicted_price: float
    ) -> float:
        """
        Calculate prediction confidence (0-100) based on multiple evidence sources.
        """
        confidence = 100.0

        # 1. Age penalty (older cars are harder to price accurately)
        car_age = input_features.get("car_age", pd.Series([0])).iloc[0]
        if car_age > 10:
            confidence -= (car_age - 10) * 1.5

        # 2. Scarcity penalty (rare configurations have lower confidence)
        scarcity = input_features.get(
            "configuration_scarcity_score", pd.Series([0.0])
        ).iloc[0]
        if scarcity > 0.7:
            confidence -= (scarcity - 0.7) * 20

        # 3. Liquidity penalty (hard to sell cars have volatile prices)
        liquidity = input_features.get("market_liquidity_score", pd.Series([1.0])).iloc[
            0
        ]
        if liquidity < 0.3:
            confidence -= (0.3 - liquidity) * 20

        return max(min(confidence, 98.0), 50.0)

    def _get_confidence_label(self, confidence_score: float) -> str:
        thresh = self.config["confidence_thresholds"]
        if confidence_score >= thresh["high_min"]:
            return "High"
        elif confidence_score >= thresh["medium_min"]:
            return "Medium"
        return "Low"

    def _calculate_dynamic_range(
        self, predicted_price: float, confidence: float, input_features: pd.DataFrame
    ) -> tuple[float, float]:
        """
        Calculate the predicted price interval using dynamic variance.
        """
        # Base variance of 3%
        variance = 0.03

        # Add variance based on lack of confidence
        # For 50% confidence, we add 10% variance. For 100% confidence, 0% variance.
        confidence_penalty = (100 - confidence) / 100.0 * 0.20
        variance += confidence_penalty

        # Market stability modifier
        stability = input_features.get("market_stability_score", pd.Series([0.5])).iloc[
            0
        ]
        variance += (1.0 - stability) * 0.05

        # Max cap variance to 25%
        variance = min(variance, 0.25)

        lower_bound = predicted_price * (1 - variance)
        upper_bound = predicted_price * (1 + variance)

        return lower_bound, upper_bound

    def _assess_risks(self, input_features: pd.DataFrame) -> Dict[str, str]:
        """
        Provide risk assessment mapping.
        """
        risk_config = self.config["risk_thresholds"]

        owner_risk_score = input_features.get(
            "owner_risk_score", pd.Series([0.0])
        ).iloc[0]
        market_risk_score = (
            1.0 - input_features.get("market_stability_score", pd.Series([1.0])).iloc[0]
        )
        maintenance_risk_score = input_features.get(
            "age_based_depreciation", pd.Series([0.0])
        ).iloc[0]

        def _get_level(score, r_type):
            if score >= risk_config[r_type]["high_min"]:
                return "High"
            elif score >= risk_config[r_type]["medium_min"]:
                return "Medium"
            return "Low"

        return {
            "Ownership Risk": _get_level(owner_risk_score, "ownership_risk"),
            "Market Risk": _get_level(market_risk_score, "market_risk"),
            "Maintenance Risk": _get_level(maintenance_risk_score, "maintenance_risk"),
        }

    def _get_market_position(self, input_features: pd.DataFrame) -> str:
        """
        Classify market position using business thresholds.
        """
        # For a single prediction, we can't truly know if the 'asking price' is overpriced without the asking price.
        # But we can assess if the CAR's intrinsic value makes it a premium-priced car relative to its baseline.
        # Let's interpret 'Market Position' as how well the car retains its value compared to an average car.

        retention = input_features.get(
            "brand_resale_retention_score", pd.Series([1.0])
        ).iloc[0]

        # If retention > 1.0, it retains better.
        # To match the prompt's thresholds (-0.05, +0.05, +0.15), we map this.
        # Since we don't have the seller's asking price, we will simulate a Market Position based on the car's general depreciation profile.
        # Alternatively, if we just assume the predicted price is the 'market value',
        # Market Position without an asking price is essentially "Value Retention".
        # For the sake of the interface, we'll map brand retention to market position.

        offset = 1.0 - retention
        thresh = self.config["market_position_thresholds"]

        if offset < thresh["undervalued_max"]:
            return "Undervalued"
        elif offset <= thresh["fairly_priced_max"]:
            return "Fairly Priced"
        elif offset <= thresh["slightly_overpriced_max"]:
            return "Slightly Overpriced"
        else:
            return "Overpriced"

    def _get_recommendation(
        self, market_position: str, confidence_label: str, risks: Dict[str, str]
    ) -> str:
        """
        Generate recommendations using multiple factors.
        """
        ownership_risk = risks.get("Ownership Risk", "Low")

        matrix = self.config["recommendation_matrix"]

        for rec, rules in matrix.items():
            if (
                market_position in rules["market_position"]
                and confidence_label in rules["confidence"]
            ):
                max_risk = rules["max_ownership_risk"]
                # Risk check: Low < Medium < High
                risk_levels = {"Low": 1, "Medium": 2, "High": 3}
                if risk_levels[ownership_risk] <= risk_levels[max_risk]:
                    return rec

        return "Fair Purchase"  # Fallback

    def _generate_ai_summary(
        self,
        brand: str,
        model: str,
        predicted_price: float,
        confidence_label: str,
        market_position: str,
        positive_factors: list,
        negative_factors: list,
    ) -> str:
        """
        Generate a concise natural-language summary explaining the valuation.
        """
        price_str = format_price_inr(predicted_price)
        summary = f"The {brand} {model} has an estimated market value of {price_str} with {confidence_label.lower()} confidence. "

        if market_position == "Undervalued":
            summary += "This configuration exhibits strong value retention and is currently positioned favorably in the market. "
        elif market_position == "Overpriced":
            summary += "This configuration is experiencing higher depreciation, positioning it as slightly overpriced relative to market baselines. "
        else:
            summary += (
                "This vehicle is fairly aligned with current market expectations. "
            )

        if positive_factors:
            best = positive_factors[0]["feature"].lower()
            summary += f"The valuation is positively supported by its {best}. "

        if negative_factors:
            worst = negative_factors[0]["feature"].lower()
            summary += (
                f"However, the vehicle's {worst} limits its maximum resale potential."
            )

        return summary

    def generate_valuation_report(
        self, model_pipeline, input_data: pd.DataFrame, input_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a complete structured valuation report.
        """
        # 1. Prediction
        predicted_price = model_pipeline.predict(input_features)[0]
        predicted_price = max(0.0, float(predicted_price))

        brand = input_data["brand"].iloc[0]
        model = input_data["model"].iloc[0]

        # 2. Explainability
        explanation = {}
        if self.explanation_engine:
            explanation = self.explanation_engine.explain_prediction(
                model_pipeline, input_features
            )

        # 3. Confidence & Ranges
        confidence_score = self._calculate_confidence(input_features, predicted_price)
        confidence_label = self._get_confidence_label(confidence_score)

        lower_bound, upper_bound = self._calculate_dynamic_range(
            predicted_price, confidence_score, input_features
        )

        # 4. Market Position & Risk
        risks = self._assess_risks(input_features)
        market_position = self._get_market_position(input_features)
        recommendation = self._get_recommendation(
            market_position, confidence_label, risks
        )

        # 5. AI Summary
        pos_factors = explanation.get("top_positive_factors", [])
        neg_factors = explanation.get("top_negative_factors", [])

        summary = self._generate_ai_summary(
            brand,
            model,
            predicted_price,
            confidence_label,
            market_position,
            pos_factors,
            neg_factors,
        )

        # Build final report
        return {
            "estimated_market_value": format_price_inr(predicted_price),
            "estimated_market_value_raw": predicted_price,
            "estimated_market_range": {
                "lower_bound": format_price_inr(lower_bound),
                "upper_bound": format_price_inr(upper_bound),
                "lower_raw": lower_bound,
                "upper_raw": upper_bound,
            },
            "confidence": {
                "score": round(confidence_score, 1),
                "label": confidence_label,
            },
            "market_position": market_position,
            "risk_assessment": risks,
            "recommendation": recommendation,
            "ai_summary": summary,
            "explanation": {
                "major_positive_factors": pos_factors,
                "major_negative_factors": neg_factors,
            },
        }
