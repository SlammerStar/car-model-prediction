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

    def _assess_risks(
        self,
        input_features: pd.DataFrame,
        input_data: pd.DataFrame,
        confidence_label: str,
    ) -> Dict[str, str]:
        """
        Provide risk assessment mapping based on realistic market heuristics.
        """
        risk_config = self.config["risk_thresholds"]

        car_age = input_data.get("car_age", pd.Series([5])).iloc[0]
        mileage = input_data.get("mileage", pd.Series([50000])).iloc[0]
        engine_size = input_data.get("engine_size", pd.Series([1.2])).iloc[0]
        brand = input_data.get("brand", pd.Series(["Unknown"])).iloc[0]

        # Get Brand Knowledge
        brand_info = {}
        if self.knowledge_engine:
            brand_info = self.knowledge_engine.get_brand_knowledge(brand)

        reliability = brand_info.get("reliability", 0.65)
        liquidity = brand_info.get("liquidity", 0.50)
        is_premium = brand_info.get("premium", False)

        # 1. Ownership Risk
        # Explicitly handle missing owner count by inferring risk based on age & confidence
        # Missing info penalty: if confidence is Low, ownership risk is higher
        base_owner_risk = 0.2 if car_age <= 3 else (0.4 if car_age <= 7 else 0.6)
        penalty = (
            0.2
            if confidence_label == "Low Confidence"
            else (0.1 if confidence_label == "Medium Confidence" else 0.0)
        )
        owner_risk_score = min(1.0, base_owner_risk + penalty)

        # 2. Market Risk
        # Based on brand liquidity, popularity, and stability
        stability = input_features.get("market_stability_score", pd.Series([0.5])).iloc[
            0
        ]
        popularity = input_features.get("brand_popularity", pd.Series([0.5])).iloc[0]
        # High liquidity + High stability + High popularity = Low Market Risk
        market_strength = (liquidity * 0.5) + (stability * 0.3) + (popularity * 0.2)
        market_risk_score = max(0.0, 1.0 - market_strength)

        # 3. Maintenance Risk
        # Based on reliability, age, mileage, premium status, and engine complexity
        age_factor = min(1.0, car_age / 15.0)
        mileage_factor = min(1.0, mileage / 150000.0)

        # Engine complexity proxy: > 1.6L or premium brand usually means more complex/expensive maintenance
        complexity_penalty = 0.15 if engine_size > 1.6 else 0.0
        premium_penalty = 0.2 if is_premium else 0.0

        # High reliability reduces risk
        maintenance_risk_score = (
            (age_factor * 0.4)
            + (mileage_factor * 0.4)
            + complexity_penalty
            + premium_penalty
        )
        maintenance_risk_score = maintenance_risk_score * (
            1.5 - reliability
        )  # High reliability reduces the final score
        maintenance_risk_score = min(1.0, max(0.0, maintenance_risk_score))

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
        input_data: pd.DataFrame,
        predicted_price: float,
        confidence_label: str,
        market_position: str,
        positive_factors: list,
        negative_factors: list,
    ) -> str:
        """
        Generate a concise natural-language summary explaining the valuation.
        1. Brief description of the vehicle.
        2. Key positive valuation factors.
        3. Key negative valuation factors.
        4. Overall market conclusion.
        """
        brand = input_data.get("brand", pd.Series(["Unknown"])).iloc[0]
        model = input_data.get("model", pd.Series(["Unknown"])).iloc[0]
        year = input_data.get("year", pd.Series([2023])).iloc[0]
        mileage = input_data.get("mileage", pd.Series([50000])).iloc[0]

        price_str = format_price_inr(predicted_price)

        # 1. Description
        summary = f"The {year} {brand} {model} (driven {mileage:,} km) has an estimated market value of {price_str}. "

        # 2. Positive
        if positive_factors:
            pos_features = [f["feature"].lower() for f in positive_factors[:2]]
            if len(pos_features) > 1:
                summary += f"This valuation is strongly supported by positive factors such as its {pos_features[0]} and {pos_features[1]}. "
            else:
                summary += (
                    f"This valuation is positively supported by its {pos_features[0]}. "
                )

        # 3. Negative
        if negative_factors:
            neg_features = [f["feature"].lower() for f in negative_factors[:2]]
            if len(neg_features) > 1:
                summary += f"Conversely, its {neg_features[0]} and {neg_features[1]} present slight negative pressure on its maximum resale potential. "
            else:
                summary += f"However, the vehicle's {neg_features[0]} limits its maximum resale potential. "

        # 4. Conclusion
        if market_position == "Undervalued":
            summary += f"Overall, this configuration exhibits strong value retention and is positioned as highly attractive relative to current market baselines with {confidence_label.lower()}."
        elif market_position == "Overpriced":
            summary += f"Overall, this specific configuration is experiencing higher depreciation, positioning it as slightly overpriced in the current market (evaluated with {confidence_label.lower()})."
        else:
            summary += f"Overall, this vehicle is fairly aligned with current market expectations and standard depreciation curves (evaluated with {confidence_label.lower()})."

        return summary

    def generate_valuation_report(
        self, model_pipeline, input_data: pd.DataFrame, input_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a complete structured valuation report.
        """
        # Determine the correct dataframe to pass based on expected features
        if hasattr(model_pipeline, "feature_names_in_"):
            expected_features = list(model_pipeline.feature_names_in_)
            missing_cols = [
                col for col in expected_features if col not in input_features.columns
            ]

            if not missing_cols:
                predict_df = input_features[expected_features]
            else:
                predict_df = input_features
        else:
            predict_df = input_features

        try:
            predicted_price = model_pipeline.predict(predict_df)[0]
        except ValueError as e:
            if "features as input" in str(e):
                # Fallback to base input data
                predict_df = input_data
                predicted_price = model_pipeline.predict(predict_df)[0]
            else:
                raise e

        predicted_price = max(0.0, float(predicted_price))

        # 2. Explainability
        explanation = {}
        if self.explanation_engine:
            explanation = self.explanation_engine.explain_prediction(
                model_pipeline, predict_df
            )

        # 3. Confidence & Ranges
        confidence_score = self._calculate_confidence(input_features, predicted_price)
        confidence_label = self._get_confidence_label(confidence_score)

        lower_bound, upper_bound = self._calculate_dynamic_range(
            predicted_price, confidence_score, input_features
        )

        # 4. Market Position & Risk
        risks = self._assess_risks(input_features, input_data, confidence_label)
        market_position = self._get_market_position(input_features)
        recommendation = self._get_recommendation(
            market_position, confidence_label, risks
        )

        # 5. AI Summary
        pos_factors = explanation.get("top_positive_factors", [])
        neg_factors = explanation.get("top_negative_factors", [])

        summary = self._generate_ai_summary(
            input_data,
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
