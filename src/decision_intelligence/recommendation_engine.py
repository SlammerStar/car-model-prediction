from typing import Dict, Any


class RecommendationEngine:
    def __init__(self):
        pass

    def calculate_remaining_life(self, car_age: int) -> Dict[str, Any]:
        """Calculates a simple remaining life index."""
        max_life = 15  # standard legal/practical life for Indian cars
        remaining = max(0, max_life - car_age)
        index = min(100, max(0, int((remaining / max_life) * 100)))

        return {
            "remaining_years": remaining,
            "life_index_score": index,
            "label": "High" if index > 60 else ("Medium" if index > 30 else "Low"),
        }

    def calculate_deal_score(
        self, valuation_report: Dict[str, Any], asking_price: float = None
    ) -> int:
        """Computes the overall Deal Score (0-100)."""
        score = 50  # Base score

        # 1. Price Fairness (if asking price exists)
        if asking_price:
            est_value = valuation_report["estimated_market_value_raw"]
            if asking_price < est_value * 0.95:
                score += 25
            elif asking_price <= est_value * 1.05:
                score += 10
            elif asking_price > est_value * 1.15:
                score -= 20
        else:
            # If no asking price, assume it's fairly priced for the baseline
            score += 10

        # 2. Confidence
        conf = valuation_report["confidence"]["score"]
        score += (conf - 70) * 0.2  # Slight bump for high confidence

        # 3. Market Position
        pos = valuation_report["market_position"]
        if pos == "Undervalued":
            score += 15
        elif pos == "Overpriced":
            score -= 15

        # 4. Risk
        risks = valuation_report["risk_assessment"]
        if (
            risks.get("Ownership Risk") == "High"
            or risks.get("Maintenance Risk") == "High"
        ):
            score -= 20
        elif (
            risks.get("Ownership Risk") == "Low"
            and risks.get("Maintenance Risk") == "Low"
        ):
            score += 10

        return min(100, max(0, int(score)))

    def generate_recommendation(
        self, deal_score: int, asking_price: float = None
    ) -> str:
        """Determines the final recommendation."""
        if deal_score >= 85:
            return "Excellent Buy"
        elif deal_score >= 70:
            return "Good Buy"
        elif deal_score >= 50:
            if asking_price:
                return "Negotiate"
            return "Fair Purchase"
        elif deal_score >= 35:
            return "Wait"
        else:
            return "Avoid"

    def process(
        self,
        valuation_report: Dict[str, Any],
        input_summary: Dict[str, Any],
        asking_price: float = None,
    ) -> Dict[str, Any]:
        """Main method."""
        car_age = input_summary.get("car_age", 5)
        remaining_life = self.calculate_remaining_life(car_age)
        deal_score = self.calculate_deal_score(valuation_report, asking_price)
        recommendation = self.generate_recommendation(deal_score, asking_price)

        return {
            "remaining_life": remaining_life,
            "deal_score": deal_score,
            "final_recommendation": recommendation,
        }
