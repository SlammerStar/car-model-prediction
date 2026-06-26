from typing import Dict, Any


class NegotiationEngine:
    def __init__(self):
        pass

    def calculate_negotiation(
        self, valuation_report: Dict[str, Any], asking_price: float = None
    ) -> Dict[str, Any]:
        if not asking_price:
            return {"is_available": False}

        est_value = valuation_report["estimated_market_value_raw"]

        # Suggested initial offer is typically 5-10% below fair value or asking price, whichever is lower
        initial_offer = min(est_value * 0.92, asking_price * 0.90)

        # Max recommended is fair value + a small premium if it's a good car
        max_offer = est_value * 1.02

        margin = max(0, asking_price - initial_offer)

        # Negotiation difficulty
        if asking_price > est_value * 1.15:
            difficulty = "Hard"
            reasoning = "The asking price is significantly above market value. The seller may have unrealistic expectations."
        elif asking_price > est_value:
            difficulty = "Moderate"
            reasoning = "The asking price is slightly above market value. There is room for standard negotiation."
        else:
            difficulty = "Easy"
            reasoning = "The asking price is already at or below fair market value. Minimal negotiation needed."

        from src.utils import format_price_inr

        return {
            "is_available": True,
            "seller_asking_price": format_price_inr(asking_price),
            "estimated_fair_value": format_price_inr(est_value),
            "suggested_initial_offer": format_price_inr(initial_offer),
            "maximum_recommended_offer": format_price_inr(max_offer),
            "estimated_negotiation_margin": format_price_inr(margin),
            "negotiation_difficulty": difficulty,
            "reasoning": reasoning,
        }

    def process(
        self, valuation_report: Dict[str, Any], asking_price: float = None
    ) -> Dict[str, Any]:
        return self.calculate_negotiation(valuation_report, asking_price)
