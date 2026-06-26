from typing import Dict, Any, List

from .recommendation_engine import RecommendationEngine
from .negotiation_engine import NegotiationEngine
from .forecast_engine import ForecastEngine
from .ownership_engine import OwnershipEngine
from .alternatives_engine import AlternativesEngine
from .buyer_insights_engine import BuyerInsightsEngine


class DecisionIntelligenceEngine:
    """
    Orchestrates the decision intelligence generation process.
    Acts as a facade over all the specific modules.
    """

    def __init__(self, knowledge_engine=None):
        self.recommendation_engine = RecommendationEngine()
        self.negotiation_engine = NegotiationEngine()
        self.forecast_engine = ForecastEngine()
        self.ownership_engine = OwnershipEngine()
        self.alternatives_engine = AlternativesEngine(knowledge_engine)
        self.buyer_insights_engine = BuyerInsightsEngine()

    def generate_decision_report(
        self,
        valuation_report: Dict[str, Any],
        input_summary: Dict[str, Any],
        asking_price: float = None,
        current_recommendations: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive decision intelligence report.
        """
        report = {}

        # 1. Ownership & Forecast
        report.update(self.ownership_engine.process(valuation_report, input_summary))
        report.update(self.forecast_engine.process(valuation_report))

        # 2. Buyer Insights
        report.update(
            self.buyer_insights_engine.process(valuation_report, input_summary)
        )

        # 3. Alternatives
        report.update(
            self.alternatives_engine.process(
                valuation_report, input_summary, current_recommendations
            )
        )

        # 4. Negotiation (if asking price is provided)
        report["negotiation_assistant"] = self.negotiation_engine.process(
            valuation_report, asking_price
        )

        # 5. Final Recommendation & Deal Score
        report.update(
            self.recommendation_engine.process(
                valuation_report, input_summary, asking_price
            )
        )

        return report
