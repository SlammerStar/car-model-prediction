from typing import Dict, Any, List


class BuyerInsightsEngine:
    def __init__(self):
        pass

    def generate_insights(
        self, valuation_report: Dict[str, Any], input_summary: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        insights = []

        # 1. Commuter / Mileage insight
        mpg = input_summary.get("mpg", 0)
        if mpg >= 20:
            insights.append(
                {
                    "insight": "Excellent Commuter",
                    "explanation": "High fuel efficiency makes this highly economical for daily city driving.",
                }
            )
        elif mpg < 12:
            insights.append(
                {
                    "insight": "High Running Cost",
                    "explanation": "Lower fuel efficiency will result in higher monthly fuel expenses.",
                }
            )

        # 2. Resale insight
        risks = valuation_report["risk_assessment"]
        if risks.get("Market Risk") == "Low":
            insights.append(
                {
                    "insight": "Strong Resale Value",
                    "explanation": "This model historically holds its value well in the secondary market.",
                }
            )

        # 3. Maintenance insight
        if risks.get("Maintenance Risk") == "Low":
            insights.append(
                {
                    "insight": "Low Maintenance",
                    "explanation": "Known for reliability with lower-than-average servicing costs.",
                }
            )

        # 4. Family / Highway insight
        engine_size = input_summary.get("engineSize", 0)
        if engine_size >= 1.5:
            insights.append(
                {
                    "insight": "Highway Friendly",
                    "explanation": "Larger engine capacity provides comfortable cruising and passing power on highways.",
                }
            )

        # Fallback if empty
        if not insights:
            insights.append(
                {
                    "insight": "Standard Market Profile",
                    "explanation": "This vehicle exhibits typical market behavior for its segment.",
                }
            )

        return insights

    def process(
        self, valuation_report: Dict[str, Any], input_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"insights": self.generate_insights(valuation_report, input_summary)}
