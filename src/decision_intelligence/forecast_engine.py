from typing import Dict, Any, List


class ForecastEngine:
    def __init__(self):
        # We could use the knowledge base or market stats here, but we'll use standard heuristics
        self.annual_depreciation_rate = 0.12

    def calculate_forecast(self, current_value: float) -> List[Dict[str, Any]]:
        from src.utils import format_price_inr

        forecast = []

        for year in range(1, 6):
            retained_pct = (1 - self.annual_depreciation_rate) ** year
            value = current_value * retained_pct
            forecast.append(
                {
                    "year": year,
                    "projected_value_raw": value,
                    "projected_value": format_price_inr(value),
                    "retention_percentage": round(retained_pct * 100, 1),
                }
            )

        return forecast

    def process(self, valuation_report: Dict[str, Any]) -> Dict[str, Any]:
        current_value = valuation_report["estimated_market_value_raw"]
        forecast_timeline = self.calculate_forecast(current_value)
        return {"forecast_timeline": forecast_timeline}
