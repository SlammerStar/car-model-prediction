from typing import Dict, Any


class OwnershipEngine:
    def __init__(self):
        # Configurable assumptions
        self.annual_insurance_rate = 0.03
        self.annual_maintenance_base = 15000
        self.annual_distance_km = 12000
        self.fuel_price_per_liter = 100

    def calculate_ownership_cost(
        self, valuation_report: Dict[str, Any], input_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        est_value = valuation_report["estimated_market_value_raw"]

        # 1. Insurance (approx 3% of current value annually)
        insurance_5y = est_value * self.annual_insurance_rate * 5

        # 2. Maintenance (scales with car age and engine size)
        car_age = input_summary.get("car_age", 5)
        engine_size = input_summary.get("engineSize", 1.2)
        annual_maintenance = (
            self.annual_maintenance_base + (car_age * 2000) + (engine_size * 5000)
        )
        maintenance_5y = annual_maintenance * 5

        # 3. Fuel
        mpg = input_summary.get("mpg", 15.0)
        if mpg <= 0:
            mpg = 15.0
        annual_fuel = (self.annual_distance_km / mpg) * self.fuel_price_per_liter
        fuel_5y = annual_fuel * 5

        # 4. Registration / Misc
        registration_5y = 10000  # Flat estimate for 5 years

        total_5y = insurance_5y + maintenance_5y + fuel_5y + registration_5y

        from src.utils import format_price_inr

        return {
            "insurance_5y": format_price_inr(insurance_5y),
            "maintenance_5y": format_price_inr(maintenance_5y),
            "fuel_5y": format_price_inr(fuel_5y),
            "registration_misc_5y": format_price_inr(registration_5y),
            "total_5y": format_price_inr(total_5y),
            "assumptions": {
                "annual_distance_km": self.annual_distance_km,
                "fuel_price": self.fuel_price_per_liter,
                "mpg_used": mpg,
            },
        }

    def process(
        self, valuation_report: Dict[str, Any], input_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.calculate_ownership_cost(valuation_report, input_summary)
