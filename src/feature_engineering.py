import pandas as pd
import numpy as np
from src.market_statistics import MarketStatistics
from src.utils import CURRENT_YEAR, PREMIUM_BRANDS


class MarketFeatureEngineer:
    """
    Intelligent Feature Engineering & Market Intelligence Layer.
    Transforms raw vehicle information into market-aware features while preserving raw columns.
    """

    def __init__(self, market_stats: MarketStatistics):
        self.stats = market_stats

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends engineered features to the provided dataset.
        Preserves all original raw features.
        """
        df = df.copy()

        # -----------------------------------------------------
        # 1. Base Transformations (if missing)
        # -----------------------------------------------------
        if "car_age" not in df.columns:
            df["car_age"] = CURRENT_YEAR - df["year"]
            df["car_age"] = df["car_age"].clip(lower=0)

        # -----------------------------------------------------
        # 2. Vehicle Features
        # -----------------------------------------------------
        df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 0.5)

        # Vehicle Segment based on engine size (fallback mapping)
        engine_col = (
            "engineSize"
            if "engineSize" in df.columns
            else ("engine_cc" if "engine_cc" in df.columns else None)
        )
        if engine_col:
            engine_liters = (
                df[engine_col] if df[engine_col].max() < 10 else df[engine_col] / 1000
            )
            conditions = [
                engine_liters < 1.1,
                engine_liters < 1.5,
                engine_liters < 2.0,
                engine_liters >= 2.0,
            ]
            choices = [
                "A-Segment/Hatchback",
                "B-Segment/Compact",
                "C-Segment/Sedan",
                "D-Segment/SUV/Luxury",
            ]
            df["vehicle_segment"] = np.select(conditions, choices, default="Unknown")
        else:
            engine_liters = None
            df["vehicle_segment"] = "Unknown"

        df["luxury_brand_flag"] = df["brand"].apply(
            lambda x: 1 if x in PREMIUM_BRANDS else 0
        )

        # Engine Performance Score (Power to displacement ratio)
        if "max_power_bhp" in df.columns and engine_liters is not None:
            df["engine_performance_score"] = df[
                "max_power_bhp"
            ] / engine_liters.replace(0, np.nan)
            df["engine_performance_score"] = df["engine_performance_score"].fillna(0)
        else:
            df["engine_performance_score"] = 0.0

        # Fuel Efficiency Score (mpg / median_mpg of fuel_type)
        if "mpg" in df.columns or "mileage_kmpl" in df.columns:
            mpg_col = "mpg" if "mpg" in df.columns else "mileage_kmpl"
            fuel_medians = df.groupby("fuel_type")[mpg_col].transform("median")
            df["fuel_efficiency_score"] = df[mpg_col] / fuel_medians.replace(0, np.nan)
            df["fuel_efficiency_score"] = df["fuel_efficiency_score"].fillna(1.0)
        else:
            df["fuel_efficiency_score"] = 1.0

        # -----------------------------------------------------
        # 3. Ownership Features
        # -----------------------------------------------------
        owner_risk_map = {
            "First Owner": 1.0,
            "Second Owner": 0.8,
            "Third Owner": 0.6,
            "Fourth & Above Owner": 0.4,
            "Test Drive Car": 0.9,
        }
        if "owner" in df.columns:
            df["owner_risk_score"] = df["owner"].map(owner_risk_map).fillna(0.7)
        else:
            df["owner_risk_score"] = 0.7

        seller_trust_map = {"Trustmark Dealer": 1.0, "Dealer": 0.8, "Individual": 0.5}
        if "seller_type" in df.columns:
            df["seller_type_score"] = (
                df["seller_type"].map(seller_trust_map).fillna(0.5)
            )
        else:
            df["seller_type_score"] = 0.5

        # -----------------------------------------------------
        # 4. Market Features
        # -----------------------------------------------------
        df["brand_popularity"] = df["brand"].apply(self.stats.get_brand_popularity)
        if "variant" in df.columns:
            df["variant_popularity"] = df["variant"].apply(
                self.stats.get_variant_popularity
            )
        else:
            df["variant_popularity"] = 0.0

        df["fuel_demand_score"] = df["fuel_type"].apply(self.stats.get_fuel_demand)
        df["transmission_popularity"] = df["transmission"].apply(
            self.stats.get_transmission_popularity
        )

        # Configuration Scarcity Score
        # Inverse of variant popularity, meaning rare variants get higher scores
        df["configuration_scarcity_score"] = 1.0 / (df["variant_popularity"] + 1e-5)
        # Normalize to 0-1 within the current dataset
        max_scarcity = df["configuration_scarcity_score"].max()
        if max_scarcity > 0:
            df["configuration_scarcity_score"] = (
                df["configuration_scarcity_score"] / max_scarcity
            )
        else:
            df["configuration_scarcity_score"] = 0.0

        # Market Stability Score
        df["market_stability_score"] = df["brand"].apply(self.stats.get_brand_stability)

        # Market Liquidity Score (How easily a vehicle might sell based on popularity combos)
        df["market_liquidity_score"] = (
            (df["brand_popularity"] * 0.4)
            + (df["variant_popularity"] * 0.3)
            + (df["fuel_demand_score"] * 0.3)
        )

        # -----------------------------------------------------
        # 5. Depreciation Features
        # -----------------------------------------------------
        df["brand_resale_retention_score"] = df["brand"].apply(
            self.stats.get_brand_retention_score
        )
        df["historical_brand_depreciation"] = df["brand"].apply(
            self.stats.get_brand_annual_depreciation_rate
        )

        # Age-based expected depreciation
        df["age_based_depreciation"] = (
            df["car_age"] * df["historical_brand_depreciation"]
        )
        df["age_based_depreciation"] = df["age_based_depreciation"].clip(upper=0.95)

        # Usage-Based expected depreciation
        # Using 150,000 km as a theoretical maximum healthy life baseline for severe penalty
        df["usage_based_depreciation"] = df["km_driven"] / 150000.0
        df["usage_based_depreciation"] = df["usage_based_depreciation"].clip(upper=1.0)

        # Data-driven Remaining Life Index
        # Integrates retention score with age and usage penalties
        df["remaining_life_index"] = (
            (1.0 - df["age_based_depreciation"])
            * (
                1.0 - df["usage_based_depreciation"] * 0.5
            )  # usage has a softer penalty than pure age
            * df["brand_resale_retention_score"]
        )
        df["remaining_life_index"] = df["remaining_life_index"].clip(
            lower=0.0, upper=1.0
        )

        # -----------------------------------------------------
        # 6. Backward Compatibility Aliases
        # -----------------------------------------------------
        if "selling_price" in df.columns:
            df["price_inr"] = df["selling_price"]
        df["mileage"] = df["km_driven"]

        if "mileage_kmpl" in df.columns:
            df["mpg"] = df["mileage_kmpl"].fillna(0)

        df["engineSize"] = engine_liters if engine_liters is not None else 0.0
        df["fuelType"] = df["fuel_type"]

        # Strip whitespace from string columns to prevent trailing space bugs
        for col in [
            "brand",
            "model",
            "transmission",
            "fuel_type",
            "fuelType",
            "variant",
            "base_model",
        ]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df
