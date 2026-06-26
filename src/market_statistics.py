import pandas as pd
import numpy as np


class MarketStatistics:
    """
    Market Statistics Layer.
    Computes and caches reusable global market statistics (popularity, depreciation baselines, etc.)
    from the raw training dataset to be consumed by the Feature Engineering layer.
    """

    def __init__(self, df: pd.DataFrame):
        self._compute_statistics(df)

    def _compute_statistics(self, df: pd.DataFrame):
        # We need a car_age proxy if not provided
        if "car_age" not in df.columns:
            from datetime import datetime

            df["car_age"] = datetime.now().year - df["year"]
            df["car_age"] = df["car_age"].clip(lower=0)

        total_cars = len(df)

        # 1. Popularity Features (Frequencies)
        self.brand_popularity = (df["brand"].value_counts() / total_cars).to_dict()
        if "variant" in df.columns:
            self.variant_popularity = (
                df["variant"].value_counts() / total_cars
            ).to_dict()
        else:
            self.variant_popularity = {}

        self.fuel_demand = (df["fuel_type"].value_counts() / total_cars).to_dict()
        self.transmission_popularity = (
            df["transmission"].value_counts() / total_cars
        ).to_dict()

        # 2. Market Stability and Liquidity Baselines
        # Market stability can be proxied by the variance in price for a brand/model
        # Lower variance = higher stability
        brand_price_stats = (
            df.groupby("brand")["selling_price"].agg(["mean", "std"]).reset_index()
        )
        brand_price_stats["cv"] = (
            brand_price_stats["std"] / brand_price_stats["mean"]
        )  # Coefficient of Variation
        max_cv = brand_price_stats["cv"].max()
        # Stability Score: Inverse of CV (normalized 0 to 1, higher is more stable)
        brand_price_stats["stability_score"] = 1 - (brand_price_stats["cv"] / max_cv)
        self.brand_stability = brand_price_stats.set_index("brand")[
            "stability_score"
        ].to_dict()

        # 3. Resale Retention & Depreciation
        # We calculate the median price of a 0-2 year old car vs 5-7 year old car per brand
        df["age_group"] = pd.cut(
            df["car_age"],
            bins=[-1, 2, 4, 7, 10, 100],
            labels=["0-2", "3-4", "5-7", "8-10", "10+"],
        )
        age_grouped = (
            df.groupby(["brand", "age_group"], observed=False)["selling_price"]
            .median()
            .unstack()
        )

        self.brand_retention_score = {}
        self.brand_annual_depreciation_rate = {}

        global_avg_retention = 0.5  # Fallback

        for brand in self.brand_popularity.keys():
            if brand in age_grouped.index:
                prices = age_grouped.loc[brand]
                p_new = prices["0-2"]
                p_mid = prices["5-7"]

                if pd.notna(p_new) and pd.notna(p_mid) and p_new > 0:
                    retention = p_mid / p_new
                    annual_dep_rate = (
                        1 - retention
                    ) / 5.0  # roughly 5 years difference
                else:
                    retention = np.nan
                    annual_dep_rate = np.nan
            else:
                retention = np.nan
                annual_dep_rate = np.nan

            self.brand_retention_score[brand] = retention
            self.brand_annual_depreciation_rate[brand] = annual_dep_rate

        # Fill NaNs with global medians
        median_retention = pd.Series(self.brand_retention_score).median()
        median_dep_rate = pd.Series(self.brand_annual_depreciation_rate).median()

        if pd.isna(median_retention):
            median_retention = global_avg_retention
        if pd.isna(median_dep_rate):
            median_dep_rate = 0.10

        for k in self.brand_retention_score:
            if pd.isna(self.brand_retention_score[k]):
                self.brand_retention_score[k] = median_retention
            if pd.isna(self.brand_annual_depreciation_rate[k]):
                self.brand_annual_depreciation_rate[k] = median_dep_rate

    # Accessors with safe defaults
    def get_brand_popularity(self, brand: str) -> float:
        return self.brand_popularity.get(brand, 0.0)

    def get_variant_popularity(self, variant: str) -> float:
        return self.variant_popularity.get(variant, 0.0)

    def get_fuel_demand(self, fuel: str) -> float:
        return self.fuel_demand.get(fuel, 0.0)

    def get_transmission_popularity(self, trans: str) -> float:
        return self.transmission_popularity.get(trans, 0.0)

    def get_brand_stability(self, brand: str) -> float:
        return self.brand_stability.get(brand, 0.5)

    def get_brand_retention_score(self, brand: str) -> float:
        return self.brand_retention_score.get(brand, 0.5)

    def get_brand_annual_depreciation_rate(self, brand: str) -> float:
        return self.brand_annual_depreciation_rate.get(brand, 0.10)
