import pytest
import pandas as pd
from src.market_statistics import MarketStatistics
from src.feature_engineering import MarketFeatureEngineer
from src.utils import CURRENT_YEAR


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "brand": "Maruti",
                "model": "Swift",
                "variant": "Dzire VDI",
                "year": CURRENT_YEAR - 5,
                "selling_price": 500000,
                "km_driven": 50000,
                "fuel_type": "Diesel",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
                "mileage_kmpl": 23.4,
                "engine_cc": 1248,
                "max_power_bhp": 74.0,
                "seats": 5.0,
            },
            {
                "brand": "BMW",
                "model": "X5",
                "variant": "xDrive 30d",
                "year": CURRENT_YEAR - 2,
                "selling_price": 5000000,
                "km_driven": 20000,
                "fuel_type": "Diesel",
                "seller_type": "Dealer",
                "transmission": "Automatic",
                "owner": "First Owner",
                "mileage_kmpl": 15.0,
                "engine_cc": 2993,
                "max_power_bhp": 265.0,
                "seats": 5.0,
            },
            {
                "brand": "Maruti",
                "model": "Alto",
                "variant": "800 LXI",
                "year": CURRENT_YEAR - 1,
                "selling_price": 300000,
                "km_driven": 10000,
                "fuel_type": "Petrol",
                "seller_type": "Trustmark Dealer",
                "transmission": "Manual",
                "owner": "Second Owner",
                "mileage_kmpl": 22.05,
                "engine_cc": 796,
                "max_power_bhp": 47.3,
                "seats": 5.0,
            },
        ]
    )


def test_market_statistics(sample_df):
    stats = MarketStatistics(sample_df)

    assert stats.get_brand_popularity("Maruti") == 2.0 / 3.0
    assert stats.get_brand_popularity("BMW") == 1.0 / 3.0

    assert stats.get_variant_popularity("Dzire VDI") == 1.0 / 3.0
    assert stats.get_fuel_demand("Diesel") == 2.0 / 3.0

    assert stats.get_transmission_popularity("Manual") == 2.0 / 3.0


def test_feature_engineering(sample_df):
    stats = MarketStatistics(sample_df)
    engineer = MarketFeatureEngineer(stats)
    df_feat = engineer.engineer_features(sample_df)

    # 1. Vehicle Features
    assert df_feat.loc[0, "car_age"] == 5
    assert df_feat.loc[0, "km_per_year"] == 10000
    assert df_feat.loc[1, "luxury_brand_flag"] == 1
    assert df_feat.loc[0, "luxury_brand_flag"] == 0
    assert df_feat.loc[0, "vehicle_segment"] == "B-Segment/Compact"  # 1.248 L
    assert df_feat.loc[1, "vehicle_segment"] == "D-Segment/SUV/Luxury"  # 2.993 L
    assert df_feat.loc[2, "vehicle_segment"] == "A-Segment/Hatchback"  # 0.796 L

    # 2. Ownership Features
    assert df_feat.loc[0, "owner_risk_score"] == 1.0  # First owner
    assert df_feat.loc[2, "owner_risk_score"] == 0.8  # Second owner
    assert df_feat.loc[2, "seller_type_score"] == 1.0  # Trustmark Dealer

    # 3. Market Features
    assert df_feat.loc[0, "brand_popularity"] == 2.0 / 3.0
    assert df_feat.loc[0, "market_liquidity_score"] > 0

    # 4. Depreciation Features
    assert "remaining_life_index" in df_feat.columns
    assert "age_based_depreciation" in df_feat.columns
    assert "historical_brand_depreciation" in df_feat.columns

    # Check backwards compatibility
    assert "price_inr" in df_feat.columns
    assert "mileage" in df_feat.columns
    assert "mpg" in df_feat.columns
    assert "engineSize" in df_feat.columns
    assert "fuelType" in df_feat.columns
