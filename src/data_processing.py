"""
Data Processing Module — DRIVEIQ 2.0
=====================================

Production-quality data engineering pipeline for Indian used-car valuation.

This module handles:
    - Loading raw datasets (CarDekho v3, Kasliwal multi-city)
    - Parsing compound fields (name → brand + model, engine CC, power bhp, mileage kmpl)
    - Standardizing column names and categorical values
    - Handling missing values with domain-aware imputation
    - Removing duplicates, invalid records, and outliers
    - Merging multiple datasets with deduplication
    - Producing a backward-compatible unified dataset

Data Sources:
    - Dataset A: CarDekho v3 (nehalbirla/vehicle-dataset-from-cardekho) — ODbL License
    - Dataset B: Kasliwal Multi-City (avikasliwal/used-cars-price-prediction) — Public/Kaggle
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    CURRENT_YEAR,
    get_data_dir,
    logger,
)

# ---------------------------------------------------------------------------
# Constants for data processing
# ---------------------------------------------------------------------------

# Brand name normalization map
BRAND_NORMALIZE = {
    "Land": "Land Rover",
    "ISUZU": "Isuzu",
    "Mercedes-Benz": "Mercedes",
}

# Fuel type normalization
FUEL_NORMALIZE = {
    "Diesel": "Diesel",
    "Petrol": "Petrol",
    "CNG": "CNG",
    "LPG": "LPG",
    "Electric": "Electric",
}

# Transmission normalization
TRANSMISSION_NORMALIZE = {
    "Manual": "Manual",
    "Automatic": "Automatic",
}

# Owner type normalization
OWNER_NORMALIZE = {
    "First Owner": "First Owner",
    "First": "First Owner",
    "Second Owner": "Second Owner",
    "Second": "Second Owner",
    "Third Owner": "Third Owner",
    "Third": "Third Owner",
    "Fourth & Above Owner": "Fourth & Above Owner",
    "Fourth & Above": "Fourth & Above Owner",
    "Test Drive Car": "Test Drive Car",
}


# ---------------------------------------------------------------------------
# Parsing Utilities
# ---------------------------------------------------------------------------


def parse_brand_model(name: str) -> Tuple[str, str, str, str]:
    """
    Extract brand, model, base_model, and variant from a combined name string.

    Handles special cases like 'Land Rover', 'Mercedes-Benz', etc.

    Args:
        name: Combined car name (e.g., 'Maruti Swift Dzire VDI').

    Returns:
        Tuple of (brand, model, base_model, variant).
    """
    if not isinstance(name, str) or not name.strip():
        return "Unknown", "Unknown", "Unknown", "Standard"

    parts = name.strip().split()
    if len(parts) == 0:
        return "Unknown", "Unknown", "Unknown", "Standard"

    # Handle multi-word brands
    if len(parts) >= 2 and parts[0] == "Land" and parts[1] == "Rover":
        brand = "Land Rover"
        rest = parts[2:]
    elif len(parts) >= 2 and parts[0] == "Ashok" and parts[1] == "Leyland":
        brand = "Ashok Leyland"
        rest = parts[2:]
    else:
        brand = parts[0]
        rest = parts[1:]

    # Normalize brand names
    brand = BRAND_NORMALIZE.get(brand, brand)

    if not rest:
        return brand, "Unknown", "Unknown", "Standard"

    # Clean model: take first 2-3 meaningful words to reduce cardinality for ML
    model = " ".join(rest[:3])

    # Base model is the first word of the rest
    base_model = rest[0]

    # Variant is everything after the base model
    variant = " ".join(rest[1:]) if len(rest) > 1 else "Standard"

    return brand.strip(), model.strip(), base_model.strip(), variant.strip()


def parse_numeric_with_unit(value, unit_to_strip: str = "") -> Optional[float]:
    """
    Parse a numeric value from a string that may contain units.

    Args:
        value: Value to parse (e.g., '1248 CC', '74 bhp', '23.4 kmpl').
        unit_to_strip: Unit suffix to remove before parsing.

    Returns:
        Parsed float value, or None if parsing fails.
    """
    if pd.isna(value) or not isinstance(value, str):
        return None

    cleaned = value.strip()
    if unit_to_strip:
        cleaned = cleaned.replace(unit_to_strip, "").strip()

    # Remove common unit suffixes
    for suffix in ["CC", "cc", "bhp", "BHP", "kmpl", "km/kg", "kmkg"]:
        cleaned = cleaned.replace(suffix, "").strip()

    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_mileage_kmpl(value) -> Optional[float]:
    """
    Parse fuel efficiency from various formats to km/l.

    Handles: '23.4 kmpl', '26.6 km/kg' (CNG/LPG).

    Args:
        value: Mileage string.

    Returns:
        Fuel efficiency in kmpl (float), or None.
    """
    if pd.isna(value) or not isinstance(value, str):
        return None

    cleaned = value.strip().lower()

    # Remove units and parse
    for unit in ["kmpl", "km/kg", "km/l"]:
        if unit in cleaned:
            cleaned = cleaned.replace(unit, "").strip()
            break

    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Dataset A: CarDekho v3 Loader
# ---------------------------------------------------------------------------


def load_cardekho_v3(filepath: Path) -> pd.DataFrame:
    """
    Load and standardize the CarDekho v3 dataset.

    Source: kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
    License: Open Database License (ODbL) v1.0

    Args:
        filepath: Path to cardekho_v3.csv.

    Returns:
        DataFrame with standardized column names.
    """
    logger.info(f"Loading CarDekho v3 from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"  Raw records: {len(df)}")

    # Parse brand and model from 'name'
    parsed = df["name"].apply(parse_brand_model)
    df["brand"] = parsed.apply(lambda x: x[0])
    df["model"] = parsed.apply(lambda x: x[1])
    df["base_model"] = parsed.apply(lambda x: x[2])
    df["variant"] = parsed.apply(lambda x: x[3])

    # Parse numeric fields from strings
    df["mileage_kmpl"] = df["mileage"].apply(parse_mileage_kmpl)
    df["engine_cc"] = df["engine"].apply(lambda x: parse_numeric_with_unit(x, "CC"))
    df["max_power_bhp"] = df["max_power"].apply(
        lambda x: parse_numeric_with_unit(x, "bhp")
    )

    # Rename columns to unified schema
    df = df.rename(
        columns={
            "selling_price": "selling_price",
            "km_driven": "km_driven",
            "fuel": "fuel_type",
            "transmission": "transmission",
            "owner": "owner_type",
            "seats": "seats",
        }
    )

    # Normalize categorical values
    df["fuel_type"] = df["fuel_type"].map(FUEL_NORMALIZE).fillna(df["fuel_type"])
    df["transmission"] = (
        df["transmission"].map(TRANSMISSION_NORMALIZE).fillna(df["transmission"])
    )
    df["owner_type"] = df["owner_type"].map(OWNER_NORMALIZE).fillna(df["owner_type"])

    # Add metadata
    df["seller_type"] = df.get("seller_type", pd.Series(dtype=str))
    df["location"] = None
    df["source"] = "cardekho_v3"

    # Select unified columns
    unified_cols = [
        "brand",
        "model",
        "base_model",
        "variant",
        "year",
        "selling_price",
        "km_driven",
        "fuel_type",
        "transmission",
        "owner_type",
        "mileage_kmpl",
        "engine_cc",
        "max_power_bhp",
        "seats",
        "seller_type",
        "location",
        "source",
    ]
    df = df[[c for c in unified_cols if c in df.columns]].copy()

    logger.info(f"  Standardized records: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Dataset B: Kasliwal Multi-City Loader
# ---------------------------------------------------------------------------


def load_kasliwal_multicity(filepath: Path) -> pd.DataFrame:
    """
    Load and standardize the Kasliwal multi-city dataset.

    Source: kaggle.com/datasets/avikasliwal/used-cars-price-prediction
    License: Public/Kaggle (listed as Other)

    Args:
        filepath: Path to kasliwal_train.csv.

    Returns:
        DataFrame with standardized column names.
    """
    logger.info(f"Loading Kasliwal multi-city from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"  Raw records: {len(df)}")

    # Drop index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Parse brand and model from 'Name'
    parsed = df["Name"].apply(parse_brand_model)
    df["brand"] = parsed.apply(lambda x: x[0])
    df["model"] = parsed.apply(lambda x: x[1])
    df["base_model"] = parsed.apply(lambda x: x[2])
    df["variant"] = parsed.apply(lambda x: x[3])

    # Convert price from Lakhs to raw INR
    df["selling_price"] = (df["Price"] * 100_000).astype(int)

    # Parse numeric fields
    df["mileage_kmpl"] = df["Mileage"].apply(parse_mileage_kmpl)
    df["engine_cc"] = df["Engine"].apply(lambda x: parse_numeric_with_unit(x, "CC"))
    df["max_power_bhp"] = df["Power"].apply(
        lambda x: (
            parse_numeric_with_unit(x, "bhp")
            if isinstance(x, str) and x.strip().lower() != "null"
            else None
        )
    )

    # Rename columns to unified schema
    df = df.rename(
        columns={
            "Year": "year",
            "Kilometers_Driven": "km_driven",
            "Fuel_Type": "fuel_type",
            "Transmission": "transmission",
            "Owner_Type": "owner_type",
            "Seats": "seats",
            "Location": "location",
        }
    )

    # Normalize categorical values
    df["fuel_type"] = df["fuel_type"].map(FUEL_NORMALIZE).fillna(df["fuel_type"])
    df["transmission"] = (
        df["transmission"].map(TRANSMISSION_NORMALIZE).fillna(df["transmission"])
    )
    df["owner_type"] = df["owner_type"].map(OWNER_NORMALIZE).fillna(df["owner_type"])

    # Add metadata
    df["seller_type"] = None
    df["source"] = "kasliwal_multicity"

    # Select unified columns
    unified_cols = [
        "brand",
        "model",
        "base_model",
        "variant",
        "year",
        "selling_price",
        "km_driven",
        "fuel_type",
        "transmission",
        "owner_type",
        "mileage_kmpl",
        "engine_cc",
        "max_power_bhp",
        "seats",
        "seller_type",
        "location",
        "source",
    ]
    df = df[[c for c in unified_cols if c in df.columns]].copy()

    logger.info(f"  Standardized records: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Unified Data Engineering Pipeline
# ---------------------------------------------------------------------------


def load_and_merge_datasets() -> pd.DataFrame:
    """
    Load all approved datasets and merge into a single unified DataFrame.

    Returns:
        Merged DataFrame with standardized columns.

    Raises:
        FileNotFoundError: If no dataset files are found.
    """
    data_dir = get_data_dir()
    raw_dir = data_dir / "raw"
    frames = []

    # Load Dataset A: CarDekho v3
    cardekho_path = raw_dir / "cardekho_v3.csv"
    if cardekho_path.exists():
        df_a = load_cardekho_v3(cardekho_path)
        frames.append(df_a)
    else:
        logger.warning(f"CarDekho v3 not found at {cardekho_path}")

    # Load Dataset B: Kasliwal multi-city
    kasliwal_path = raw_dir / "kasliwal_train.csv"
    if kasliwal_path.exists():
        df_b = load_kasliwal_multicity(kasliwal_path)
        frames.append(df_b)
    else:
        logger.warning(f"Kasliwal multi-city not found at {kasliwal_path}")

    if not frames:
        raise FileNotFoundError(
            "No dataset files found. Ensure CSV files are in 'data/raw/'."
        )

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Merged dataset: {len(merged)} records from {len(frames)} sources, "
        f"{merged['brand'].nunique()} brands"
    )
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataset with domain-aware rules for Indian used cars.

    Steps:
        1. Remove exact duplicates
        2. Remove records with invalid/missing critical fields
        3. Remove unrealistic price values
        4. Remove unrealistic mileage (km_driven) values
        5. Remove unrealistic year values
        6. Handle missing numeric values with brand-aware median imputation
        7. Standardize brand names
        8. Remove near-duplicate listings

    Args:
        df: Raw merged DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial_rows = len(df)
    logger.info(f"Starting cleaning: {initial_rows} records")

    # Step 1: Remove exact duplicates
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    logger.info(f"  Removed {dupes_removed} exact duplicates")

    # Step 2: Remove records missing critical fields
    critical_cols = [
        "brand",
        "model",
        "year",
        "selling_price",
        "km_driven",
        "fuel_type",
        "transmission",
    ]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    df = df[df["brand"] != "Unknown"]
    df = df[df["model"] != "Unknown"]
    logger.info(f"  Removed {before - len(df)} records with missing critical fields")

    # Step 3: Remove unrealistic prices
    # Indian used-car prices: ₹20,000 (very old/damaged) to ₹2 Crore (luxury)
    before = len(df)
    df = df[(df["selling_price"] >= 20_000) & (df["selling_price"] <= 2_00_00_000)]
    logger.info(f"  Removed {before - len(df)} records with unrealistic prices")

    # Step 4: Remove unrealistic mileage
    # Reasonable range: 100 km (nearly new) to 10,00,000 km (commercial vehicles)
    before = len(df)
    df = df[(df["km_driven"] >= 100) & (df["km_driven"] <= 10_00_000)]
    logger.info(f"  Removed {before - len(df)} records with unrealistic km_driven")

    # Step 5: Remove unrealistic years
    before = len(df)
    df = df[(df["year"] >= 1995) & (df["year"] <= CURRENT_YEAR)]
    logger.info(f"  Removed {before - len(df)} records with unrealistic year")

    # Step 6: Handle missing numeric values — brand-aware median imputation
    for col in ["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Use brand-level median, falling back to global median
                brand_medians = df.groupby("brand")[col].transform("median")
                global_median = df[col].median()
                df[col] = df[col].fillna(brand_medians).fillna(global_median)
                logger.info(
                    f"  Imputed {missing_count} missing {col} values "
                    f"(brand-median, fallback global median: {global_median:.1f})"
                )

    # Step 7: Standardize brand names
    df["brand"] = df["brand"].apply(lambda x: BRAND_NORMALIZE.get(x, x))

    # Step 8: Remove near-duplicate listings (same brand+model+year+km within 5%)
    before = len(df)
    df = df.sort_values("selling_price")
    df["_dedup_key"] = (
        df["brand"]
        + "|"
        + df["model"]
        + "|"
        + df["year"].astype(str)
        + "|"
        + df["fuel_type"]
        + "|"
        + df["transmission"]
    )
    # For same key, remove records with km_driven within 2% and price within 5%
    mask = pd.Series(True, index=df.index)
    seen = {}
    for idx, row in df.iterrows():
        key = row["_dedup_key"]
        km = row["km_driven"]
        price = row["selling_price"]
        if key in seen:
            for prev_km, prev_price in seen[key]:
                km_close = abs(km - prev_km) < max(500, 0.02 * prev_km)
                price_close = abs(price - prev_price) < 0.05 * prev_price
                if km_close and price_close:
                    mask[idx] = False
                    break
            if mask[idx]:
                seen[key].append((km, price))
        else:
            seen[key] = [(km, price)]

    df = df[mask].drop(columns=["_dedup_key"])
    logger.info(f"  Removed {before - len(df)} near-duplicate listings")

    # Step 9: Statistical outlier removal using IQR per brand for price
    before = len(df)
    clean_frames = []
    for brand, group in df.groupby("brand"):
        if len(group) < 10:
            # Too few samples — keep all
            clean_frames.append(group)
            continue
        q1 = group["selling_price"].quantile(0.02)
        q99 = group["selling_price"].quantile(0.98)
        filtered = group[
            (group["selling_price"] >= q1) & (group["selling_price"] <= q99)
        ]
        clean_frames.append(filtered)
    df = pd.concat(clean_frames, ignore_index=True)
    logger.info(f"  Removed {before - len(df)} brand-level price outliers")

    logger.info(
        f"Cleaning complete: {len(df)} records remaining "
        f"({initial_rows - len(df)} removed total)"
    )
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the Indian market dataset.

    Features are now intelligently engineered using the Phase 4
    Market Intelligence Layer, preserving raw features.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with additional engineered features.
    """
    from src.market_statistics import MarketStatistics
    from src.feature_engineering import MarketFeatureEngineer

    # 1. Calculate and cache global market statistics
    stats = MarketStatistics(df)

    # 2. Instantiate intelligent feature engineer
    engineer = MarketFeatureEngineer(stats)

    # 3. Transform features
    df_engineered = engineer.engineer_features(df)

    logger.info(
        f"Engineered intelligent features. car_age range: {df_engineered['car_age'].min()}-{df_engineered['car_age'].max()} years, "
        f"brands: {df_engineered['brand'].nunique()}, models: {df_engineered['model'].nunique()}"
    )

    return df_engineered


def convert_price_to_inr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure price_inr column exists (identity for Indian data).

    In DRIVEIQ v1, this converted GBP to INR with multipliers.
    In DRIVEIQ v2, prices are already in INR — this is a no-op for compatibility.

    Args:
        df: DataFrame with selling_price in INR.

    Returns:
        DataFrame with price_inr column (same as selling_price).
    """
    if "price_inr" not in df.columns:
        df["price_inr"] = df["selling_price"]
    logger.info(
        f"Prices verified in INR. "
        f"Range: ₹{df['price_inr'].min():,.0f} – ₹{df['price_inr'].max():,.0f}"
    )
    return df


def prepare_data() -> pd.DataFrame:
    """
    Complete data preparation pipeline: load, merge, clean, and create features.

    Returns:
        Fully prepared DataFrame ready for feature engineering and training.
    """
    df = load_and_merge_datasets()
    df = clean_data(df)
    df = convert_price_to_inr(df)
    return df


# ---------------------------------------------------------------------------
# Feature / Target Splitting
# ---------------------------------------------------------------------------


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix (X) and target vector (y).

    Args:
        df: DataFrame with all features and target column.

    Returns:
        Tuple of (X, y) where X contains feature columns and y is the target.
    """
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df[TARGET_COLUMN].copy()

    logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing features.

    Uses:
        - OneHotEncoder for categorical features
        - StandardScaler for numerical features

    Args:
        X: Feature matrix to determine available columns.

    Returns:
        ColumnTransformer ready to fit.
    """
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_features = [c for c in NUMERICAL_FEATURES if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
            (
                "numerical",
                StandardScaler(),
                num_features,
            ),
        ],
        remainder="drop",
    )

    logger.info(
        f"Preprocessor built - Categorical: {cat_features}, Numerical: {num_features}"
    )
    return preprocessor


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Full feature preparation pipeline.

    Args:
        df: Cleaned and price-converted DataFrame.

    Returns:
        Tuple of (X, y, preprocessor).
    """
    df = create_features(df)
    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(X)
    return X, y, preprocessor
