"""
Data Processing Module
======================

Handles loading, merging, cleaning, preparing features,
and building the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import (
    BRAND_FILE_MAP,
    EXCHANGE_RATE,
    INDIAN_MARKET_MULTIPLIERS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    CURRENT_YEAR,
    PREMIUM_BRANDS,
    get_data_dir,
    logger,
)



def load_single_brand(filepath: Path, brand_name: str) -> pd.DataFrame:
    """
    Load a single brand's CSV file and add a 'brand' column.

    Args:
        filepath: Path to the CSV file.
        brand_name: Name of the car brand (e.g., 'BMW', 'Audi').

    Returns:
        DataFrame with standardized columns and a 'brand' column.
    """
    df = pd.read_csv(filepath)

    # Standardize column names (some datasets have 'tax(£)' instead of 'tax')
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {"tax(£)": "tax", "fueltype": "fuelType"}
    df.rename(columns=rename_map, inplace=True)

    # Re-capitalize to match expected schema
    col_map = {
        "model": "model",
        "year": "year",
        "price": "price",
        "transmission": "transmission",
        "mileage": "mileage",
        "fueltype": "fuelType",
        "tax": "tax",
        "mpg": "mpg",
        "enginesize": "engineSize",
    }
    df.rename(columns=col_map, inplace=True)

    df["brand"] = brand_name
    logger.info(f"Loaded {brand_name}: {len(df)} records")
    return df


def load_and_merge_datasets() -> pd.DataFrame:
    """
    Load all brand CSV files and merge them into a single DataFrame.

    Returns:
        Merged DataFrame containing all brands with standardized columns.

    Raises:
        FileNotFoundError: If the data directory or required files are missing.
    """
    data_dir = get_data_dir()
    frames = []

    for brand_name, filename in BRAND_FILE_MAP.items():
        filepath = data_dir / filename

        # Handle legacy filenames
        if not filepath.exists():
            legacy_map = {
                "hyundai.csv": "hyundi.csv",
                "mercedes.csv": "merc.csv",
            }
            alt_name = legacy_map.get(filename, filename)
            filepath = data_dir / alt_name

        if filepath.exists():
            df = load_single_brand(filepath, brand_name)
            frames.append(df)
        else:
            logger.warning(f"File not found for {brand_name}: {filepath}")

    if not frames:
        raise FileNotFoundError("No dataset files were loaded successfully.")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Total merged dataset: {len(merged)} records, {merged['brand'].nunique()} brands")
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataset by handling missing values,
    removing duplicates, and filtering outliers.

    Args:
        df: Raw merged DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    initial_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Select only the columns we need
    required_cols = [
        "brand", "model", "year", "price", "transmission",
        "mileage", "fuelType", "mpg", "engineSize",
    ]
    # Keep only columns that exist
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols].copy()

    # Handle missing values
    # Numerical columns: fill with median
    for col in ["mileage", "mpg", "engineSize", "year", "price"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {col} NaN values with median: {median_val}")

    # Categorical columns: fill with mode
    for col in ["transmission", "fuelType"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"Filled {col} NaN values with mode: {mode_val}")

    # Drop rows where 'model' or 'brand' is missing
    df.dropna(subset=["model", "brand"], inplace=True)

    # Remove rows with zero or negative prices
    df = df[df["price"] > 0].copy()

    # Remove extreme outliers using IQR for price
    q1 = df["price"].quantile(0.01)
    q99 = df["price"].quantile(0.99)
    df = df[(df["price"] >= q1) & (df["price"] <= q99)].copy()

    # Remove rows with zero engine size
    df = df[df["engineSize"] > 0].copy()

    logger.info(f"Cleaned dataset: {len(df)} records remaining")
    return df


def convert_price_to_inr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices from British Pounds (£) to Indian Rupees (₹).

    Args:
        df: DataFrame with 'price' column in GBP.

    Returns:
        DataFrame with new 'price_inr' column.
    """
    df["price_inr"] = df["price"] * EXCHANGE_RATE
    
    # Apply Indian market brand-specific multipliers
    def get_multiplier(brand):
        # Fallback to 1.5 if brand not exactly matched
        return INDIAN_MARKET_MULTIPLIERS.get(brand, 1.5)
        
    df["market_multiplier"] = df["brand"].apply(get_multiplier)
    df["price_inr"] = df["price_inr"] * df["market_multiplier"]
    df.drop(columns=["market_multiplier"], inplace=True)
    
    logger.info(
        f"Converted prices to INR with brand multipliers. "
        f"Range: ₹{df['price_inr'].min():,.0f} - ₹{df['price_inr'].max():,.0f}"
    )
    return df


def prepare_data() -> pd.DataFrame:
    """
    Complete data preparation pipeline: load, merge, clean, and convert.

    Returns:
        Fully prepared DataFrame ready for feature engineering.
    """
    df = load_and_merge_datasets()
    df = clean_data(df)
    df = convert_price_to_inr(df)
    return df



def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from the raw dataset.

    Features created:
        - car_age: Age of the car (CURRENT_YEAR - year)

    Args:
        df: Cleaned DataFrame with price_inr column.

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    # Create car_age feature
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["car_age"] = df["car_age"].clip(lower=0)  # ensure non-negative

    # Create km_per_year feature (converting miles to km)
    # 1 mile = 1.60934 km
    km_driven = df["mileage"] * 1.60934
    df["km_per_year"] = km_driven / df["car_age"].replace(0, 0.5)

    # Create premium_brand_flag
    df["premium_brand_flag"] = df["brand"].apply(lambda x: 1 if x in PREMIUM_BRANDS else 0)

    logger.info(
        f"Created car_age feature. Range: {df['car_age'].min()} - {df['car_age'].max()} years"
    )

    # Strip whitespace from string columns
    for col in ["brand", "model", "transmission", "fuelType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


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
        - OneHotEncoder for categorical features (brand, model, transmission, fuelType)
        - StandardScaler for numerical features

    Args:
        X: Feature matrix to determine available columns.

    Returns:
        Fitted-ready ColumnTransformer.
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


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
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

