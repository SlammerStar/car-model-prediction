import pandas as pd
import numpy as np
from src.utils import TARGET_COLUMN, logger

class DataValidator:
    """
    Validates data prior to training to catch leakage, high correlation, and consistency issues.
    """
    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        
    def validate(self, df: pd.DataFrame, features: list):
        logger.info("Starting data validation...")
        self._check_missing_values(df, features)
        self._check_target_leakage(df, features)
        self._check_correlation(df, features)
        logger.info("Data validation passed.")
        
    def _check_missing_values(self, df: pd.DataFrame, features: list):
        missing = df[features].isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            raise ValueError(f"Data validation failed: Missing values detected in features:\n{missing}")

    def _check_target_leakage(self, df: pd.DataFrame, features: list):
        if TARGET_COLUMN in features or 'selling_price' in features:
            raise ValueError(f"Target leakage detected! Target columns found in feature set: {features}")
            
    def _check_correlation(self, df: pd.DataFrame, features: list):
        # Only check numerical features
        num_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        if len(num_features) < 2:
            return
            
        corr_matrix = df[num_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        if high_corr:
            logger.warning(f"Highly correlated features detected (>{self.correlation_threshold}): {high_corr}")
            # We only warn, not fail, as tree models handle correlation okay, 
            # but it is important for the ML Engineer to know.
