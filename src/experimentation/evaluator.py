import time
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)


class ModelEvaluator:
    """
    Evaluates a model via cross-validation and standalone test sets.
    Computes R², MAE, RMSE, MAPE, Median Absolute Error.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def evaluate_pipeline(
        self,
        pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate full training and test sets and return metrics.
        """
        # Training time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Inference time (batch)
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        inference_time = time.time() - start_time

        # Metrics
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            "median_ae": float(median_absolute_error(y_test, y_pred)),
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
        }

        return metrics, y_pred

    def cross_validate_pipeline(
        self, pipeline, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Perform Cross Validation.
        """
        # scoring metrics
        scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        return {
            "cv_r2_mean": float(np.mean(scores["test_r2"])),
            "cv_r2_std": float(np.std(scores["test_r2"])),
            "cv_mae_mean": float(-np.mean(scores["test_neg_mean_absolute_error"])),
            "cv_rmse_mean": float(-np.mean(scores["test_neg_root_mean_squared_error"])),
        }
