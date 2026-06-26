import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.utils import logger, RANDOM_STATE


class HyperparameterOptimizer:
    """
    Handles Bayesian Hyperparameter optimization using Optuna.
    """

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        n_trials: int = 20,
        timeout: int = 600,
        n_jobs: int = 1,
    ):
        self.preprocessor = preprocessor
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        # Make Optuna less verbose
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(
        self, model_name: str, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        logger.info(f"Starting Optuna optimization for {model_name}...")

        def objective(trial):
            params = self._get_search_space(model_name, trial)
            model = self._get_model_instance(model_name, params)

            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", model)]
            )

            # Sub-sample data for faster HP search if dataset is very large
            if len(X) > 10000:
                idx = np.random.RandomState(RANDOM_STATE).choice(
                    X.index, 10000, replace=False
                )
                X_search, y_search = X.loc[idx], y.loc[idx]
            else:
                X_search, y_search = X, y

            cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            # Use negative MAE for robustness against outliers
            scores = cross_val_score(
                pipeline,
                X_search,
                y_search,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=self.n_jobs,
            )
            return scores.mean()  # we want to maximize the negative MAE (closer to 0)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_model = self._get_model_instance(model_name, best_params)

        best_pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("regressor", best_model)]
        )

        logger.info(
            f"Optimization finished for {model_name}. Best params: {best_params}"
        )
        return best_pipeline, best_params

    def _get_search_space(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        if model_name == "DecisionTree":
            return {
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
        elif model_name == "RandomForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            }
        elif model_name == "ExtraTrees":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            }
        elif model_name == "GradientBoosting":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
            }
        elif model_name == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        elif model_name == "LightGBM":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", -1, 15),
            }
        elif model_name == "CatBoost":
            return {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.3, log=True
                ),
                "depth": trial.suggest_int("depth", 4, 10),
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        if model_name == "DecisionTree":
            return DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
        elif model_name == "RandomForest":
            return RandomForestRegressor(random_state=RANDOM_STATE, **params)
        elif model_name == "ExtraTrees":
            return ExtraTreesRegressor(random_state=RANDOM_STATE, **params)
        elif model_name == "GradientBoosting":
            return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        elif model_name == "XGBoost":
            return xgb.XGBRegressor(random_state=RANDOM_STATE, **params)
        elif model_name == "LightGBM":
            return lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **params)
        elif model_name == "CatBoost":
            return cb.CatBoostRegressor(
                random_state=RANDOM_STATE, verbose=False, **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
