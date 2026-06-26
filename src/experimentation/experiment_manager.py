import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.data_processing import prepare_data
from src.experimentation.data_validator import DataValidator
from src.experimentation.evaluator import ModelEvaluator
from src.experimentation.optimizer import HyperparameterOptimizer
from src.experimentation.registry import ModelRegistry
from src.utils import TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, logger

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ExperimentManager:
    """
    Central controller for the ML Experimentation framework.
    """

    def __init__(self, config_path: str = "src/experimentation/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.registry = ModelRegistry()
        self.validator = DataValidator()
        self.evaluator = ModelEvaluator(
            n_splits=self.config["validation"]["n_splits"],
            shuffle=self.config["validation"]["shuffle"],
            random_state=self.config["random_seed"],
        )
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []

    def _build_preprocessor(self, features: list) -> ColumnTransformer:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in features]
        num_features = [c for c in features if c not in CATEGORICAL_FEATURES]

        return ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_features,
                ),
                ("numerical", StandardScaler(), num_features),
            ],
            remainder="drop",
        )

    def run_experiments(self):
        logger.info(f"Starting Experiment ID: {self.experiment_id}")

        # 1. Dataset Prep
        df = prepare_data()

        from src.market_statistics import MarketStatistics
        from src.feature_engineering import MarketFeatureEngineer

        self.stats = MarketStatistics(df)
        engineer = MarketFeatureEngineer(self.stats)
        df = engineer.engineer_features(df)

        for subset_name, features in self.config["feature_subsets"].items():
            logger.info(f"--- Evaluating Feature Subset: {subset_name} ---")

            # 2. Data Validation
            self.validator.validate(df, features)

            X = df[features]
            y = df[TARGET_COLUMN]

            # Preprocessor for this subset
            preprocessor = self._build_preprocessor(features)

            optimizer = HyperparameterOptimizer(
                preprocessor=preprocessor,
                n_trials=self.config["optimization"]["n_trials"],
                timeout=self.config["optimization"]["timeout_seconds"],
                n_jobs=self.config["optimization"]["n_jobs"],
            )

            # Train Test split for final evaluation
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config["random_seed"]
            )

            for model_name in self.config["models_to_evaluate"]:
                try:
                    logger.info(f"Evaluating {model_name} on {subset_name}")

                    # 3. Hyperparameter Optimization & Training
                    best_pipeline, best_params = optimizer.optimize(
                        model_name, X_train, y_train
                    )

                    # 4. Evaluation
                    # Cross validation on full training set
                    cv_metrics = self.evaluator.cross_validate_pipeline(
                        best_pipeline, X_train, y_train
                    )

                    # Standalone test evaluation
                    test_metrics, _ = self.evaluator.evaluate_pipeline(
                        best_pipeline, X_train, y_train, X_test, y_test
                    )

                    all_metrics = {**cv_metrics, **test_metrics}

                    # 5. Metadata and Tracking
                    metadata = {
                        "experiment_id": self.experiment_id,
                        "model_name": model_name,
                        "feature_subset": subset_name,
                        "features": features,
                        "params": best_params,
                        "metrics": all_metrics,
                        "random_seed": self.config["random_seed"],
                    }

                    # Feature Importance Extraction
                    importances = None
                    model_step = best_pipeline.named_steps["regressor"]
                    if hasattr(model_step, "feature_importances_"):
                        importances = model_step.feature_importances_

                    if importances is not None:
                        # Map back to feature names using the preprocessor
                        # We try to get feature names out, if preprocessor supports it
                        try:
                            # Simple assumption for feature names (can be complex with OHE)
                            cat_transformer = best_pipeline.named_steps[
                                "preprocessor"
                            ].named_transformers_.get("categorical")
                            if cat_transformer is not None and hasattr(
                                cat_transformer, "get_feature_names_out"
                            ):
                                cat_names = (
                                    cat_transformer.get_feature_names_out().tolist()
                                )
                            else:
                                cat_names = []
                            num_names = [
                                c for c in features if c not in CATEGORICAL_FEATURES
                            ]
                            feature_names = cat_names + num_names

                            if len(feature_names) == len(importances):
                                metadata["feature_importances"] = dict(
                                    zip(feature_names, importances.tolist())
                                )
                        except Exception as e:
                            logger.warning(f"Failed to map feature importances: {e}")

                    # Save candidate
                    candidate_path = self.registry.save_candidate(
                        self.experiment_id,
                        f"{model_name}_{subset_name}",
                        best_pipeline,
                        metadata,
                    )

                    self.results.append(
                        {
                            "Model": model_name,
                            "Features": subset_name,
                            "R2": all_metrics["r2"],
                            "CV_R2_Mean": all_metrics["cv_r2_mean"],
                            "MAE": all_metrics["mae"],
                            "MAPE": all_metrics["mape"],
                            "Train_Time": all_metrics["train_time_sec"],
                            "Path": candidate_path,
                            "Metadata": metadata,
                        }
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate {model_name} on {subset_name}: {e}"
                    )

        self._generate_leaderboard()
        self._promote_best_model()

    def _generate_leaderboard(self):
        if not self.results:
            logger.warning("No results to generate leaderboard.")
            return

        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(
            by="CV_R2_Mean", ascending=False
        ).reset_index(drop=True)

        # Save CSV
        out_dir = self.registry.candidates_dir / self.experiment_id
        csv_path = out_dir / "leaderboard.csv"
        df_results.drop(columns=["Path", "Metadata"]).to_csv(csv_path, index=False)

        # Save Markdown
        md_path = out_dir / "leaderboard.md"
        with open(md_path, "w") as f:
            f.write(f"# Experiment Leaderboard: {self.experiment_id}\n\n")
            f.write(df_results.drop(columns=["Path", "Metadata"]).to_markdown())

        logger.info(f"Leaderboard saved to {csv_path}")

    def _promote_best_model(self):
        if not self.results:
            return

        # Best model is the one with highest CV R2
        best_run = max(self.results, key=lambda x: x["CV_R2_Mean"])

        logger.info(
            f"Selecting best model: {best_run['Model']} ({best_run['Features']}) with CV R2: {best_run['CV_R2_Mean']:.4f}"
        )

        self.registry.promote_to_production(
            candidate_path=best_run["Path"], metadata=best_run["Metadata"]
        )

        # Save the market statistics object for inference
        import joblib

        stats_path = self.registry.production_dir / "market_stats.pkl"
        joblib.dump(self.stats, stats_path)
        logger.info(f"Saved market statistics to {stats_path}")


if __name__ == "__main__":
    manager = ExperimentManager()
    manager.run_experiments()
