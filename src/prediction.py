"""
Prediction Module
=================

Trains models, evaluates them, and makes predictions.
Consolidated from train_model, evaluate_model, and predict.
"""

import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

try:
    import xgboost  # noqa: F401

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.utils import (
    RANDOM_STATE,
    TEST_SIZE,
    PIPELINE_PATH,
    METADATA_PATH,
    IMAGES_DIR,
    CURRENT_YEAR,
    PREMIUM_BRANDS,
    format_price_inr,
    save_model,
    load_model,
    logger,
)

from src.data_processing import (
    prepare_data,
    prepare_features,
    create_features,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def get_models() -> Dict[str, Any]:
    """
    Define the set of regression models to train and compare.

    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    }

    return models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with R², MAE, and RMSE scores.
    """
    return {
        "R2_Score": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
    }


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor,
) -> pd.DataFrame:
    """
    Train all models and return a comparison DataFrame.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_test: Test target vector.
        preprocessor: Fitted ColumnTransformer.

    Returns:
        DataFrame with model names and their evaluation metrics, sorted by R².
    """
    models = get_models()
    results = []

    # Setup MLflow experiment
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Car_Price_Prediction")

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Create pipeline
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", model),
            ]
        )

        # MLflow tracking
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=name):
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                metrics = evaluate_model(y_test, y_pred)

                # Log parameters
                mlflow.log_param("model_type", name)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

                logger.info(f"{name} - {metrics}")
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
            logger.info(f"{name} - {metrics}")

        metrics["Model"] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[["Model", "R2_Score", "MAE", "RMSE"]]
    comparison_df = comparison_df.sort_values("R2_Score", ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)

    return comparison_df


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    best_model_name: str = "Random Forest",
) -> Pipeline:
    """
    Perform hyperparameter tuning on the best model using RandomizedSearchCV.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        preprocessor: ColumnTransformer for preprocessing.
        best_model_name: Name of the best-performing model to tune.

    Returns:
        Tuned sklearn Pipeline.
    """
    logger.info(f"Tuning {best_model_name}...")

    # Define hyperparameter grids
    param_grids = {
        "Random Forest": {
            "regressor__n_estimators": [100, 200, 300, 500],
            "regressor__max_depth": [10, 15, 20, 25, None],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [3, 5, 7, 10],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__subsample": [0.8, 0.9, 1.0],
        },
        "XGBoost": {
            "regressor__n_estimators": [100, 200, 300, 500],
            "regressor__max_depth": [3, 5, 7, 10],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__subsample": [0.8, 0.9, 1.0],
            "regressor__colsample_bytree": [0.8, 0.9, 1.0],
        },
    }

    # Select model and params
    models_map = get_models()
    if best_model_name not in models_map:
        best_model_name = "Random Forest"

    base_model = models_map[best_model_name]
    param_grid = param_grids.get(best_model_name, param_grids["Random Forest"])

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", base_model),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=30,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV R² score: {search.best_score_:.4f}")

    # Log best run to MLflow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"{best_model_name}_Tuned"):
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_r2", search.best_score_)
            mlflow.sklearn.log_model(
                search.best_estimator_,
                "tuned_model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

    return search.best_estimator_


def run_training_pipeline() -> Tuple[Pipeline, pd.DataFrame, dict]:
    """
    Execute the full training pipeline:
        1. Load and prepare data
        2. Engineer features
        3. Train and compare models
        4. Tune the best model
        5. Save the final model

    Returns:
        Tuple of (best_pipeline, comparison_df, metadata).
    """
    # Step 1: Prepare data
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)
    df = prepare_data()

    # Step 2: Feature engineering
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 60)
    X, y, preprocessor = prepare_features(df)

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 4: Train and compare
    logger.info("=" * 60)
    logger.info("STEP 3: Model Training & Comparison")
    logger.info("=" * 60)
    comparison_df = train_and_compare(X_train, X_test, y_train, y_test, preprocessor)
    print("\n📊 Model Comparison Results:")
    print(comparison_df.to_string(index=False))

    # Step 5: Identify and tune the best model
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info("=" * 60)
    logger.info(f"STEP 4: Hyperparameter Tuning ({best_model_name})")
    logger.info("=" * 60)

    best_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", get_models()[best_model_name])]
    )
    best_pipeline.fit(X_train, y_train)

    # Evaluate tuned model
    y_pred_tuned = best_pipeline.predict(X_test)
    tuned_metrics = evaluate_model(y_test, y_pred_tuned)
    logger.info(f"Tuned model metrics: {tuned_metrics}")

    # Step 6: Train Recommender Model
    logger.info("=" * 60)
    logger.info("STEP 5: Train Recommendation Engine")
    logger.info("=" * 60)

    # We use the preprocessor to transform all data for the nearest neighbors
    X_transformed = preprocessor.transform(X)
    recommender = NearestNeighbors(
        n_neighbors=4, metric="minkowski"
    )  # 4 because 1 is the query itself
    recommender.fit(X_transformed)

    # Save the recommender
    # save_model(recommender, RECOMMENDER_PATH)

    # Step 7: Save the model
    logger.info("=" * 60)
    logger.info("STEP 6: Saving Model")
    logger.info("=" * 60)
    save_model(best_pipeline, PIPELINE_PATH)

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "tuned_metrics": tuned_metrics,
        "best_params": {
            k: str(v)
            for k, v in best_pipeline.named_steps["regressor"].get_params().items()
        },
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "features": {
            "categorical": [
                c
                for c in ["brand", "model", "transmission", "fuelType"]
                if c in X.columns
            ],
            "numerical": [
                c
                for c in ["year", "car_age", "mileage", "mpg", "engineSize"]
                if c in X.columns
            ],
        },
        "comparison": comparison_df.to_dict(orient="records"),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {METADATA_PATH}")

    return best_pipeline, comparison_df, metadata


if __name__ == "__main__":
    pipeline, comparison, meta = run_training_pipeline()
    print("\n✅ Training complete!")
    print(f"Best model: {meta['best_model']}")
    print(f"R² Score: {meta['tuned_metrics']['R2_Score']}")


# ---------------------------------------------------------------------------
# EDA Visualizations
# ---------------------------------------------------------------------------
def plot_price_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of car prices in INR (Lakhs)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    price_lakhs = df["price_inr"] / 1_00_000
    ax.hist(price_lakhs, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Price (₹ Lakhs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Car Prices (INR)")
    ax.axvline(
        price_lakhs.median(),
        color="#e74c3c",
        linestyle="--",
        label=f"Median: ₹{price_lakhs.median():.1f}L",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "price_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: price_distribution.png")


def plot_brand_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of cars across brands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    brand_counts = df["brand"].value_counts()
    colors = sns.color_palette("husl", len(brand_counts))
    brand_counts.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Cars")
    ax.set_ylabel("Brand")
    ax.set_title("Car Distribution by Brand")
    # Add count labels
    for i, (val, name) in enumerate(zip(brand_counts.values, brand_counts.index)):
        ax.text(val + 50, i, f"{val:,}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "brand_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: brand_distribution.png")


def plot_fuel_type_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of fuel types."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fuel_counts = df["fuelType"].value_counts()
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    ax.pie(
        fuel_counts.values,
        labels=fuel_counts.index,
        autopct="%1.1f%%",
        colors=colors[: len(fuel_counts)],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("Fuel Type Distribution")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "fueltype_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: fueltype_distribution.png")


def plot_transmission_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of transmission types."""
    fig, ax = plt.subplots(figsize=(8, 6))
    trans_counts = df["transmission"].value_counts()
    colors = ["#1abc9c", "#e67e22", "#8e44ad"]
    trans_counts.plot(
        kind="bar",
        ax=ax,
        color=colors[: len(trans_counts)],
        edgecolor="white",
    )
    ax.set_xlabel("Transmission Type")
    ax.set_ylabel("Count")
    ax.set_title("Transmission Type Distribution")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "transmission_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: transmission_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numerical features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: correlation_heatmap.png")


def plot_mileage_vs_price(df: pd.DataFrame) -> None:
    """Scatter plot of mileage vs price, colored by brand."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for brand in df["brand"].unique():
        subset = df[df["brand"] == brand]
        ax.scatter(
            subset["mileage"],
            subset["price_inr"] / 1_00_000,
            alpha=0.4,
            s=15,
            label=brand,
        )
    ax.set_xlabel("Mileage (miles)")
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.set_title("Mileage vs Price")
    ax.legend(title="Brand", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "mileage_vs_price.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: mileage_vs_price.png")


def plot_engine_vs_price(df: pd.DataFrame) -> None:
    """Box plot of engine size vs price."""
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df.copy()
    df_plot["price_lakhs"] = df_plot["price_inr"] / 1_00_000
    # Bin engine sizes for readability
    engine_bins = sorted(df_plot["engineSize"].unique())
    if len(engine_bins) > 15:
        df_plot["engine_bin"] = pd.cut(df_plot["engineSize"], bins=10)
        sns.boxplot(
            x="engine_bin", y="price_lakhs", data=df_plot, ax=ax, palette="viridis"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    else:
        sns.boxplot(
            x="engineSize", y="price_lakhs", data=df_plot, ax=ax, palette="viridis"
        )
    ax.set_xlabel("Engine Size (L)")
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.set_title("Engine Size vs Price")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "engine_vs_price.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: engine_vs_price.png")


def plot_car_age_vs_price(df: pd.DataFrame) -> None:
    """Scatter plot of car age vs price."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if "car_age" not in df.columns:
        from src.utils import CURRENT_YEAR

        df = df.copy()
        df["car_age"] = CURRENT_YEAR - df["year"]

    scatter = ax.scatter(
        df["car_age"],
        df["price_inr"] / 1_00_000,
        c=df["car_age"],
        cmap="YlOrRd",
        alpha=0.4,
        s=10,
    )
    plt.colorbar(scatter, label="Car Age (years)")
    ax.set_xlabel("Car Age (years)")
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.set_title("Car Age vs Price")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "car_age_vs_price.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: car_age_vs_price.png")


def generate_eda_plots(df: pd.DataFrame) -> None:
    """Generate all EDA visualizations."""
    logger.info("Generating EDA plots...")
    plot_price_distribution(df)
    plot_brand_distribution(df)
    plot_fuel_type_distribution(df)
    plot_transmission_distribution(df)
    plot_correlation_heatmap(df)
    plot_mileage_vs_price(df)
    plot_engine_vs_price(df)
    plot_car_age_vs_price(df)
    logger.info("All EDA plots saved to images/")


# ---------------------------------------------------------------------------
# Model Evaluation Visualizations
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Scatter plot of actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(10, 8))
    y_true_lakhs = np.array(y_true) / 1_00_000
    y_pred_lakhs = np.array(y_pred) / 1_00_000

    ax.scatter(y_true_lakhs, y_pred_lakhs, alpha=0.3, s=10, color="#3498db")

    # Perfect prediction line
    min_val = min(y_true_lakhs.min(), y_pred_lakhs.min())
    max_val = max(y_true_lakhs.max(), y_pred_lakhs.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax.set_xlabel("Actual Price (₹ Lakhs)")
    ax.set_ylabel("Predicted Price (₹ Lakhs)")
    ax.set_title("Actual vs Predicted Car Prices")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: actual_vs_predicted.png")


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot the distribution of prediction residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    residuals = (np.array(y_true) - np.array(y_pred)) / 1_00_000

    # Histogram
    axes[0].hist(residuals, bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Residual (₹ Lakhs)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution")

    # Residual vs Predicted
    y_pred_lakhs = np.array(y_pred) / 1_00_000
    axes[1].scatter(y_pred_lakhs, residuals, alpha=0.3, s=10, color="#e74c3c")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted Price (₹ Lakhs)")
    axes[1].set_ylabel("Residual (₹ Lakhs)")
    axes[1].set_title("Residuals vs Predicted Values")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "residual_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: residual_distribution.png")


def plot_feature_importance(pipeline: Pipeline, feature_names: list) -> None:
    """
    Plot feature importance from a tree-based model within a pipeline.

    Args:
        pipeline: Trained sklearn Pipeline.
        feature_names: List of original feature names.
    """
    regressor = pipeline.named_steps.get("regressor")
    if not hasattr(regressor, "feature_importances_"):
        logger.warning("Model does not support feature_importances_. Skipping.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    # Get transformed feature names
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except AttributeError:
        transformed_names = [
            f"feature_{i}" for i in range(len(regressor.feature_importances_))
        ]

    importances = regressor.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color=sns.color_palette("viridis", len(indices)),
        edgecolor="white",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([transformed_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: feature_importance.png")


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------
def generate_shap_plots(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
) -> None:
    """
    Generate SHAP explainability plots.

    Args:
        pipeline: Trained sklearn Pipeline.
        X_sample: Sample of original feature data for explanation.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed. Skipping explainability plots.")
        return

    logger.info("Generating SHAP plots (this may take a moment)...")

    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Transform the sample data
    X_transformed = preprocessor.transform(X_sample)

    # Get feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Use TreeExplainer for tree models, otherwise KernelExplainer
    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        try:
            # Subsample for KernelExplainer (slow on large datasets)
            background = shap.sample(X_transformed, min(50, len(X_transformed)))
            explainer = shap.KernelExplainer(regressor.predict, background)
            shap_values = explainer.shap_values(
                X_transformed[: min(100, len(X_transformed))]
            )
            X_transformed = X_transformed[: min(100, len(X_transformed))]
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return

    # 1. SHAP Feature Importance (bar plot)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_feature_importance.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_feature_importance.png")

    # 2. SHAP Summary Plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_summary.png")

    # 3. SHAP Waterfall Plot (single prediction)
    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=(
                explainer.expected_value
                if isinstance(explainer.expected_value, float)
                else explainer.expected_value[0]
            ),
            data=X_transformed[0] if hasattr(X_transformed, "__getitem__") else None,
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title("SHAP Waterfall Plot (Single Prediction)")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "shap_waterfall.png", bbox_inches="tight")
        plt.close()
        logger.info("Saved: shap_waterfall.png")
    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")


# ---------------------------------------------------------------------------
# Full Evaluation Pipeline
# ---------------------------------------------------------------------------
def run_evaluation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
) -> None:
    """
    Run the complete evaluation pipeline:
        1. Generate EDA plots
        2. Generate model evaluation plots
        3. Generate SHAP plots

    Args:
        pipeline: Trained sklearn Pipeline.
        X: Full feature matrix.
        y: Full target vector.
        df: Full DataFrame (for EDA plots).
    """
    # EDA plots
    generate_eda_plots(df)

    # Train-test split for evaluation plots
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_pred = pipeline.predict(X_test)

    # Model evaluation plots
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residual_distribution(y_test, y_pred)

    feature_cols = list(X.columns)
    plot_feature_importance(pipeline, feature_cols)

    # SHAP plots (use a sample to speed things up)
    sample_size = min(500, len(X_test))
    generate_shap_plots(pipeline, X_test.iloc[:sample_size])

    logger.info("✅ All evaluation plots generated successfully!")


if __name__ == "__main__":

    # Load data
    df = prepare_data()

    df = create_features(df)
    X, y, _ = prepare_features(df)

    # Load trained model
    pipeline = load_model(PIPELINE_PATH)

    # Run evaluation
    run_evaluation(pipeline, X, y, df)


def create_input_dataframe(
    brand: str,
    model: str,
    year: int,
    transmission: str,
    mileage: int,
    fuel_type: str,
    mpg: float,
    engine_size: float,
) -> pd.DataFrame:
    """
    Create a properly formatted DataFrame from user inputs for prediction.

    Args:
        brand: Car brand (e.g., 'BMW', 'Audi').
        model: Car model (e.g., 'X5', 'A3').
        year: Manufacturing year.
        transmission: Transmission type ('Manual', 'Automatic', 'Semi-Auto').
        mileage: Odometer reading in miles.
        fuel_type: Fuel type ('Petrol', 'Diesel', 'Hybrid', 'Electric').
        mpg: Miles per gallon.
        engine_size: Engine size in litres.

    Returns:
        DataFrame with one row formatted for the prediction pipeline.
    """
    car_age = CURRENT_YEAR - year
    km_driven = mileage * 1.60934
    km_per_year = km_driven / max(car_age, 0.5)
    premium_brand_flag = 1 if brand in PREMIUM_BRANDS else 0

    input_data = pd.DataFrame(
        {
            "brand": [brand.strip()],
            "model": [model.strip()],
            "year": [year],
            "car_age": [max(0, car_age)],
            "transmission": [transmission.strip()],
            "mileage": [mileage],
            "fuelType": [fuel_type.strip()],
            "mpg": [mpg],
            "engineSize": [engine_size],
            "km_per_year": [km_per_year],
            "premium_brand_flag": [premium_brand_flag],
        }
    )

    return input_data


def predict_price(
    brand: str,
    model: str,
    year: int,
    transmission: str,
    mileage: int,
    fuel_type: str,
    mpg: float,
    engine_size: float,
    pipeline=None,
    asking_price: float = None,
) -> Dict[str, Any]:
    """
    Make a price prediction for a used car using the production Model Registry.
    """
    if pipeline is None:
        pipeline = load_model(PIPELINE_PATH)

    # Load market statistics to initialize Feature Engineer
    import joblib
    from src.utils import MODELS_DIR

    stats_path = MODELS_DIR / "production" / "market_stats.pkl"
    try:
        stats = joblib.load(stats_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing market_stats.pkl at {stats_path}. Run Experiment Manager first."
        )

    from src.feature_engineering import MarketFeatureEngineer

    engineer = MarketFeatureEngineer(stats)

    # 1. Create base input dataframe
    input_df = create_input_dataframe(
        brand, model, year, transmission, mileage, fuel_type, mpg, engine_size
    )

    # 2. Engineer full market features
    input_features_df = engineer.engineer_features(input_df)

    # 3. Instantiate Engines
    from src.explanation_engine import ShapExplanationProvider, ExplanationEngine
    from src.valuation_intelligence import ValuationIntelligenceEngine
    from src.decision_intelligence import DecisionIntelligenceEngine
    from src.knowledge_engine import VehicleKnowledgeEngine

    # We load the config and engines
    shap_provider = ShapExplanationProvider()

    import json

    with open("src/valuation_config.json", "r") as f:
        config = json.load(f)

    explanation_engine = ExplanationEngine(shap_provider, config)
    knowledge_engine = VehicleKnowledgeEngine()

    valuation_engine = ValuationIntelligenceEngine(
        config_path="src/valuation_config.json",
        explanation_engine=explanation_engine,
        knowledge_engine=knowledge_engine,
    )

    decision_engine = DecisionIntelligenceEngine(knowledge_engine=knowledge_engine)

    # 4. Generate Comprehensive Report
    report = valuation_engine.generate_valuation_report(
        model_pipeline=pipeline, input_data=input_df, input_features=input_features_df
    )

    # 5. Original Price Simulation (kept for backward compatibility of UI)
    car_age = CURRENT_YEAR - year
    depreciation_rate = stats.get_brand_annual_depreciation_rate(brand) * car_age
    depreciation_rate = max(0.1, min(depreciation_rate, 0.75))
    original_price = report["estimated_market_value_raw"] / (1 - depreciation_rate)

    input_summary = {
        "brand": brand,
        "model": model,
        "year": year,
        "car_age": car_age,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "mpg": mpg,
        "engineSize": engine_size,
    }

    # 6. Generate Decision Report
    decision_report = decision_engine.generate_decision_report(
        valuation_report=report,
        input_summary=input_summary,
        asking_price=asking_price,
        current_recommendations=[],
    )

    # Reconstruct result maintaining backward compatibility for the UI
    result = {
        "predicted_price": report["estimated_market_value"],
        "predicted_price_raw": round(report["estimated_market_value_raw"], 2),
        "price_range": f"{report['estimated_market_range']['lower_bound']} - {report['estimated_market_range']['upper_bound']}",
        "original_price": format_price_inr(original_price),
        "depreciation_percent": f"{depreciation_rate * 100:.0f}%",
        "confidence": f"{report['confidence']['score']:.0f}%",
        "recommendations": decision_report.get(
            "alternatives", []
        ),  # Use new alternatives
        # New Valuation & Decision Intelligence fields
        "valuation_report": report,
        "decision_report": decision_report,
        "input_summary": input_summary,
    }

    logger.info(
        f"Valuation Generated: {result['predicted_price']} for {brand} {model} ({year})"
    )
    return result


def batch_predict(input_data: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    """
    Make predictions for multiple cars at once.

    Args:
        input_data: DataFrame with columns matching the training features.
        pipeline: Pre-loaded pipeline (optional).

    Returns:
        Input DataFrame with added 'predicted_price_inr' and
        'predicted_price_formatted' columns.
    """
    if pipeline is None:
        pipeline = load_model(PIPELINE_PATH)

    # Load market statistics to initialize Feature Engineer
    import joblib
    from src.utils import MODELS_DIR

    stats_path = MODELS_DIR / "production" / "market_stats.pkl"
    try:
        stats = joblib.load(stats_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing market_stats.pkl at {stats_path}. Run Experiment Manager first."
        )

    from src.feature_engineering import MarketFeatureEngineer

    engineer = MarketFeatureEngineer(stats)

    input_data = input_data.copy()

    # Engineer full market features
    input_features_df = engineer.engineer_features(input_data)

    predictions = pipeline.predict(input_features_df)

    input_data["predicted_price_inr"] = np.maximum(0, predictions)
    input_data["predicted_price_formatted"] = input_data["predicted_price_inr"].apply(
        format_price_inr
    )

    return input_data
