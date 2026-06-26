# Model Experimentation Framework

DRIVEIQ employs a professional Machine Learning Experimentation framework orchestrated by the `ExperimentManager`. The framework evaluates multiple gradient-boosted and tree-based algorithms across different engineered feature subsets to discover the most performant model for production.

## Core Capabilities

1. **Automated Hyperparameter Tuning**: Bayesian optimization via `Optuna` automatically searches parameter spaces efficiently.
2. **Robust Validation**: `K-Fold` cross-validation evaluating R², MAE, RMSE, MAPE, and Median Absolute Error.
3. **Data Quality Validation**: Pre-training checks for target leakage, NaN values, and heavily correlated features (>0.95 Pearson).
4. **Model Registry**: Models are automatically versioned and partitioned into `candidates`, `production`, and `archive`.
5. **Leaderboard Generation**: Every run generates a dynamic `leaderboard.md` and `.csv` ranking combinations of `Feature Subsets` and `Models`.

## Configuration (`config.json`)

Experiments are controlled entirely through `src/experimentation/config.json`.

* **`optimization`**: Tune the number of `n_trials` (search depth) and `timeout_seconds` per model.
* **`feature_subsets`**: A dictionary where keys are subset names (e.g. `baseline`, `full_engineered`) and values are arrays of features. The system trains every model against every subset.
* **`models_to_evaluate`**: Algorithms to test (e.g., `XGBoost`, `LightGBM`, `RandomForest`).

## Output Directory (`models/`)

The output of the framework is written to the `models/` folder:

```
models/
├── production/
│   ├── pipeline.pkl         # Auto-deployed production model
│   ├── market_stats.pkl     # Pre-computed market stats logic
│   └── metadata.json        # Manifest
├── candidates/
│   └── exp_YYYYMMDD_HHMM/
│       ├── leaderboard.csv
│       ├── XGBoost_baseline.pkl
│       └── ...
└── archive/
```

## Running an Experiment

To trigger a full training run evaluating the search space:

```bash
PYTHONPATH=. python3 src/experimentation/experiment_manager.py
```

*Note: A comprehensive run across all 7 models and subsets may take 1+ hours depending on `n_trials`.*
