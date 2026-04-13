"""
Centralized configuration for the project.

All hardcoded constants, paths, column definitions, and
hyperparameter grids live here so they can be changed in one place.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

DATA_PATH = DATA_DIR / "credit_data.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.joblib"
COMPARISON_PATH = MODEL_DIR / "comparison_results.csv"

# ── Column definitions ────────────────────────────────────────────────────
TARGET_COL = "Approved"
NUMERIC_COLS = [
    "Age",
    "Income",
    "Credit_Score",
    "Existing_Loans",
    "Loan_Amount",
    "Income_to_Loan_Ratio",
    "Credit_per_Loan",
    "Age_Credit_Interaction",
    "Has_Existing_Loans",
]
CATEGORICAL_COLS = ["Employment_Status"]

# ── Train / test split ───────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Classifiers & param grids ────────────────────────────────────────────
CLASSIFIERS = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
        "clf__max_iter": [1000],
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_split": [2, 5],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5],
    },
    "LightGBM": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__num_leaves": [15, 31],
    },
}

# ── GridSearchCV settings ────────────────────────────────────────────────
CV_FOLDS = 5
SCORING_METRIC = "f1"

# ── MLflow ────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "credit_card_approval"
MLFLOW_TRACKING_URI = f"file://{MODEL_DIR / 'mlruns'}"