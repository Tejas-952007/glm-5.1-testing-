"""
Model training module.

Supports multi-model comparison using GridSearchCV.
Uses joblib for serialization (scikit-learn recommended).
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import CLASSIFIERS, CV_FOLDS, SCORING_METRIC
from src.logger import get_logger

logger = get_logger(__name__)

# ── Estimator registry ────────────────────────────────────────────────────
ESTIMATORS = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": None,   # imported lazily to avoid hard dependency
    "LightGBM": None,  # imported lazily to avoid hard dependency
}


def _get_estimator(name: str):
    """Return an estimator instance, lazily importing optional deps."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=42, verbose=-1)
    return ESTIMATORS[name]


def train_model(name: str, preprocessor, X_train, y_train):
    """Train a single model using GridSearchCV wrapped in a Pipeline.

    Parameters
    ----------
    name : str — key in CLASSIFIERS config
    preprocessor : fitted ColumnTransformer
    X_train : DataFrame
    y_train : Series

    Returns
    -------
    best_model : fitted Pipeline
    grid : fitted GridSearchCV
    """
    estimator = _get_estimator(name)
    param_grid = CLASSIFIERS[name]

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])

    logger.info("Training %s with GridSearchCV (%d-fold CV, scoring=%s)...",
                name, CV_FOLDS, SCORING_METRIC)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info("  Best params: %s", grid.best_params_)
    logger.info("  Best CV %s: %.4f", SCORING_METRIC, grid.best_score_)

    return grid.best_estimator_, grid


def train_all_models(preprocessor, X_train, y_train) -> dict:
    """Train all configured models and return results.

    Returns
    -------
    results : dict mapping model name to {"pipeline": Pipeline, "grid": GridSearchCV}
    """
    results = {}
    for name in CLASSIFIERS:
        try:
            pipeline, grid = train_model(name, preprocessor, X_train, y_train)
            results[name] = {"pipeline": pipeline, "grid": grid}
        except ImportError as e:
            logger.warning("Skipping %s — dependency not installed: %s", name, e)
    return results


def save_model(model, filepath: str) -> None:
    """Serialize the trained model (Pipeline) to disk using joblib."""
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)


def load_model(filepath: str):
    """Load a serialized model from disk."""
    model = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return model