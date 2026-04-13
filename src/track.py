"""
MLflow experiment tracking module.

Logs parameters, metrics, models, and comparison results
for each training run.
"""

import os

import pandas as pd

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow():
    """Configure MLflow tracking URI and experiment."""
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
        logger.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)
        return True
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking. "
                       "Install with: pip install mlflow")
        return False


def log_model_run(model_name: str, params: dict, metrics: dict,
                  model, X_test_path: str = None):
    """Log a single model run to MLflow.

    Parameters
    ----------
    model_name : str
    params : dict of hyperparameters
    metrics : dict of evaluation metrics
    model : fitted sklearn Pipeline
    X_test_path : optional path to test data artifact
    """
    try:
        import mlflow
        import mlflow.sklearn

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, model_name)

            if X_test_path:
                mlflow.log_artifact(X_test_path)

            logger.info("Logged MLflow run: %s", model_name)
    except ImportError:
        logger.warning("mlflow not installed — skipping run logging.")


def log_comparison_results(comparison_df: pd.DataFrame):
    """Save model comparison results to CSV and optionally log to MLflow."""
    results_path = str(MODEL_DIR / "comparison_results.csv")
    comparison_df.to_csv(results_path, index=False)
    logger.info("Comparison results saved to %s", results_path)

    try:
        import mlflow
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_artifact(results_path)
            logger.info("Comparison results logged to MLflow.")
    except ImportError:
        pass