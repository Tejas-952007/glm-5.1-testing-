"""Tests for the evaluation module."""

import pytest
import numpy as np

from src.preprocessing import preprocess
from src.model import train_model
from src.evaluate import evaluate_model, compare_models


@pytest.fixture(scope="module")
def trained_model():
    """Train a model and return data for evaluation tests."""
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess()
    pipeline, grid = train_model("LogisticRegression", preprocessor, X_train, y_train)
    return pipeline, X_test, y_test


class TestEvaluateModel:
    def test_returns_expected_metrics(self, trained_model):
        model, X_test, y_test = trained_model
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics, "Missing accuracy metric"
        assert "precision" in metrics, "Missing precision metric"
        assert "recall" in metrics, "Missing recall metric"
        assert "f1" in metrics, "Missing f1 metric"

    def test_metrics_are_valid(self, trained_model):
        model, X_test, y_test = trained_model
        metrics, _ = evaluate_model(model, X_test, y_test)

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= metrics[key] <= 1, f"{key} is out of [0, 1] range: {metrics[key]}"

    def test_predictions_shape_matches(self, trained_model):
        model, X_test, y_test = trained_model
        _, y_pred = evaluate_model(model, X_test, y_test)
        assert len(y_pred) == len(y_test), "Prediction count mismatch"


class TestCompareModels:
    def test_comparison_df_has_all_models(self):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess()

        # Train only the fast models for testing
        from src.model import train_model
        results = {}
        for name in ["LogisticRegression", "RandomForest"]:
            pipeline, grid = train_model(name, preprocessor, X_train, y_train)
            results[name] = {"pipeline": pipeline, "grid": grid}

        comparison_df = compare_models(results, X_test, y_test)

        assert "Model" in comparison_df.columns, "Missing Model column"
        assert "F1" in comparison_df.columns, "Missing F1 column"
        assert len(comparison_df) == 2, "Expected 2 models in comparison"
        assert comparison_df["F1"].max() > 0, "All F1 scores are zero"