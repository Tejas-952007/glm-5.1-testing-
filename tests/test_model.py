"""Tests for the model training module."""

import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocess
from src.model import train_model, save_model, load_model


@pytest.fixture(scope="module")
def training_data():
    """Get preprocessed data for model tests."""
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess()
    return X_train, X_test, y_train, y_test, preprocessor, feature_names


class TestTrainModel:
    def test_logistic_regression_trains(self, training_data):
        X_train, _, y_train, _, preprocessor, _ = training_data
        pipeline, grid = train_model("LogisticRegression", preprocessor, X_train, y_train)
        assert pipeline is not None, "Pipeline is None"
        assert hasattr(pipeline, "predict"), "Pipeline missing predict method"
        assert hasattr(pipeline, "predict_proba"), "Pipeline missing predict_proba method"

    def test_random_forest_trains(self, training_data):
        X_train, _, y_train, _, preprocessor, _ = training_data
        pipeline, grid = train_model("RandomForest", preprocessor, X_train, y_train)
        assert pipeline is not None, "Pipeline is None"

    def test_pipeline_has_expected_steps(self, training_data):
        X_train, _, y_train, _, preprocessor, _ = training_data
        pipeline, _ = train_model("LogisticRegression", preprocessor, X_train, y_train)
        assert "preprocessor" in pipeline.named_steps, "Missing preprocessor step"
        assert "clf" in pipeline.named_steps, "Missing clf step"


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, training_data, tmp_path):
        X_train, _, y_train, _, preprocessor, _ = training_data
        pipeline, _ = train_model("LogisticRegression", preprocessor, X_train, y_train)

        model_path = str(tmp_path / "test_model.joblib")
        save_model(pipeline, model_path)
        loaded = load_model(model_path)

        assert hasattr(loaded, "predict"), "Loaded model missing predict"
        # Verify predictions match
        import numpy as np
        sample = X_train.iloc[:5]
        np.testing.assert_array_equal(pipeline.predict(sample), loaded.predict(sample))