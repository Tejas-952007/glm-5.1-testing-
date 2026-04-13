"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL
from src.preprocessing import (
    build_preprocessor,
    engineer_features,
    handle_missing_values,
    load_data,
    preprocess,
)


@pytest.fixture(scope="module")
def sample_df():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Age": np.random.randint(18, 75, n),
        "Income": np.random.normal(45000, 10000, n).clip(8000),
        "Credit_Score": np.random.randint(300, 850, n),
        "Employment_Status": np.random.choice(["Employed", "Self-Employed", "Unemployed", "Student"], n),
        "Existing_Loans": np.random.poisson(1.5, n).clip(0, 8),
        "Loan_Amount": np.random.normal(12000, 5000, n).clip(500),
        "Approved": np.random.randint(0, 2, n),
    })
    # Inject some NaNs
    df.loc[df.sample(frac=0.05).index, "Income"] = np.nan
    df.loc[df.sample(frac=0.03).index, "Employment_Status"] = np.nan
    return df


class TestHandleMissingValues:
    def test_fills_all_nans(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result.isnull().sum().sum() == 0, "Missing values remain after imputation"

    def test_preserves_shape(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result.shape == sample_df.shape, "Shape changed after imputation"


class TestEngineerFeatures:
    def test_adds_new_columns(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        result = engineer_features(df_clean)
        expected_cols = ["Income_to_Loan_Ratio", "Credit_per_Loan", "Age_Credit_Interaction", "Has_Existing_Loans"]
        for col in expected_cols:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_has_existing_loans_is_binary(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        result = engineer_features(df_clean)
        assert set(result["Has_Existing_Loans"].unique()).issubset({0, 1}), "Has_Existing_Loans is not binary"


class TestBuildPreprocessor:
    def test_preprocessor_is_pipeline_ready(self):
        preprocessor = build_preprocessor()
        assert preprocessor is not None, "Preprocessor is None"
        assert len(preprocessor.transformers) == 2, "Expected 2 transformers (num + cat)"


class TestPreprocessEndToEnd:
    def test_preprocess_returns_expected_shapes(self):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess()
        assert X_train.shape[0] > 0, "X_train is empty"
        assert X_test.shape[0] > 0, "X_test is empty"
        assert len(feature_names) > 0, "No feature names returned"
        assert y_train.shape[0] == X_train.shape[0], "X_train / y_train row mismatch"
        assert y_test.shape[0] == X_test.shape[0], "X_test / y_test row mismatch"

    def test_preprocessor_transforms_correctly(self):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess()
        X_transformed = preprocessor.transform(X_train)
        assert X_transformed.shape[0] == X_train.shape[0], "Row count mismatch after transform"
        assert X_transformed.shape[1] == len(feature_names), "Column count mismatch after transform"