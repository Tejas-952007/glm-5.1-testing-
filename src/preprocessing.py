"""
Data preprocessing module.

Handles missing values, feature engineering, and builds a scikit-learn
Pipeline (ColumnTransformer + classifier) that bundles encoding and
scaling — preventing data leakage and simplifying deployment.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load CSV and return a DataFrame."""
    if filepath is None:
        filepath = str(DATA_PATH)
    df = pd.read_csv(filepath)
    logger.info("Loaded data: %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values — median for numeric, mode for categorical."""
    df = df.copy()

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled %s missing values with median=%.2f", col, median_val)

    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Filled %s missing values with mode=%s", col, mode_val)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw columns.

    New features:
    - Income_to_Loan_Ratio: income relative to requested loan
    - Credit_per_Loan: credit quality per outstanding obligation
    - Age_Credit_Interaction: combined age-credit signal
    - Has_Existing_Loans: binary flag for any existing debt
    """
    df = df.copy()

    df["Income_to_Loan_Ratio"] = df["Income"] / (df["Loan_Amount"] + 1)
    df["Credit_per_Loan"] = df["Credit_Score"] / (df["Existing_Loans"] + 1)
    df["Age_Credit_Interaction"] = df["Age"] * df["Credit_Score"]
    df["Has_Existing_Loans"] = (df["Existing_Loans"] > 0).astype(int)

    logger.info("Engineered 4 new features: Income_to_Loan_Ratio, Credit_per_Loan, "
                "Age_Credit_Interaction, Has_Existing_Loans")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for numeric + categorical columns.

    - StandardScaler on numeric columns
    - OneHotEncoder (drop first) on categorical columns
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            feature_names.extend(transformer.get_feature_names_out(cols).tolist())
    return feature_names


def preprocess(filepath: str = None):
    """Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    preprocessor : fitted ColumnTransformer
    feature_names : list[str]
    """
    logger.info("=" * 50)
    logger.info("PREPROCESSING STARTED")
    logger.info("=" * 50)

    # 1. Load
    df = load_data(filepath)

    # 2. Missing values
    logger.info("Handling missing values...")
    df = handle_missing_values(df)

    # 3. Feature engineering
    logger.info("Engineering features...")
    df = engineer_features(df)

    # 4. Split features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train size: %d, Test size: %d", X_train.shape[0], X_test.shape[0])
    logger.info("Approval rate — train: %.2f%%, test: %.2f%%",
                y_train.mean() * 100, y_test.mean() * 100)

    # 6. Build and fit preprocessor (ColumnTransformer)
    logger.info("Building preprocessor pipeline...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    feature_names = get_feature_names(preprocessor)
    logger.info("Output features (%d): %s", len(feature_names), feature_names)

    logger.info("Preprocessing complete.\n")
    return X_train, X_test, y_train, y_test, preprocessor, feature_names