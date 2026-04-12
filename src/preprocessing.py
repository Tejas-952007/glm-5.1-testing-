"""
Data preprocessing module.

Handles missing values, encodes categoricals, scales features,
and performs train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


CATEGORICAL_COLS = ["Employment_Status"]
NUMERIC_COLS = ["Age", "Income", "Credit_Score", "Existing_Loans", "Loan_Amount"]
TARGET_COL = "Approved"


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values — median for numeric, mode for categorical."""
    df = df.copy()

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {col} missing values with median={median_val:.2f}")

    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Filled {col} missing values with mode={mode_val}")

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns; return transformed df and encoders."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training data and transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def preprocess(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline — returns train/test splits + artifacts.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    scaler : StandardScaler
    encoders : dict of LabelEncoder
    feature_names : list[str]
    """
    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    # 1. Load
    df = load_data(filepath)

    # 2. Missing values
    print("\nHandling missing values...")
    df = handle_missing_values(df)

    # 3. Encode categoricals
    print("\nEncoding categorical variables...")
    df, encoders = encode_categoricals(df)

    # 4. Split features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Approval rate — train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")

    # 6. Scale features
    print("\nScaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    print("\nPreprocessing complete.\n")
    return X_train, X_test, y_train, y_test, scaler, encoders, feature_names