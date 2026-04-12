"""
Model training module.

Uses Logistic Regression with GridSearchCV for hyperparameter tuning.
"""

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "max_iter": [1000],
}


def train_model(X_train, y_train, cv: int = 5, scoring: str = "f1"):
    """Run GridSearchCV over LogisticRegression and return the best estimator.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    cv : int, number of cross-validation folds
    scoring : str, metric to optimise

    Returns
    -------
    best_model : fitted LogisticRegression
    grid : fitted GridSearchCV object
    """
    print("=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)

    lr = LogisticRegression()
    grid = GridSearchCV(
        lr,
        PARAM_GRID,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV {scoring}: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    return best_model, grid


def save_model(model, filepath: str) -> None:
    """Pickle the trained model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a pickled model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model