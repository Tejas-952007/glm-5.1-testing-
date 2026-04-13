"""
Model evaluation and visualization module.

Computes classification metrics and generates plots.
Works with any fitted Pipeline that exposes predict / predict_proba.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test) -> tuple[dict, np.ndarray]:
    """Compute standard classification metrics.

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1
    y_pred : ndarray
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    logger.info("=" * 50)
    logger.info("EVALUATION")
    logger.info("=" * 50)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall   : %.4f", metrics["recall"])
    logger.info("F1 Score : %.4f", metrics["f1"])

    return metrics, y_pred


def compare_models(results: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all trained models on the test set and return a comparison DataFrame.

    Parameters
    ----------
    results : dict from train_all_models
    X_test, y_test : test data

    Returns
    -------
    comparison_df : DataFrame sorted by F1 descending
    """
    rows = []
    for name, res in results.items():
        model = res["pipeline"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": auc(*roc_curve(y_test, y_proba)[:2]),
        })

    comparison_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    logger.info("\nModel Comparison:\n%s", comparison_df.to_string(index=False))
    return comparison_df


def plot_confusion_matrix(y_test, y_pred, save_path: str = "confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Rejected", "Approved"],
           yticklabels=["Rejected", "Approved"],
           title="Confusion Matrix",
           ylabel="Actual",
           xlabel="Predicted")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_feature_importance(model, feature_names, save_path: str = "feature_importance.png"):
    """Plot feature importance — coefficients for linear models, feature_importances_ for tree models."""
    clf = model.named_steps["clf"]

    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):
        coefs = clf.feature_importances_
    else:
        logger.warning("Classifier %s has no coef_ or feature_importances_; skipping plot.", type(clf).__name__)
        return

    sorted_idx = np.argsort(coefs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs[sorted_idx]]
    ax.barh(np.array(feature_names)[sorted_idx], coefs[sorted_idx], color=colors)
    ax.set_title(f"Feature Importance ({type(clf).__name__})")
    ax.set_xlabel("Importance / Coefficient Value")
    ax.axvline(x=0, color="grey", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path)


def plot_roc_curve(model, X_test, y_test, save_path: str = "roc_curve.png"):
    """Plot and save the ROC curve with AUC."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to %s", save_path)


def plot_precision_recall_curve(model, X_test, y_test, save_path: str = "precision_recall_curve.png"):
    """Plot and save the Precision-Recall curve."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="#e74c3c", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Precision-Recall curve saved to %s", save_path)