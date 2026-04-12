"""
Model evaluation and visualization module.

Computes classification metrics and generates plots.
"""

import numpy as np
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
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute standard classification metrics.

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print("=" * 50)
    print("EVALUATION")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    return metrics, y_pred


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

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path: str = "feature_importance.png"):
    """Plot logistic-regression coefficients as a bar chart."""
    coefs = model.coef_[0]
    sorted_idx = np.argsort(coefs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs[sorted_idx]]
    ax.barh(np.array(feature_names)[sorted_idx], coefs[sorted_idx], color=colors)
    ax.set_title("Feature Importance (Logistic Regression Coefficients)")
    ax.set_xlabel("Coefficient Value")
    ax.axvline(x=0, color="grey", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot saved to {save_path}")


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
    print(f"ROC curve saved to {save_path}")