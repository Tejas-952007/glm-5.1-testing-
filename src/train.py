"""
Training pipeline — ties preprocessing, model training, and evaluation together.

Usage:
    python -m src.train
"""

import os
import pickle

# Allow running as `python src/train.py` from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess
from model import train_model, save_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance, plot_roc_curve

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "credit_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Preprocess
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = preprocess(DATA_PATH)

    # 2. Train
    best_model, grid = train_model(X_train, y_train)

    # 3. Evaluate
    metrics, y_pred = evaluate_model(best_model, X_test, y_test)

    # 4. Plots
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_feature_importance(best_model, feature_names, save_path=os.path.join(PLOTS_DIR, "feature_importance.png"))
    plot_roc_curve(best_model, X_test, y_test, save_path=os.path.join(PLOTS_DIR, "roc_curve.png"))

    # 5. Save model + artifacts
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    save_model(best_model, model_path)

    artifacts = {
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
    }
    artifacts_path = os.path.join(MODEL_DIR, "artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Artifacts saved to {artifacts_path}")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()