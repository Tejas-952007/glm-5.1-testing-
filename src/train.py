"""
Training pipeline — ties preprocessing, multi-model training,
evaluation, and experiment tracking together.

Usage:
    python -m src.train
"""

import os
import sys

# Ensure project root is on the path for `src.*` imports
# Works for both `python src/train.py` and `python -m src.train`
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import joblib
import pandas as pd

from src.config import (
    COMPARISON_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PLOTS_DIR,
    CLASSIFIERS,
)
from src.preprocessing import preprocess, build_preprocessor
from src.model import train_all_models, save_model
from src.evaluate import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.explain import generate_shap_summary
from src.track import setup_mlflow, log_model_run, log_comparison_results
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Preprocess ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess()

    # ── 2. Train all models ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("MODEL TRAINING")
    logger.info("=" * 50)

    results = train_all_models(preprocessor, X_train, y_train)

    if not results:
        logger.error("No models trained successfully. Exiting.")
        return

    # ── 3. Compare models ────────────────────────────────────────────────
    comparison_df = compare_models(results, X_test, y_test)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    logger.info("Comparison results saved to %s", COMPARISON_PATH)

    # ── 4. Select champion model ──────────────────────────────────────────
    champion_name = comparison_df.iloc[0]["Model"]
    champion_pipeline = results[champion_name]["pipeline"]
    logger.info("\n🏆 Champion model: %s (F1=%.4f)", champion_name, comparison_df.iloc[0]["F1"])

    # ── 5. Evaluate champion ─────────────────────────────────────────────
    metrics, y_pred = evaluate_model(champion_pipeline, X_test, y_test)

    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_feature_importance(champion_pipeline, feature_names,
                           save_path=os.path.join(PLOTS_DIR, "feature_importance.png"))
    plot_roc_curve(champion_pipeline, X_test, y_test,
                   save_path=os.path.join(PLOTS_DIR, "roc_curve.png"))
    plot_precision_recall_curve(champion_pipeline, X_test, y_test,
                                save_path=os.path.join(PLOTS_DIR, "precision_recall_curve.png"))

    # ── 6. SHAP summary ──────────────────────────────────────────────────
    try:
        generate_shap_summary(champion_pipeline, X_train, X_test, save_dir=str(PLOTS_DIR))
    except Exception as e:
        logger.warning("SHAP summary failed: %s — skipping.", e)

    # ── 7. Save champion model + preprocessor ────────────────────────────
    save_model(champion_pipeline, str(MODEL_PATH))

    artifacts = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "champion_name": champion_name,
        "comparison_results": comparison_df.to_dict("records"),
    }
    artifacts_path = str(MODEL_DIR / "artifacts.joblib")
    joblib.dump(artifacts, artifacts_path)
    logger.info("Artifacts saved to %s", artifacts_path)

    # ── 8. MLflow tracking ───────────────────────────────────────────────
    mlflow_enabled = setup_mlflow()
    if mlflow_enabled:
        for name, res in results.items():
            grid = res["grid"]
            log_model_run(
                model_name=name,
                params=grid.best_params_,
                metrics={
                    "accuracy": comparison_df.loc[comparison_df["Model"] == name, "Accuracy"].values[0],
                    "precision": comparison_df.loc[comparison_df["Model"] == name, "Precision"].values[0],
                    "recall": comparison_df.loc[comparison_df["Model"] == name, "Recall"].values[0],
                    "f1": comparison_df.loc[comparison_df["Model"] == name, "F1"].values[0],
                    "roc_auc": comparison_df.loc[comparison_df["Model"] == name, "ROC_AUC"].values[0],
                },
                model=res["pipeline"],
            )
        log_comparison_results(comparison_df)

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE — Champion: %s", champion_name)
    logger.info("=" * 50)
    print(f"\n🏆 Champion: {champion_name}")
    print(f"📊 F1 Score: {metrics['f1']:.4f}")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()