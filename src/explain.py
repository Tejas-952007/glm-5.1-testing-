"""
SHAP-based model interpretability module.

Generates summary (beeswarm) plots and per-prediction
waterfall explanations. Automatically selects the correct
SHAP explainer based on model type.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap

from src.logger import get_logger

logger = get_logger(__name__)


def _get_explainer(model, X_train, X_test):
    """Select the appropriate SHAP explainer for the model type."""
    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]

    # Transform data through the pipeline's preprocessor
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if isinstance(clf, shap.models._tree.TreeExplainer):
        return shap.TreeExplainer(clf, X_train_transformed), X_test_transformed

    # Tree-based models
    if hasattr(clf, "apply"):  # RandomForest, XGBoost, LightGBM
        explainer = shap.TreeExplainer(clf, X_train_transformed)
        return explainer, X_test_transformed

    # Linear models (LogisticRegression)
    explainer = shap.LinearExplainer(clf, X_train_transformed)
    return explainer, X_test_transformed


def _get_feature_names(model) -> list[str]:
    """Extract feature names from the pipeline's preprocessor."""
    preprocessor = model.named_steps["preprocessor"]
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            feature_names.extend(transformer.get_feature_names_out(cols).tolist())
    return feature_names


def generate_shap_summary(model, X_train, X_test, save_dir: str = "."):
    """Generate SHAP beeswarm and bar summary plots.

    Parameters
    ----------
    model : fitted Pipeline
    X_train, X_test : DataFrames
    save_dir : directory to save plots
    """
    logger.info("Generating SHAP summary plots...")

    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = _get_feature_names(model)

    # Select explainer
    if hasattr(clf, "apply"):
        explainer = shap.TreeExplainer(clf, X_train_transformed)
    else:
        explainer = shap.LinearExplainer(clf, X_train_transformed)

    shap_values = explainer.shap_values(X_test_transformed)

    # For binary classification, TreeExplainer returns a list; take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    beeswarm_path = f"{save_dir}/shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm plot saved to %s", beeswarm_path)

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = f"{save_dir}/shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar plot saved to %s", bar_path)

    return shap_values


def generate_shap_waterfall(model, X_train, input_df, save_path: str = "shap_waterfall.png"):
    """Generate a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    model : fitted Pipeline
    X_train : training DataFrame (for background)
    input_df : single-row DataFrame
    save_path : path to save the waterfall plot
    """
    logger.info("Generating SHAP waterfall plot for single prediction...")

    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]

    X_train_transformed = preprocessor.transform(X_train)
    input_transformed = preprocessor.transform(input_df)
    feature_names = _get_feature_names(model)

    # Select explainer
    if hasattr(clf, "apply"):
        explainer = shap.TreeExplainer(clf, X_train_transformed)
    else:
        explainer = shap.LinearExplainer(clf, X_train_transformed)

    shap_values = explainer.shap_values(input_transformed)

    # For binary classification, take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Waterfall plot for the single instance
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                       else explainer.expected_value[1],
        data=input_transformed[0],
        feature_names=feature_names,
    )

    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP waterfall plot saved to %s", save_path)