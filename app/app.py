"""
Credit Card Approval Prediction — Streamlit App (Multi-page dashboard)

Pages:
  1. Home — project overview & model metrics
  2. Single Prediction — form + SHAP waterfall
  3. Batch Prediction — CSV upload → predictions
  4. Model Comparison — bar charts & confusion matrices
"""

import io
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
ARTIFACTS_PATH = os.path.join(MODEL_DIR, "artifacts.joblib")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")


# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    return pipeline, artifacts


pipeline, artifacts = load_model_and_artifacts()
preprocessor = artifacts["preprocessor"]
feature_names = artifacts["feature_names"]
champion_name = artifacts.get("champion_name", "Unknown")
comparison_results = artifacts.get("comparison_results", [])


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Card Approval", page_icon="💳", layout="wide")


# ── Sidebar navigation ──────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Single Prediction", "Batch Prediction", "Model Comparison"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Champion Model:** `{champion_name}`")
if comparison_results:
    best = comparison_results[0]
    st.sidebar.metric("Best F1", f"{best['F1']:.4f}")
    st.sidebar.metric("Best Accuracy", f"{best['Accuracy']:.4f}")


# ── Helper: build input DataFrame ──────────────────────────────────────────
def build_input_df(age, income, credit_score, employment_status, existing_loans, loan_amount):
    """Build a single-row DataFrame matching the training schema."""
    return pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Credit_Score": credit_score,
        "Employment_Status": employment_status,
        "Existing_Loans": existing_loans,
        "Loan_Amount": loan_amount,
    }])


# ── Helper: engineer features on input ──────────────────────────────────────
def engineer_features_df(df):
    """Add engineered features to match the training pipeline."""
    df = df.copy()
    df["Income_to_Loan_Ratio"] = df["Income"] / (df["Loan_Amount"] + 1)
    df["Credit_per_Loan"] = df["Credit_Score"] / (df["Existing_Loans"] + 1)
    df["Age_Credit_Interaction"] = df["Age"] * df["Credit_Score"]
    df["Has_Existing_Loans"] = (df["Existing_Loans"] > 0).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Credit Card Approval Prediction")
    st.markdown(
        "An end-to-end ML project that predicts whether a credit card application "
        "will be **Approved** or **Rejected** using Logistic Regression, Random Forest, "
        "XGBoost, and LightGBM with hyperparameter tuning via GridSearchCV."
    )

    st.markdown("---")

    # ── Dataset stats ───────────────────────────────────────────────────────
    data_path = os.path.join(PROJECT_ROOT, "data", "credit_data.csv")
    if os.path.exists(data_path):
        df_stats = pd.read_csv(data_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df_stats):,}")
        col2.metric("Approval Rate", f"{df_stats['Approved'].mean():.1%}")
        col3.metric("Avg Credit Score", f"{df_stats['Credit_Score'].mean():.0f}")
        col4.metric("Avg Income", f"${df_stats['Income'].mean():,.0f}")

    st.markdown("---")

    # ── Model comparison table ──────────────────────────────────────────────
    if comparison_results:
        st.subheader("Model Comparison")
        comp_df = pd.DataFrame(comparison_results)
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC_AUC": "{:.4f}",
            }).highlight_max(subset=["F1"], color="#2ecc71"),
            use_container_width=True,
        )

    # ── Project features ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Project Features")
    feat_cols = st.columns(3)
    features = [
        ("Multi-Model Training", "Logistic Regression, Random Forest, XGBoost, LightGBM with GridSearchCV"),
        ("Feature Engineering", "Income-to-Loan Ratio, Credit-per-Loan, Age-Credit Interaction, Has Existing Loans"),
        ("SHAP Explanations", "Beeswarm summary and per-prediction waterfall plots"),
        ("MLflow Tracking", "Full experiment logging — params, metrics, model artifacts"),
        ("sklearn Pipeline", "ColumnTransformer prevents data leakage, simplifies deployment"),
        ("Batch Prediction", "Upload CSV for bulk scoring via the Streamlit app"),
    ]
    for i, (title, desc) in enumerate(features):
        feat_cols[i % 3].markdown(f"**{title}**\n\n{desc}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Single Prediction":
    st.title("Single Prediction")
    st.markdown("Enter applicant details and click **Predict** to see the decision with an explanation.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", min_value=18, max_value=75, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, value=45000, step=1000)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)

        with col2:
            employment_status = st.selectbox("Employment Status",
                                              options=["Employed", "Self-Employed", "Unemployed", "Student"])
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=12000, step=500)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        input_df = build_input_df(age, income, credit_score, employment_status, existing_loans, loan_amount)
        input_df = engineer_features_df(input_df)

        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        # ── Result display ──────────────────────────────────────────────
        st.markdown("---")

        if prediction == 1:
            st.success(f"APPROVED — Approval probability: {probability:.1%}")
        else:
            st.error(f"REJECTED — Approval probability: {probability:.1%}")

        # ── Probability gauge ───────────────────────────────────────────
        st.subheader("Approval Probability")
        gauge_val = int(probability * 100)
        st.progress(gauge_val)
        st.caption(f"{probability:.1%} confidence in approval")

        # ── SHAP waterfall ──────────────────────────────────────────────
        try:
            import shap
            import matplotlib.pyplot as plt

            clf = pipeline.named_steps["clf"]
            preproc = pipeline.named_steps["preprocessor"]

            # Get training data for background
            data_path = os.path.join(PROJECT_ROOT, "data", "credit_data.csv")
            train_df = pd.read_csv(data_path)
            from src.preprocessing import handle_missing_values, engineer_features as eng_feat
            train_df = handle_missing_values(train_df)
            train_df = eng_feat(train_df)
            X_train_bg = train_df.drop(columns=["Approved"])

            X_bg_transformed = preproc.transform(X_train_bg[:100])
            input_transformed = preproc.transform(input_df)

            if hasattr(clf, "apply"):
                explainer = shap.TreeExplainer(clf, X_bg_transformed)
            else:
                explainer = shap.LinearExplainer(clf, X_bg_transformed)

            shap_values = explainer.shap_values(input_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                           else explainer.expected_value[1],
                data=input_transformed[0],
                feature_names=feature_names,
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.waterfall_plot(explanation, show=False)
            st.subheader("SHAP Explanation — Why this decision?")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

        # ── Input summary ───────────────────────────────────────────────
        with st.expander("View input details"):
            st.dataframe(input_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown("Upload a CSV file with applicant data to get predictions for multiple rows at once.")

    st.markdown("**Expected columns:** Age, Income, Credit_Score, Employment_Status, Existing_Loans, Loan_Amount")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(batch_df.head(10), use_container_width=True)

            # Engineer features
            from src.preprocessing import handle_missing_values as _hmv, engineer_features as _ef
            batch_df = _hmv(batch_df)
            batch_df = _ef(batch_df)

            # Predict
            predictions = pipeline.predict(batch_df)
            probabilities = pipeline.predict_proba(batch_df)[:, 1]

            result_df = batch_df.copy()
            result_df["Prediction"] = ["Approved" if p == 1 else "Rejected" for p in predictions]
            result_df["Approval_Probability"] = [f"{prob:.1%}" for prob in probabilities]

            st.subheader("Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            # ── Aggregate stats ──────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            approval_count = (predictions == 1).sum()
            col1.metric("Total Applicants", len(predictions))
            col2.metric("Approved", approval_count)
            col3.metric("Approval Rate", f"{approval_count / len(predictions):.1%}")

            # ── Download button ──────────────────────────────────────────
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="credit_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison Dashboard")

    if not comparison_results:
        st.warning("No comparison results found. Run the training pipeline first.")
    else:
        comp_df = pd.DataFrame(comparison_results)

        # ── Metrics comparison ───────────────────────────────────────────
        st.subheader("Performance Metrics")
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC_AUC": "{:.4f}",
            }).highlight_max(subset=["F1", "Accuracy", "ROC_AUC"], color="#2ecc71"),
            use_container_width=True,
        )

        # ── Bar chart ───────────────────────────────────────────────────
        st.subheader("F1 Score Comparison")
        chart_data = comp_df[["Model", "F1"]].sort_values("F1", ascending=True)
        st.bar_chart(chart_data.set_index("Model"))

        # ── Multi-metric chart ───────────────────────────────────────────
        st.subheader("All Metrics")
        metrics_to_show = st.multiselect(
            "Select metrics to display",
            options=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
            default=["Accuracy", "F1", "ROC_AUC"],
        )
        if metrics_to_show:
            chart_df = comp_df[["Model"] + metrics_to_show].set_index("Model")
            st.bar_chart(chart_df)

        # ── Saved plots ─────────────────────────────────────────────────
        st.subheader("Evaluation Plots (Champion Model)")
        plot_files = {
            "Confusion Matrix": "confusion_matrix.png",
            "Feature Importance": "feature_importance.png",
            "ROC Curve": "roc_curve.png",
            "Precision-Recall Curve": "precision_recall_curve.png",
            "SHAP Beeswarm": "shap_beeswarm.png",
            "SHAP Bar": "shap_bar.png",
        }
        for label, filename in plot_files.items():
            filepath = os.path.join(PLOTS_DIR, filename)
            if os.path.exists(filepath):
                with st.expander(label):
                    st.image(filepath, use_container_width=True)