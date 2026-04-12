"""
Streamlit app for Credit Card Approval Prediction.

Loads the trained model and preprocessing artifacts, presents an
input form, and displays the prediction result.
"""

import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "..", "models", "artifacts.pkl")


# ── Load model & artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    return model, artifacts


model, artifacts = load_artifacts()
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
feature_names = artifacts["feature_names"]


# ── App UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Card Approval", page_icon="💳", layout="centered")
st.title("Credit Card Approval Prediction")
st.markdown("Enter applicant details below and click **Predict** to see the decision.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=75, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, value=45000, step=1000)
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)

    with col2:
        employment_status = st.selectbox("Employment Status", options=encoders["Employment_Status"].classes_)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=12000, step=500)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode the categorical input using the saved encoder
    emp_encoded = encoders["Employment_Status"].transform([employment_status])[0]

    input_df = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Credit_Score": credit_score,
        "Employment_Status": emp_encoded,
        "Existing_Loans": existing_loans,
        "Loan_Amount": loan_amount,
    }])

    # Ensure column order matches training
    input_df = input_df[feature_names]

    # Scale
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=feature_names,
    )

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"**APPROVED** — Approval probability: {probability:.1%}")
    else:
        st.error(f"**REJECTED** — Approval probability: {probability:.1%}")