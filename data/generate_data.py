"""
Generate a synthetic credit card approval dataset.

Creates a realistic dataset with features correlated to the approval
outcome, then injects missing values and saves as CSV.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

# --- Continuous features ---
age = np.random.randint(18, 75, N)
income = np.random.normal(45000, 20000, N).clip(8000)
credit_score = np.random.normal(650, 100, N).clip(300, 850).astype(int)
existing_loans = np.random.poisson(1.5, N).clip(0, 8)
loan_amount = np.random.normal(12000, 8000, N).clip(500)

# --- Categorical features ---
employment_statuses = ["Employed", "Self-Employed", "Unemployed", "Student"]
employment_status = np.random.choice(employment_statuses, N, p=[0.50, 0.20, 0.18, 0.12])

# --- Target: approval probability depends on features ---
# Higher income, credit score, and employment favour approval
emp_bonus = np.array([3.0 if e == "Employed" else
                       1.5 if e == "Self-Employed" else
                       -1.0 if e == "Student" else
                       -2.0 for e in employment_status])

logit = (-6.0
         + 0.04 * (income / 1000)
         + 0.02 * (credit_score - 600)
         + 0.03 * (age - 35)
         + emp_bonus
         - 0.3 * existing_loans
         - 0.15 * (loan_amount / 1000))

prob = 1 / (1 + np.exp(-logit))
approved = (np.random.rand(N) < prob).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Income": income.round(2),
    "Credit_Score": credit_score,
    "Employment_Status": employment_status,
    "Existing_Loans": existing_loans,
    "Loan_Amount": loan_amount.round(2),
    "Approved": approved
})

# --- Inject ~5 % missing values randomly ---
for col in ["Age", "Income", "Credit_Score", "Loan_Amount"]:
    mask = np.random.rand(N) < 0.05
    df.loc[mask, col] = np.nan

for col in ["Employment_Status"]:
    mask = np.random.rand(N) < 0.03
    df.loc[mask, col] = np.nan

# Use a path relative to this script so it works on any machine
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "credit_data.csv")

df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Approval rate: {df['Approved'].mean():.2%}")
print(f"Missing values per column:\n{df.isnull().sum()}")