
# Credit Card Approval Prediction

End-to-end ML project that predicts whether a credit card application will be **Approved** or **Rejected** using Logistic Regression with hyperparameter tuning via GridSearchCV.

## Project Overview

| Component | Details |
|---|---|
| **Algorithm** | Logistic Regression |
| **Tuning** | GridSearchCV (C, penalty, solver) |
| **Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **App** | Streamlit web interface |

### Features Used

- Age
- Annual Income
- Credit Score
- Employment Status (Employed / Self-Employed / Unemployed / Student)
- Number of Existing Loans
- Loan Amount

## Project Structure

```
credit_card_approval/
├── data/
│   ├── generate_data.py      # Script to generate synthetic dataset
│   └── credit_data.csv        # Generated dataset
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data loading, cleaning, encoding, scaling
│   ├── model.py               # Logistic Regression + GridSearchCV
│   ├── train.py               # Training pipeline (entry point)
│   └── evaluate.py            # Metrics, confusion matrix, ROC
├── app/
│   └── app.py                 # Streamlit web app
├── models/                    # Saved model & artifacts (created after training)
├── plots/                     # Evaluation plots (created after training)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone or navigate to the project
cd credit_card_approval

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

### 1. Generate the Dataset

```bash
python data/generate_data.py
```

### 2. Train the Model

```bash
cd /path/to/credit_card_approval
python src/train.py
```

This will:
- Preprocess the data
- Run GridSearchCV for hyperparameter tuning
- Print evaluation metrics
- Save the trained model to `models/model.pkl`
- Save preprocessing artifacts to `models/artifacts.pkl`
- Generate plots in `plots/`

### 3. Launch the Streamlit App

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`, fill in the applicant details, and click **Predict**.

## Screenshots

<!-- Add screenshots after running the app -->
*Coming soon — run the app and take a screenshot!*

## Evaluation

Typical results on the synthetic dataset:

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~0.93  |
| Precision  | ~0.85  |
| Recall     | ~0.70  |
| F1 Score   | ~0.77  |

*Actual numbers may vary depending on the random seed and dataset generation.*

## License

MIT