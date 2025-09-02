import pandas as pd
import numpy as np
import streamlit as st 
import joblib

# load the saved model
model = joblib.load("gradients_boost_models.pkl")

# add title---
st.title("Loan Default Prediction App")

# Accept User Input
loanamount = st.number_input("Loan Amount", min_value=100, max_value=1000000, value=50000)
termdays = st.slider("Loan Term (days)", 10, 720, 90)
repayment_curr_ratio = st.slider("Repayment Current Ratio", 0.0, 2.0, 1.0)
num_prev_loans = st.number_input("Number of Previous Loans", 0.00, 50.00, 3.00)
avg_repay_delay_days = st.number_input("Average Repay Delay (days)", -50.00, 365.00, 10.00)
total_firstrepaid_late = st.number_input("Total First Repaid Late", 0.00, 50.00, 2.00)
avg_prev_repayment_ratio = st.number_input("Avg Previous Repayment Ratio", 0.0, 2.0, 1.0)
avg_duration_days = st.slider("Avg Duration of Previous Loans (days)", 0.00, 720.00, 180.00)
avg_prev_interest = st.number_input("Avg Previous Interest", 0.00, 100000.00, 5000.00)
age = st.slider("Client Age", 18, 100, 30)
bank_account_type = st.selectbox("Bank Account Type", ['Other', 'Savings', 'Current'])
employment_status_clients = st.selectbox("Employment Status", ['Permanent', 'Unknown', 'Unemployed', 'Self-Employed', 'Student', 'Retired', 'Contract'])

# --- Predict Button ---
if st.button("Predict Loan Default Risk"):

    # create dict
    user_input = {
        'loanamount': loanamount,
        'termdays': termdays,
        'repayment_curr_ratio': repayment_curr_ratio,
        'num_prev_loans': num_prev_loans,
        'avg_repay_delay_days': avg_repay_delay_days,
        'total_firstrepaid_late': total_firstrepaid_late,
        'avg_prev_repayment_ratio': avg_prev_repayment_ratio,
        'avg_duration_days': avg_duration_days,
        'avg_prev_interest': avg_prev_interest,
        'age': age,
        'bank_account_type': bank_account_type,
        'employment_status_clients': employment_status_clients
    }

    # convert to DataFrame
    df = pd.DataFrame([user_input])

    # --- feature engineering ---
    df['late_payment_rate'] = df['total_firstrepaid_late'] / (df['num_prev_loans'] + 1e-6)
    df['repayment_efficiency'] = df['repayment_curr_ratio'] / (df['avg_prev_repayment_ratio'] + 1e-6)

    # --- "square" transformations (reverse of sqrt used in training) ---
    df['sqrt_loanamount'] = np.square(df['loanamount'])
    df['sqrt_termdays'] = np.square(df['termdays'])
    df['sqrt_avg_prev_interest'] = np.square(df['avg_prev_interest'])
    df['sqrt_repayment_efficiency'] = np.square(df['repayment_efficiency'])
    df['sqrt_late_payment_rate'] = np.square(df['late_payment_rate'])

    # --- final features for the model ---
    features = [
        'repayment_curr_ratio', 'num_prev_loans', 'avg_repay_delay_days',
        'total_firstrepaid_late', 'avg_prev_repayment_ratio',
        'avg_duration_days', 'age',
        'sqrt_late_payment_rate', 'sqrt_termdays',
        'sqrt_loanamount', 'sqrt_avg_prev_interest',
        'sqrt_repayment_efficiency',
        'bank_account_type', 'employment_status_clients'
    ]
    
    # select the columns correctly
    X = df[features]

   # --- Predict probability of being 'good' ---
    proba_good = model.predict_proba(X)[0, 1]  # probability of no default

    # --- Convert to credit score ---
    min_score, max_score = 300, 850
    credit_score = min_score + (max_score - min_score) * proba_good

    # --- Optional: classify based on threshold ---
    good_threshold = 575
    classification = "Good" if credit_score >= good_threshold else "Bad"

    # --- Display results ---
    st.write(f"💳 Credit Score: {credit_score:.0f}")
    st.write(f"Probability of Good Loan: {proba_good:.2f}")
    if classification == "Good":
        st.success("✅ Safe Loan: This client is likely to repay (Good Loan)")
    else:
        st.error("⚠️ Risky Loan: This client is likely to default (Bad Loan)")

