import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature names
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
model_features = joblib.load("model_features.joblib")   # list of column names

st.title("🏦 Loan Approval Prediction App")

st.header("Enter Applicant Details")

# --------------- USER INPUTS ----------------
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
LoanTerm = st.number_input("Loan Amount Term", min_value=0)
CreditScore = st.number_input("Credit Score", min_value=0)
ResidentialAsset = st.number_input("Residential Asset Value", min_value=0)
CommercialAsset = st.number_input("Commercial Asset Value", min_value=0)
LuxuryAsset = st.number_input("Luxury Asset Value", min_value=0)
BankAsset = st.number_input("Bank Asset Value", min_value=0)

# --------------- PREPROCESSING ----------------

# Convert categorical values
dep_val = 3 if Dependents == "3+" else int(Dependents)
edu_val = 1 if Education == "Graduate" else 0
emp_val = 1 if Self_Employed == "Yes" else 0

# Create DataFrame with correct feature names (order must match model)
input_data = pd.DataFrame([[ 
    dep_val,
    edu_val,
    emp_val,
    ApplicantIncome,
    LoanAmount,
    LoanTerm,
    CreditScore,
    ResidentialAsset,
    CommercialAsset,
    LuxuryAsset,
    BankAsset
]], columns=model_features)

# Scale numeric columns
num_cols = input_data.select_dtypes(include=['int64','float64']).columns
input_data[num_cols] = scaler.transform(input_data[num_cols])

# --------------- PREDICT ----------------
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")