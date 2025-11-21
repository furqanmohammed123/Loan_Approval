# app.py
import streamlit as st
import pandas as pd
import joblib

# Load saved model & scaler
scaler = joblib.load('loan_approval_dataset2.csvscaler.joblib')
model = joblib.load('loan_approval_dataset2.csv.joblib')

st.title("🏦 Loan Approval Prediction App")

st.write("This app predicts whether a loan application will be approved or not.")

# --- Load test dataset for optional preview ---
test_df = pd.read_csv('loan_approval_dataset2.csv')

# --- Input fields for user ---
st.header("Enter Applicant Details")

# Loan_Id=st.number_input("Loan Id", min_value=0)
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Education = 1 if Education == 'Graduate' else 0
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
Self_Employed = 1 if Self_Employed == 'Yes' else 0
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_Score= st.number_input("Credit Score", min_value=0)
Residential_Asset_Value= st.number_input("Residential Asset Value", min_value=0)
Commercial_Asset_Value= st.number_input("Commercial Asset Value", min_value=0)
Luxury_Asset_Value= st.number_input("Luxury Asset Value", min_value=0)
Bank_Asset_Value= st.number_input("Bank Asset Value", min_value=0)


# --- Prepare input for model ---
input_data = pd.DataFrame({
    # 'loan_id': [Loan_Id],
    'no_of_dependents': [Dependents],
    'education': [Education],
    'self_employed': [Self_Employed],
    'income_annum': [ApplicantIncome],
    'loan_amount': [LoanAmount],
    'loan_term': [Loan_Amount_Term],
    'cibil_score': [Credit_Score],
    'residential_assets_value': [Residential_Asset_Value],
    'commercial_assets_value': [Commercial_Asset_Value],
    'luxury_assets_value': [Luxury_Asset_Value],
    'bank_asset_value': [Bank_Asset_Value]
})


# --- Preprocessing: same as training (encoding + scaling) ---
# Map categorical values same as preprocessing
input_data.replace({
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Dependents': {'3+': 3}
}, inplace=True)

# Scale numeric values
num_cols = ['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term',
            'cibil_score','residential_asset_value','commercial_asset_value','luxury_asset_value','bank_asset_value']
# input_data[num_cols] = scaler.transform(input_data[num_cols])

# --- Predict ---
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    if prediction == 1 or prediction == 'Y':
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

st.divider()
# st.subheader("📄 Example Test Data")
st.dataframe(test_df)
