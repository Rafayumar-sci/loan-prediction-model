import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Loan Prediction Model", layout="centered")

# Load the model and label encoders


@st.cache_resource
def load_model():
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, label_encoders


model, label_encoders = load_model()

# App title and description
st.title("🏦 Loan Prediction Model")
st.markdown("---")
st.write("Enter the applicant details below to predict loan approval")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Applicant's age"
    )

    person_income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        value=50000,
        step=5000,
        help="Applicant's annual income"
    )

    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Years of employment history"
    )

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=0,
        value=10000,
        step=1000,
        help="Amount of loan requested"
    )

with col2:
    # Categorical inputs
    person_home_ownership = st.selectbox(
        "Home Ownership",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
        help="Home ownership status"
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        options=["PERSONAL", "EDUCATION", "MEDICAL",
                 "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        help="Purpose of the loan"
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        options=["A", "B", "C", "D", "E", "F", "G"],
        help="Loan grade based on creditworthiness"
    )

    cb_person_default_on_file = st.selectbox(
        "Has Defaulted Before?",
        options=["N", "Y"],
        help="Has the person defaulted on a loan before?"
    )

# Additional numerical inputs
loan_percent_income = st.slider(
    "Loan Percent Income (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="Loan amount as percentage of annual income"
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0,
    max_value=50,
    value=5,
    step=1,
    help="Length of credit history"
)

# Make prediction
st.markdown("---")

if st.button("🔮 Predict Loan Approval", use_container_width=True):
    try:
        # Prepare input data
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_percent_income': loan_percent_income / 100,  # Convert to decimal
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Encode categorical variables
        categorical_cols = ['person_home_ownership',
                            'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        for col in categorical_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if prediction == 1:
            st.success("✅ Loan APPROVED", icon="✅")
            approval_prob = probability[1] * 100
            st.metric("Approval Probability", f"{approval_prob:.2f}%")
        else:
            st.error("❌ Loan REJECTED", icon="❌")
            rejection_prob = probability[0] * 100
            st.metric("Rejection Probability", f"{rejection_prob:.2f}%")

        # Show key metrics summary
        st.markdown("#### 📋 Application Summary")
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Loan Amount", f"${loan_amnt:,.0f}")
        with summary_cols[1]:
            st.metric("Annual Income", f"${person_income:,.0f}")
        with summary_cols[2]:
            st.metric("Age", f"{person_age}")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {str(e)}")
        st.info("Make sure you've provided all required information")
