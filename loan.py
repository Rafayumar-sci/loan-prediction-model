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
    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=0,
        value=100000,
        step=10000,
        help="Amount of loan requested"
    )

    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        value=50000,
        step=5000,
        help="Applicant's annual income"
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10,
        help="Credit score (300-850)"
    )

with col2:
    # Additional inputs
    employment_years = st.number_input(
        "Employment Years",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Years of employment history"
    )

    debt_to_income = st.slider(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=5.0,
        help="Percentage of income going to debt"
    )

    loan_type = st.selectbox(
        "Loan Type",
        options=list(label_encoders.get('Loan_Type', {}).keys())
        if 'Loan_Type' in label_encoders else ["Personal", "Home", "Auto", "Business"]
    )

# Make prediction
st.markdown("---")

if st.button("🔮 Predict Loan Approval", use_container_width=True):
    try:
        # Prepare input data
        input_data = {
            'Loan_Amount': loan_amount,
            'Annual_Income': annual_income,
            'Credit_Score': credit_score,
            'Employment_Years': employment_years,
            'Debt_to_Income': debt_to_income,
            'Loan_Type': loan_type
        }

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Encode categorical variables if needed
        for col in df.columns:
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
            st.metric("Loan Amount", f"${loan_amount:,.0f}")
        with summary_cols[1]:
            st.metric("Annual Income", f"${annual_income:,.0f}")
        with summary_cols[2]:
            st.metric("Credit Score", f"{credit_score}")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {str(e)}")
        st.info("Make sure you've provided all required information")
