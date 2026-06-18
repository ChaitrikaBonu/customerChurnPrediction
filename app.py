import streamlit as st
import pickle
import pandas as pd

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

st.write("Enter customer details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)

if st.button("Predict"):

    gender = 1 if gender == "Male" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0

    data = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer likely to Churn")
    else:
        st.success("Customer likely to Stay")