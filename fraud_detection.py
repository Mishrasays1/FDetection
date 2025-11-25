import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Fraud Detection Prediction System")

st.markdown("Please enter the transaction details and use the predict button")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"])


amount =st.number_input("Amount", min_value= 0.0, value= 10000.0)

oldbalanceOrg = st.number_input("Old Balance (sender)", min_value=0.0, value=  10000.0)

newbalanceOrig = st.number_input("New Balance (sender)",min_value= 0.0, value=  9000.0)

oldbalanceDest = st.number_input("Old Balance (receiver)", min_value=0.0, value= 0.0)

newbalanceDest = st.number_input("New Balance (receiver)", min_value=0.0, value= 0.0)

if st.button("predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction: '{int(prediction)}'")

    if prediction == 1:
        st.error("This transaction seems Fishy")

    else:
        st.success("Transaction seems legit !")     