
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection App")
st.write("This app uses a trained XGBoost model to predict whether a credit card transaction is fraudulent or legitimate.")

st.markdown("---")

st.subheader("Project Context")
st.write(
    "Credit card fraud detection is a highly imbalanced classification problem. "
    "The goal is to identify suspicious transactions while reducing false alerts for genuine customers."
)

st.subheader("Upload Transaction Data")

uploaded_file = st.file_uploader("Upload a CSV file with transaction features", type=["csv"])

model = joblib.load("models/xgboost_fraud_detection_model.pkl")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(input_data.head())

    if "Class" in input_data.columns:
        input_data = input_data.drop("Class", axis=1)

    expected_features = model.get_booster().feature_names

    missing_cols = [col for col in expected_features if col not in input_data.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        input_data = input_data[expected_features]

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        results = input_data.copy()
        results["Fraud_Probability"] = probabilities
        results["Prediction"] = predictions
        results["Prediction_Label"] = results["Prediction"].map({
            0: "Legitimate",
            1: "Fraudulent"
        })

        st.subheader("Prediction Results")
        st.dataframe(results[["Fraud_Probability", "Prediction_Label"]].head(20))

        fraud_count = int((results["Prediction"] == 1).sum())
        total_count = len(results)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total_count)
        col2.metric("Predicted Fraud", fraud_count)
        col3.metric("Predicted Legitimate", total_count - fraud_count)

        st.subheader("Business Recommendation")

        if fraud_count > 0:
            st.warning(
                "Some transactions were flagged as fraudulent. "
                "Recommended action: send flagged transactions for manual review or additional verification."
            )
        else:
            st.success(
                "No fraudulent transactions were detected in the uploaded sample. "
                "Recommended action: continue monitoring."
            )

else:
    st.info("Please upload a CSV file to generate fraud predictions.")
