
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_fraud_detection_model.pkl")

model = load_model()

# Sidebar
st.sidebar.title("💳 Fraud Detection App")
st.sidebar.markdown(
    """
    This app uses a trained **XGBoost model** to predict whether uploaded credit card transactions are legitimate or fraudulent.

    **Project Focus**
    - Fraud detection
    - Risk scoring
    - Imbalanced classification
    - Business decision support
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Used")
st.sidebar.write("XGBoost Classifier")

st.sidebar.subheader("Business Goal")
st.sidebar.write(
    "Identify suspicious transactions while reducing unnecessary false alerts for genuine customers."
)

# Main title
st.title("💳 Credit Card Fraud Detection & Risk Scoring App")

st.markdown(
    """
    This app demonstrates how a machine learning model can support fraud risk decision-making.
    Upload transaction data in CSV format and the model will predict whether transactions are **Legitimate** or **Fraudulent**.
    """
)

st.markdown("---")

# Business context
st.subheader("📌 Business Context")
st.info(
    "Credit card fraud detection is a highly imbalanced classification problem. "
    "Fraudulent transactions are rare, but they can cause financial loss, customer dissatisfaction, "
    "and operational burden for fraud investigation teams."
)

# Model performance summary
st.subheader("📊 Final Model Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected Model", "XGBoost")
col2.metric("ROC-AUC", "0.9755")
col3.metric("PR-AUC", "0.7978")
col4.metric("Fraud Recall", "81%")

st.markdown("---")

# Upload section
st.subheader("📂 Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction features",
    type=["csv"]
)

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully.")

    st.subheader("🔎 Uploaded Data Preview")
    st.dataframe(input_data.head(), use_container_width=True)

    # Drop target column if included
    if "Class" in input_data.columns:
        input_data = input_data.drop("Class", axis=1)

    # Expected model features
    expected_features = model.get_booster().feature_names

    missing_cols = [col for col in expected_features if col not in input_data.columns]
    extra_cols = [col for col in input_data.columns if col not in expected_features]

    if missing_cols:
        st.error(f"The uploaded file is missing required columns: {missing_cols}")
    else:
        # Keep only expected columns in correct order
        input_data = input_data[expected_features]

        # Predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        results = input_data.copy()
        results["Fraud_Probability"] = probabilities
        results["Fraud_Risk_%"] = (probabilities * 100).round(2)
        results["Prediction"] = predictions
        results["Prediction_Label"] = results["Prediction"].map({
            0: "Legitimate",
            1: "Fraudulent"
        })

        def recommended_action(row):
            if row["Prediction"] == 1:
                return "Send to manual review"
            elif row["Fraud_Probability"] >= 0.30:
                return "Monitor closely"
            else:
                return "Approve / Continue monitoring"

        results["Recommended_Action"] = results.apply(recommended_action, axis=1)

        # Summary metrics
        total_transactions = len(results)
        predicted_fraud = int((results["Prediction"] == 1).sum())
        predicted_legitimate = total_transactions - predicted_fraud
        avg_risk = round(results["Fraud_Risk_%"].mean(), 2)

        st.subheader("📈 Prediction Summary")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Total Transactions", total_transactions)
        m2.metric("Predicted Fraud", predicted_fraud)
        m3.metric("Predicted Legitimate", predicted_legitimate)
        m4.metric("Average Fraud Risk", f"{avg_risk:.2f}%")

        # Prediction results
        st.subheader("🧾 Prediction Results")

        display_cols = [
            "Fraud_Risk_%",
            "Prediction_Label",
            "Recommended_Action"
        ]

        st.dataframe(results[display_cols].head(50), use_container_width=True)

        # Business recommendation
        st.subheader("💼 Business Recommendation")

        if predicted_fraud > 0:
            st.warning(
                f"{predicted_fraud} transaction(s) were flagged as fraudulent. "
                "Recommended action: send flagged transactions for manual review or additional customer verification."
            )
        elif avg_risk >= 30:
            st.warning(
                "No transaction was directly classified as fraud, but the average fraud risk is elevated. "
                "Recommended action: monitor these transactions closely."
            )
        else:
            st.success(
                "No fraudulent transactions were detected in the uploaded sample. "
                "Recommended action: approve or continue routine monitoring."
            )

        # Download results
        st.subheader("⬇️ Download Prediction Results")

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="fraud_prediction_results.csv",
            mime="text/csv"
        )

        # Technical note
        st.markdown("---")
        st.caption(
            "Note: This app is a portfolio demonstration. In a real financial institution, "
            "model thresholds should be selected based on fraud loss, manual review capacity, "
            "regulatory requirements, and customer experience impact."
        )

else:
    st.info("Please upload a CSV file to generate fraud predictions.")
