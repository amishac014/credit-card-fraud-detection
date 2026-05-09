
# Credit Card Fraud Detection and Risk Scoring Using Machine Learning

## Project Overview

This project focuses on building a machine learning-based credit card fraud detection system. The goal is to identify fraudulent transactions from transaction-level data while balancing fraud detection performance and false alert reduction.

Credit card fraud is a major challenge for banks, fintech companies, payment networks, and merchants. Since fraudulent transactions are rare compared to legitimate transactions, this project deals with a highly imbalanced classification problem.

## Business Problem

Financial institutions process millions of credit card transactions every day. Even a small percentage of fraud can result in financial loss, customer dissatisfaction, and operational burden for fraud investigation teams.

The business objective of this project is to build a fraud detection model that can help identify suspicious transactions and support risk-based decision-making.

## Data Science Problem

This is a supervised binary classification problem.

The target variable is `Class`:

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

## Dataset

Original dataset:

- Total transactions: 284,807
- Fraud transactions: 492
- Legitimate transactions: 284,315
- Original columns: 31

After duplicate removal:

- Total transactions: 283,726
- Fraud transactions: 473
- Legitimate transactions: 283,253

## Key Features

The dataset includes:

- `Time`
- `Amount`
- `V1` to `V28` anonymized transaction features
- `Class` target variable

Additional engineered features:

- `is_zero_amount`
- `transaction_hour`

## Exploratory Data Analysis Findings

Key findings from EDA:

- Fraud transactions represent only about 0.17% of the dataset.
- The dataset is highly imbalanced.
- Most transactions are legitimate.
- Fraud transactions are not always high-value transactions.
- Many fraud transactions are small.
- Zero-amount transactions have a higher fraud rate than the overall dataset.
- Fraud rate varies across transaction hours.

## Models Used

Three machine learning models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

## Model Performance

| Model | ROC-AUC | PR-AUC | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9684 | 0.6721 | 0.06 | 0.87 | 0.10 |
| Random Forest | 0.9193 | 0.8021 | 0.97 | 0.69 | 0.81 |
| XGBoost | 0.9755 | 0.7978 | 0.45 | 0.81 | 0.58 |

## Final Model Selection

XGBoost was selected as the recommended balanced model because it achieved strong fraud recall, better precision than Logistic Regression, and the highest ROC-AUC score.

Random Forest can be considered a conservative alternative when minimizing false positives is the top business priority.

## Business Impact

This fraud detection model can help financial institutions:

- Identify suspicious transactions
- Reduce fraud-related losses
- Support fraud operations teams
- Improve transaction risk monitoring
- Balance fraud prevention with customer experience

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Joblib

## Project Files

- `creditfrad.ipynb` — Main notebook
- `eda_summary_table.csv` — EDA summary results
- `final_model_comparison_results.csv` — Final model comparison table
- `fraud_class_performance_comparison.png` — Fraud performance chart
- `roc_pr_auc_model_comparison.png` — ROC-AUC and PR-AUC comparison chart
- `xgboost_fraud_detection_model.pkl` — Saved final model

## Conclusion

This project shows how machine learning can be used to detect fraudulent credit card transactions in a highly imbalanced dataset. The analysis highlights the importance of using fraud-specific evaluation metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC instead of relying only on accuracy.

The final recommended model, XGBoost, provides a balanced solution for fraud detection by catching a high percentage of fraud cases while reducing false alerts compared with the baseline model.
