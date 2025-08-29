
🏦 Problem Statement

Credit rating is one of the most critical aspects in financial decision-making. Banks, NBFCs, and lending institutions rely on credit scores to determine a borrower's repayment ability. Incorrect predictions may lead to loan defaults or loss of customers.

The challenge is to build a robust ML-based credit rating prediction system that can generalize well across different borrowers while maintaining high accuracy.

🎯 Objectives

Build a machine learning model to predict credit rating (Good / Bad).

Optimize the model using cross-validation and hyperparameter tuning.

Deploy the model in an interactive Streamlit web app so that anyone can test predictions on new data.

Ensure the pipeline is scalable, reproducible, and explainable.

🛠️ Methodology
1️⃣ Data Preprocessing

Handled missing values (mean/median imputation).

Encoded categorical features (LabelEncoding, OneHotEncoding).

Scaled numerical variables for consistent feature range.

Final feature set had 11 input variables.

2️⃣ Model Development

Selected XGBoost Classifier (robust with imbalanced data, handles non-linearity well).

Performed 5-Fold Cross Validation to avoid overfitting.

Used GridSearchCV for hyperparameter tuning (learning rate, max_depth, n_estimators).

3️⃣ Model Evaluation

Accuracy: 87.3%

Weighted F1-score: 0.85

Confusion Matrix showed balanced performance across classes.

ROC-AUC Curve: Strong separation between good and bad credit classes.

4️⃣ Deployment

Trained model exported as xgb_model.pkl.

Built a Streamlit app (app.py) with an intuitive user interface.

Users can input borrower details and instantly get a Credit Rating Prediction.

📊 Key Insights from the Model

Features like Income, Existing Loan Amount, Default History, and Credit Utilization Ratio had the highest impact on predictions.

XGBoost outperformed Logistic Regression and Random Forest during testing.

Cross-validation helped maintain generalization with minimal variance across folds.

🚀 Impact of the Project

Provides a scalable solution for banks/fintech companies to automate risk assessment.

Reduces manual evaluation time while maintaining decision accuracy.

Can be extended to multi-class ratings (AAA, AA, A, BBB, etc.).

🌐 End-to-End Workflow

Data → Preprocessing → Feature Engineering → Model Training (XGBoost) → Hyperparameter Tuning → Model Evaluation → Export Model → Streamlit Deployment → Real-Time Prediction
