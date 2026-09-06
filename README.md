# Loan Approval Prediction — Decision Tree

## Problem
Predict whether a bank loan application will be approved based on applicant details
such as income, credit history, education, and property area.

## Dataset
614 loan applicants, 13 features (Kaggle Loan Prediction dataset). The data included
missing values in 7 columns, handled via mode imputation (categorical features) and
median imputation (numerical features).

## Approach
1. Data cleaning — filled missing values, encoded categorical columns to numeric
2. Baseline (unconstrained) Decision Tree → severe overfitting (train accuracy = 100%,
   test accuracy = 75.6%)
3. Pruning experiments — tested `max_depth` (1–15) and `min_samples_leaf` (1–30) to
   reduce overfitting
4. GridSearchCV with 5-fold cross-validation to select the best hyperparameters
   objectively

## Key Finding
`Credit_History` alone accounts for 93.3% of the model's feature importance — by far
the dominant predictor of loan approval. The best-generalizing model turned out to be
the simplest one: a single-split tree (depth=1), achieving 85.4% test accuracy and
outperforming every deeper tree tried. This shows that more complexity does not always
mean a better model — sometimes one strong signal beats twelve weak ones.

- Applicants with poor credit history are rejected ~91% of the time
- Applicants with good credit history are approved ~78% of the time

## Tech stack
Python, pandas, scikit-learn, matplotlib
