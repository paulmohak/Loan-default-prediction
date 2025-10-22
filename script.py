"""
Loan Default Prediction using Random Forest

Goal:
Predict whether a loan applicant will default on their loan. 

Pipeline:
1. Load and inspect dataset
2. Encode categorical variables
3. Scale numeric features
4. Split data into train/test
5. Train Random Forest classifier
6. Evaluate model using precision-recall and adjust threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\Sanjul Paul\Desktop\Loan_default.csv")
print("Dataset loaded successfully.")
print(df.head())
print(df.info())
print("Missing values in each column:\n", df.isnull().sum())

# 2. Encode categorical variables
# One-hot encoding avoids any ordinal assumption
loan_df = pd.get_dummies(
    df,
    columns=['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose'],
    drop_first=True
)
print("Categorical columns encoded successfully.")

# 3. Scale numeric features
# Scaling ensures features are on similar range for model stability
num_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'NumCreditLines', 'InterestRate',
    'LoanTerm', 'DTIRatio'
]
scaler = StandardScaler()
loan_df[num_cols] = scaler.fit_transform(loan_df[num_cols])
print("Numeric features scaled.")

# 4. Define features and target
numeric_df = loan_df.select_dtypes(include=['number'])
X = numeric_df.drop('Default', axis=1)  # predictors
y = numeric_df['Default']               # target variable

# 5. Split dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train-test split completed.")

# 6. Train Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=200,        # number of decision trees
    max_depth=10,            # limit depth to reduce overfitting
    random_state=42,
    class_weight='balanced'  # adjusts weights to handle class imbalance
)
rf.fit(X_train, y_train)
print("Random Forest model trained.")

# 7. Predict probabilities on test set
# Probabilities are needed to tune threshold for practical recall/precision tradeoff
y_prob = rf.predict_proba(X_test)[:, 1]

# 8. Precision-Recall vs Threshold visualization
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,5))
plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', color='red')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.show()

# 9. Choose a threshold to achieve target recall
target_recall = 0.62
recall_diff = np.abs(recall[:-1] - target_recall)
best_idx = np.argmin(recall_diff)

# Ensure precision is acceptable (avoid too many false positives)
best_threshold = thresholds[best_idx]
while precision[best_idx] < 0.2 and best_idx < len(thresholds)-1:
    best_idx += 1
    best_threshold = thresholds[best_idx]

print("Chosen threshold for deployment:", best_threshold)

# 10. Apply threshold and evaluate
y_pred_adjusted = (y_prob > best_threshold).astype(int)
print("Evaluation report at chosen threshold:")
print(classification_report(y_test, y_pred_adjusted))
# 11. Feature Importance
importances = rf.feature_importances_
features = X.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Random Forest Feature Importance")
plt.bar(range(len(features)), importances[indices], color="skyblue")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

