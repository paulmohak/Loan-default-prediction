# Loan Default Prediction using Random Forest

## Project Overview
This project predicts whether a loan applicant will default using machine learning. 
It helps financial institutions assess credit risk and make data-driven lending decisions. 

The project demonstrates a full end-to-end ML workflow, including: 
* Data preprocessing
* Modeling
* Threshold tuning
* Evaluation
* Feature importance analysis

---

## Dataset
The dataset `Loan_default.csv` is included in the `data/` folder. 

**Features include:** 
- Age
- Income
- Credit Score
- Loan Amount
- Loan Term
- Months Employed
- Employment Type
- Education Level
- Marital Status
- Loan Purpose
- Debt-to-Income Ratio (DTI)

**Target variable:** `Default` (1 = defaulted, 0 = no default)

> Note: If the original dataset is confidential, a synthetic dataset can be generated for reproducibility.

---

## Project Workflow
1. **Load and inspect dataset**  
2. **Encode categorical variables** using one-hot encoding  
3. **Scale numeric features** using standardization  
4. **Split data** into training (80%) and testing (20%) sets  
5. **Train Random Forest Classifier**  
   * n_estimators = 200  
   * max_depth = 10  
   * class_weight = balanced  
6. **Predict probabilities** instead of labels  
7. **Visualize Precision-Recall vs Threshold**  
8. **Select optimal threshold** to meet target recall while keeping precision reasonable  
9. **Apply threshold** to generate final predictions  
10. **Evaluate predictions** with classification report (precision, recall, F1-score)  
11. **Feature importance analysis** to interpret model behavior

---

## Threshold Tuning
* The model predicts probabilities rather than default labels.  
* A threshold is selected to achieve **target recall** (e.g., 0.62) while keeping precision at a reasonable level.  
* This makes the model **deployment-ready** and demonstrates real-world business considerations.

---

## Feature Importance
* Feature importance shows which variables most influence loan defaults.  
* Common top features: `Credit Score`, `DTI`, `Loan Amount`, `Income`.  
* Helps improve **interpretability** and **decision-making transparency**.  

---

## Technology Stack
* Python 3  
* Pandas & NumPy – data manipulation  
* Scikit-learn – modeling & evaluation  
* Matplotlib – visualization  

---

## How to Run
1. Clone the repository:  
2. Install dependencies:  
3. Run the script:
   
## Output
* Precision-Recall vs Threshold plot  
* Chosen optimal threshold  
* Classification report (precision, recall, F1-score)  
* Feature importance visualization  
