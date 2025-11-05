# Customer Churn Prediction | End-to-End ML Pipeline

## Project Overview
This project demonstrates an **end-to-end machine learning pipeline** for predicting customer churn.  
The workflow covers **data preprocessing**, **handling imbalanced datasets**, **feature encoding**, **model training**, **hyperparameter tuning**, **scaling**, and **performance evaluation**.  
The goal is to **identify potential churners** and provide actionable insights for customer retention strategies.

---

##  Tools & Technologies
- **Python (Pandas, NumPy, Matplotlib, Scikit-learn)** – Data processing, modeling, evaluation  
- **Imbalanced-learn (SMOTE)** – Handling class imbalance  
- **Joblib** – Saving models and scalers  
- **Jupyter Notebook** – Interactive development and visualization  

---

##  Project Workflow
1. **Data Loading & Cleaning** – Handle missing values and duplicates  
2. **Exploratory Data Analysis (EDA)** – Analyze churn patterns by age, tenure, contract type, and monthly charges  
3. **Feature Engineering** – Encode categorical variables, scale numerical features  
4. **Handling Imbalance** – Apply SMOTE to oversample minority class (churned customers)  
5. **Model Training & Hyperparameter Tuning** – Logistic Regression, KNN, SVM, Decision Tree, Random Forest  
6. **Evaluation Metrics** – Accuracy, precision, recall, F1-score, and confusion matrix  
7. **Model Selection & Saving** – Identify the best-performing model and save it for deployment  

---

## 📊 Model Performance Comparison

| Model                     | Accuracy | Precision | Recall | F1-score |
|----------------------------|----------|-----------|--------|----------|
| Logistic Regression        | 0.82     | 0.79      | 0.75   | 0.77     |
| K-Nearest Neighbors (KNN)  | 0.81     | 0.78      | 0.73   | 0.75     |
| Support Vector Classifier  | 0.84     | 0.81      | 0.78   | 0.79     |
| Decision Tree Classifier   | 0.83     | 0.80      | 0.76   | 0.78     |
| **Random Forest Classifier** | **0.87** | **0.85**  | **0.82** | **0.83** |

✅ **Best Model:** Random Forest Classifier  
- Highest overall accuracy and F1-score  
- Handles non-linear relationships effectively  
- Suitable for deployment and real-world churn prediction  

---

##  Key Features Added
- SMOTE oversampling to address **imbalanced data**  
- Label encoding for categorical variables (`Gender`, `ContractType`, `InternetService`, `TechSupport`)  
- Stratified train-test split to maintain **class distribution**  
- Proper scaling using `StandardScaler`  
- Detailed performance metrics: **accuracy, precision, recall, F1-score, confusion matrix**  
