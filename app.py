# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# scaler is exported as scaler.pkl
# model is exported as model.pkl
# order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'], dtype='object'

import streamlit as st
import numpy as np
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model.pkl')

st.title('Churn Prediction App')
st.divider()
st.write('Please enter the values and hit the Predict button for getting the prediction.')
st.divider()

gender = st.selectbox('Enter the gender', ['Female', 'Male'])
tenure = st.number_input('Enter Tenure', min_value=0, max_value=130, value=10)
monthlycharge = st.number_input('Enter monthly charge', min_value=30, max_value=150)
age = st.number_input('Enter Age', min_value=10, max_value=100, value=30)

st.divider()

predictbutton = st.button('Predict!')
st.divider()
if predictbutton:
    gender_selected = 1 if gender == 'Female' else 0
    X = [age, gender_selected, tenure, monthlycharge]
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    predicted = 'Churn' if prediction == 1 else 'Not Churn'
    st.balloons()
    st.write(f'predicted: {predicted}')
else:
    st.write('Please enter the values and use Predict button')




# handling imbalanced data


# logistic regression model -> accuracy 87.5%
# KNN classifier model -> accuracy 87%          (GridSearchCV: finding the best hyperparameters)/('uniform': All neighbors contribute equally(simple majority vote).'distance': Closer neighbors have more influence than farther ones(weighted by inverse distance')/ cv=5 -> 5-fold cross-validation )

# SVM model (suport vector machine) -> accuracy 89% 
# decision tree classifier model -> accuracy 87.5% 
# random forest classifier model -> accuracy 90.5%