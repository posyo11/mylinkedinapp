import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.markdown("# Welcome to the LinkedIn User App!")

st.markdown("### Please enter your inputs below and press enter.")

def linkedin_app(new_data):
    
    coefficients = [0.24463705,  0.37186463,  0.32107696, -0.25360688, -0.23406452, -0.02684652]
    intercept = -2.0623808081527537
    log_odds = intercept + np.dot(coefficients, new_data)
    probability = 1 / (1 + np.exp(-log_odds))
    probability_percent = f'{probability*100:.2f}%'
    if probability >= 0.5:
        classification = "LinkedIn User"
    else:
        classification = "Not a LinkedIn User"

    st.markdown(f"<h2 style='text-align: center;'>{classification}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>LinkedIn Use Probability: {probability_percent}</h3>", unsafe_allow_html=True)

income = st.number_input('Income (1. Less than $10,000, 2. $10,000 to under $20,000, 3. $20,000 to under $30,000, 4. $30,000 to under $40,000, 5. $40,000 to under $50,000, 6. $50,000 to under $75,000, 7. $75,000 to under $100,000, 8, $100,000 to under $150,000, OR 9. $150,000+', min_value = 0, max_value = 9, step = 1)
education = st.number_input('Education (1. Less than High School, 2. High School Incomplete, 3. High School Graduate, 4. Some College (No Degree), 5. Two-Year Associate Degree, 6. Four-Year College Degree, 7. Some Postgraduate School (No Postgraduate Degree), 8. Postgraduate Degree', min_value = 0, max_value = 8, step = 1)
parent = st.selectbox('Parent (1 for Yes, 0 for No)', [1, 0])
married = st.selectbox('Married (1 for Yes, 0 for No)', [1, 0])
female = st.selectbox('Female (1 for Yes, 0 for No)', [1, 0])
age = st.number_input('Age', min_value = 0, max_value = 97, step = 1)

new_data = [income, education, parent, married, female, age]

linkedin_app(new_data)