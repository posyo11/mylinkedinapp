import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.markdown("# Welcome to the LinkedIn User App!")

st.markdown("### Please enter your inputs below and press enter.")

def linkedin_app(new_data):
    
    s = pd.read_csv('social_media_usage.csv')
    
    def clean_sm(x):
        x = np.where(x == 1, 1, 0)
        return x
        
    ss = pd.DataFrame()
    ss['sm_li'] = clean_sm(s['web1h'])
    ss['income'] = s['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)
    ss['education'] = s['educ2'].apply(lambda x: x if 1 <= x <= 8 else np.nan)
    ss['parent'] = s['par'].apply(lambda x: 1 if x==1 else (0 if x==2 else np.nan))
    ss['married'] = s['marital'].apply(lambda x: 1 if x==1 else (0 if 2 <= x <= 6 else np.nan))
    ss['female'] = s['gender'].apply(lambda x: 1 if x==2 else (0 if x in [1, 3] else np.nan))
    ss['age'] = s['age'].apply(lambda x: x if 1 <= x <= 97 else np.nan)
    ss.dropna(inplace=True)
    
    y = ss['sm_li']
    X = ss.drop(columns=['sm_li'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    linkedin_model = LogisticRegression(class_weight='balanced')
    linkedin_model.fit(X_train, y_train)

    return linkedin_model.predict_proba(new_data)[0][1]

income = st.number_input('Income (1. Less than $10,000, 2. $10,000 to under $20,000, 3. $20,000 to under $30,000, 4. $30,000 to under $40,000, 5. $40,000 to under $50,000, 6. $50,000 to under $75,000, 7. $75,000 to under $100,000, 8, $100,000 to under $150,000, OR 9. $150,000+', min_value = 0, max_value = 9, step = 1)
education = st.number_input('Education (1. Less than High School, 2. High School Incomplete, 3. High School Graduate, 4. Some College (No Degree), 5. Two-Year Associate Degree, 6. Four-Year College Degree, 7. Some Postgraduate School (No Postgraduate Degree), 8. Postgraduate Degree', min_value = 0, max_value = 8, step = 1)
parent = st.selectbox('Parent (1 for Yes, 0 for No)', [1, 0])
married = st.selectbox('Married (1 for Yes, 0 for No)', [1, 0])
female = st.selectbox('Female (1 for Yes, 0 for No)', [1, 0])
age = st.number_input('Age', min_value = 0, max_value = 97, step = 1)

new_data = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [married],
    'female': [female],
    'age': [age]
})

linkedin_app(new_data)