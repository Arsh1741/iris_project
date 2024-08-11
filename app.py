import numpy as np
import pandas as pd
import pickle
import streamlit as st


st.set_page_config(page_title='Iris Project Arshad', layout='wide')

st.title('Iris Project')

# Load Dataset
iris_data = pd.read_csv('notebook/iris.csv')

# Prepare the data for dropdown menus
sepal_length = sorted(iris_data['SepalLengthCm'].unique())
sepal_width = sorted(iris_data['SepalWidthCm'].unique())
patel_length = sorted(iris_data['PetalLengthCm'].unique())
patel_width = sorted(iris_data['PetalWidthCm'].unique())

sep_len = st.selectbox('Sepal Length in CM:', sepal_length)
sep_wid = st.selectbox('Sepal Width in CM:', sepal_width)
pet_len = st.selectbox('Petal Length in Cm:', patel_length)
pet_wid = st.selectbox('Petal Width in Cm:', patel_width)


submit = st.button('Predict')


st.subheader('Predictions Are:')


def predict_species(scaler_path, model_path):
    with open(scaler_path, 'rb') as file1:
        scaler = pickle.load(file1)
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    
    dct = {
        'SepalLengthCm': [sep_len],
        'SepalWidthCm': [sep_wid],
        'PetalLengthCm': [pet_len],
        'PetalWidthCm': [pet_wid]
    }
    xnew = pd.DataFrame(dct)
    
    xnew_pre = scaler.transform(xnew)
    
    pred = model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    max_prob = np.max(probs)
    
    return pred, max_prob

if submit:
    scaler_path = 'notebook/scaler.pkl'  
    model_path = 'notebook/model.pkl'    
    pred, max_prob = predict_species(scaler_path, model_path)
    st.subheader(f'Predicted Species is: {pred[0]}')
    st.subheader(f'Probability of Prediction: {max_prob:.2f}')
    st.progress(max_prob)