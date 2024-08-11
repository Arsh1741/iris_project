import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title='Iris Project Lokesh', layout='wide')

# Add a title in the body of the browser
st.title('Iris Project')

# Take sepal length, sepal width, petal length, and petal width as input from the user
sep_len = st.number_input('Sepal Length in CM:', min_value=0.00, step=0.01)
sep_wid = st.number_input('Sepal Width in CM:', min_value=0.00, step=0.01)
pet_len = st.number_input('Petal Length in Cm:', min_value=0.00, step=0.01)
pet_wid = st.number_input('Petal Width in Cm:', min_value=0.00, step=0.01)

# Add a button for prediction
submit = st.button('Predict')

# Add a subheader for predictions
st.subheader('Predictions Are:')

# Function to predict species along with probability
def predict_species(scaler_path, model_path):
    with open(scaler_path, 'rb') as file1:
        scaler = pickle.load(file1)
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    
    # Prepare the input data for prediction
    dct = {
        'SepalLengthCm': [sep_len],
        'SepalWidthCm': [sep_wid],
        'PetalLengthCm': [pet_len],
        'PetalWidthCm': [pet_wid]
    }
    xnew = pd.DataFrame(dct)
    
    # Scale the input data
    xnew_pre = scaler.transform(xnew)
    
    # Predict the species and the probability
    pred = model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    max_prob = np.max(probs)
    
    return pred, max_prob

# Show the results in Streamlit
if submit:
    # Specify correct paths or check the path where the script is being executed
    scaler_path = 'notebook/scaler.pkl'  # Ensure this path is correct
    model_path = 'notebook/model.pkl'    # Ensure this path is correct
    
    try:
        pred, max_prob = predict_species(scaler_path, model_path)
        st.subheader(f'Predicted Species is: {pred[0]}')
        st.subheader(f'Probability of Prediction: {max_prob:.2f}')
        st.progress(max_prob)
    except FileNotFoundError:
        st.error('Scaler or model file not found. Please check the paths.')
    except Exception as e:
        st.error(f'An error occurred: {e}')