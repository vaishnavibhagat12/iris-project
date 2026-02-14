import pickle
import pandas as pd
import numpy as np
import streamlit as st 

def predict_species(sep_len, sep_width, pet_len, pet_width, scaler_path, model_path):
    try:
        # load the scaler
        with open(scaler_path, 'rb') as file1:
            scaler = pickle.load(file1)

        # load the model
        with open(model_path, 'rb') as file2:
            model = pickle.load(file2)

        # create dataframe
        dec = {
            'SepalLengthCm': [sep_len],
            'SepalWidthCm': [sep_width],
            'PetalLengthCm': [pet_len],
            'PetalWidthCm': [pet_width]
        }

        x_new = pd.DataFrame(dec)

        # scale input
        xnew_pre = scaler.transform(x_new)

        # prediction
        pred = model.predict(xnew_pre)
        prob = model.predict_proba(xnew_pre)
        max_prob = np.max(prob)

        return pred, max_prob

    except Exception as e:
        st.error(f'Error during prediction: {str(e)}')
        return None, None


st.title('Iris Species Predictor')

sep_len=st.number_input('Sepallength',min_value=0.0,step=0.1,value=5.1)

if st.button("predict"):
    scaler_path='notebook/scaler.pkl'