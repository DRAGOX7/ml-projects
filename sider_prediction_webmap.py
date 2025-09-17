# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 18:21:31 2025

@author: aljaf
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/aljaf/Downloads/modeldeploy/model_deployment.sav', 'rb'))

# creating a function for prediction
def diabetes_prediction(input_data):
    # change input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # giving a title
    st.title("Diabetes Prediction Web App")

    # getting the input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the person")

    # code for prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        try:
            input_values = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            st.error("⚠️ Please enter valid numerical values!")

    st.success(diagnosis)

if __name__ == '__main__':
    main()
