import numpy as np
import joblib
import streamlit as st

load_model=joblib.load("C:/Users/KIIT/Heart Disease Predictor.joblib")

def heartdisease_prediction(input_data):

    input_data_as_numpy=np.asarray(input_data)
    input_data_as_numpy=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy.reshape(1,-1)
    prediction=load_model.predict(input_data_reshaped)
    if(prediction[0]==0):
        return "Person doesn't have heart disease"
    else:
        return "Person have heart disease"



def main():
    
    st.title("Heart Disease Prediction Web App")

    age = st.text_input("Age: ")
    sex = st.text_input("Sex: ")
    cp = st.text_input("Chest Pain: ")
    trestbps = st.text_input("Resting blood pressure: ")
    restecg = st.text_input("Resting electrocardiographic measurement: ")
    thalach	= st.text_input("Maximum heart rate achieved: ")
    exang	= st.text_input("Exercise induced angina: ")
    oldpeak	= st.text_input("Oldpeak: ")
    slope	= st.text_input("Slope: ")
    ca	= st.text_input("Calcium: ")
    thal = st.text_input("Thalassemia: ")
        
    detection = ''
        
    if st.button('Heart Disease Test Result'):
        detection = heartdisease_prediction([age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope, ca, thal])
            
    st.success(detection)
        
        
        
        
if __name__ == '__main__':
    main()
            
    
