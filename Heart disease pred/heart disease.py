import numpy as np
import joblib

load_model=joblib.load("C:/Users/KIIT/Heart Disease Predictor.joblib")

input_data=[72,2,2,150,0,151,0,2.5,0,0,2]

input_data_as_numpy=np.asarray(input_data)
input_data_as_numpy.shape

input_data_as_numpy=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy.reshape(1,-1)
prediction=load_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("Person doesn't have heart disease")
else:
    print("Person have heart disease")


