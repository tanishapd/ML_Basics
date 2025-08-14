#!/usr/bin/env python
# coding: utf-8

# # 1. Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# In[2]:


heart_data=pd.read_csv("heart_disease_data.csv")
heart_data.head(10)


# # 2.Perform Preprocessing on data
# 1. Finding info and describing the dataset
# 2. Check for all null values,if any,remove them or fill them with median/mean. 
# 3. Find correlation between the features.
# 

# In[3]:


heart_data.info()


# In[4]:


heart_data.describe()


# In[5]:


heart_data.isnull().sum()


# In[6]:


import seaborn as sns
plt.figure(figsize=(15,15))
sns.heatmap(heart_data.corr(),annot=True)
plt.show()


# In[7]:


heart_data=heart_data.drop(['fbs','chol'],axis=1)
heart_data.head(10)


# # 3.  Training our model

# In[8]:


heart_data['target'].value_counts()


# In[9]:


X=heart_data.drop(columns=['target'],axis=1)
y=heart_data.target
y


# In[10]:


X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[11]:


X_train


# In[12]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)


# In[13]:


model.fit(X_train,y_train)


# In[14]:


test_pred=model.predict(x_test)
test_pred


# In[15]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[16]:


cm=confusion_matrix(y_true=y_test,y_pred=test_pred)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False,True])
cm_display.plot()
plt.show()


# In[17]:


score=accuracy_score(test_pred,y_test)
print("Accuracy percentage= ",score*100,"%")


# # 4. Saving the model

# In[18]:


joblib.dump(model,"Heart Disease Predictor.joblib")


# In[19]:


load_model=joblib.load("Heart Disease Predictor.joblib")


# In[20]:


load_model


# In[21]:


input_data=[72,2,2,150,0,151,0,2.5,0,0,2]


# In[22]:


input_data_as_numpy=np.asarray(input_data)
input_data_as_numpy.shape


# In[23]:


input_data_as_numpy=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy.reshape(1,-1)
prediction=load_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("Person doesn't have heart disease")
else:
    print("Person have heart disease")


# In[ ]:




