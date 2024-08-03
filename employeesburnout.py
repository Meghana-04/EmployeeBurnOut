# -*- coding: utf-8 -*-
"""EmployeesBurnOut.ipynb

Original file is located at
    https://colab.research.google.com/drive/16JWsH5NkkATFtrkIhtIobHkbGsdZNeVs

'Employees Burn Out' 
This project is a Python-based data analysis and machine learning initiative aimed at predicting employee burnout rates. 
Utilizing the Pandas library for data manipulation and visualization, the project conducts exploratory data analysis (EDA) and feature engineering. 
A Linear Regression model is then built using Scikit-learn, incorporating AI/ML techniques to evaluate and predict burnout trends.
"""

import pandas as pd           #pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pickle
import numpy as np
import os
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

drive.mount('/content/drive')

"""
Get the dataset from
https://drive.google.com/file/d/1FqBxmcoN1zBeTIWQ-yNTbfPSdq8abJCc/view?usp=drive_link
"""

data=pd.read_csv('/content/drive/MyDrive/employee_burnout_analysis-AI.csv')

data.info()

data.nunique()

data.isnull().sum()

data.isnull().sum().values.sum()

data.corr(numeric_only=True)['Burn Rate'][:-1]

sns.pairplot(data)
plt.show()

data.shape

data = data.dropna()

data.shape

data.dtypes

data_obj = data.select_dtypes(object)
# prints a dictionary of max 10 unique values for each non-numeric column
print({ c : data_obj[c].unique()[:10] for c in data_obj.columns})

data = data.drop('Employee ID', axis = 1)

print(f"Min date {data['Date of Joining'].min()}")
print(f"Max date {data['Date of Joining'].max()}")
data_month = data.copy()

data_month["Date of Joining"] = data_month['Date of Joining'].astype("datetime64[ns]")
data_month["Date of Joining"].groupby(
    data_month['Date of Joining'].dt.month
).count().plot(kind="bar", xlabel='Month', ylabel="Hired employees")

data_2008 = pd.to_datetime(["2008-01-01"]*len(data))
data["Days"] = data['Date of Joining'].astype("datetime64[ns]").sub(data_2008).dt.days
data.Days

data.corr(numeric_only=True)['Burn Rate'][:]

data = data.drop(['Date of Joining','Days'], axis = 1)

data.head()

cat_columns = data.select_dtypes(object).columns
fig, ax = plt.subplots(nrows=1, ncols=len(cat_columns), sharey=True, figsize=(10, 5))
for i, c in enumerate(cat_columns):
    sns.countplot(x=c, data=data, ax=ax[i])
plt.show()

data=pd.get_dummies(data,columns=['Company Type','WFH Setup Available','Gender'],drop_first=True) #one-Hot Encoding for categorical features
data.head()
encoded_columns=data.columns

y=data['Burn Rate']
x=data.drop(['Burn Rate'],axis=1)

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7,shuffle=True,random_state=1) #train-test split
#scale x
scaler=StandardScaler()
scaler.fit(x_train)
x_train=pd.DataFrame(scaler.transform(x_train),index=x_train.index,columns=x_train.columns)
x_test=pd.DataFrame(scaler.transform(x_test),index=x_test.index,columns=x_test.columns)

scaler_filename='../models/scaler.pkl'
#create the model directory if it doesn't exist
os.makedirs(os.path.dirname(scaler_filename),exist_ok=True)
#use pickle to save the scaler to the file
with open(scaler_filename,'wb') as scaler_file:
  pickle.dump(scaler,scaler_file)

x_train

y_train

#saving the processed data
path='../models/processed_data.pkl'
#create the directory if it doesn't exist
os.makedirs(os.path.dirname(path),exist_ok=True)

x_train.to_csv(path+'x_train_prosessed.csv',index=False)
y_train.to_csv(path+'y_train_processed.csv',index=False)

#create an instance of the LinearRegression
linear_regression_model=LinearRegression()
#train the model
linear_regression_model.fit(x_train,y_train)

#linear Regression Model performance metrices
print("Linear Regression Model performance metrices:\n")
#make prediction on test set
y_pred=linear_regression_model.predict(x_test)
#calculate mean square error
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
#calculate root mean squared error
rmse=mean_squared_error(y_test,y_pred,squared=False)
print("Root Mean Squared Error:",rmse)
#calculate mean absolute error
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error:",mae)
#calculate r.squared score
r2=r2_score(y_test,y_pred)
print("R-squared Score:",r2)

