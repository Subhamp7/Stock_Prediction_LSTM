# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 18:00:55 2020

@author: subham
"""
#importing libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error

#loading dataset
dataset=pd.read_csv("infy_stock.csv")

#checking for nan values
print("There are {} Nan values".format(dataset.isnull().sum().sum()))

#column name for missing data 
print(dataset.columns[dataset.isnull().any()])

#getting the accurate row number
null_row=dataset[dataset.isnull().any(axis=1)].index

#removing the row with missing data
dataset.update(dataset.drop(null_row,inplace=True))

#taking the opening stock price
open_price=dataset.iloc[:,2].values

#plotting the open price
plt.plot(open_price)

#scaling the open price
scaler=StandardScaler()
open_price=scaler.fit_transform(open_price.reshape(-1,1))

#plotting the open price to compare data pattern
plt.plot(open_price)

#splitting the data into training and test data, 0,25 by default
def split(data_set,size=0.25):
    length=len(data_set)
    train_size=int(length*size)
    return data_set[:train_size,:],data_set[train_size:,:]
    
train_data, test_data = split(open_price, 0.60)

#splitting data into dependent and independent value
def splitxy(data_set,interval=10):
    X_data,Y_data=[],[]
    for index in range(len(data_set)-interval-1):
        temp_Y=index + interval
        temp_X=data_set[index:temp_Y,0]
        X_data.append(temp_X)
        Y_data.append(data_set[temp_Y,0])
    return np.array(X_data), np.array(Y_data)
    
X_train, Y_train = splitxy(train_data, 200)
X_test , Y_test  = splitxy(test_data,  200)

#reshaping the data required for LSTM model
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test  =X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#creating stacked LSTM model
model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(200,1)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer='adam')

#printing model details
model.summary()

#fitting the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=100, batch_size=64, verbose=1)

#predict from model
train_pred =model.predict(X_train)
test_pred  =model.predict(X_test)

#reverse scaling
train_pred =scaler.inverse_transform(train_pred)
test_pred  =scaler.inverse_transform(test_pred)

#checking the error
print("For Train data",math.sqrt(mean_squared_error(Y_train,train_pred)))
print("For Test data",math.sqrt(mean_squared_error(Y_test,test_pred)))



plt.plot(Y_test, color='blue', label='Real_stock_price')
plt.plot(test_pred, color='red', label='Predicted_stock_price')
plt.title('Infy Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.plot(Y_train, color='blue', label='Real_stock_price')
plt.plot(train_pred, color='red', label='Predicted_stock_price')
plt.title('Infy Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()