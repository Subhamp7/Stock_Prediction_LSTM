# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:56:15 2020

@author: subham
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#loading dataset
dataset=pd.read_csv("INFY.NS.csv")

#checking for nan values
print("There are {} Nan values".format(dataset.isnull().sum().sum()))

#column name for missing data 
print(dataset.columns[dataset.isnull().any()])

#visualizing the missing data
sns.heatmap(dataset.isnull(),cbar=False)
#we can see that data is missing for a single row btw row 40-50

#getting the accurate row number
null_row=dataset[dataset.isnull().any(axis=1)]

#removing the row with missing data
dataset.update(dataset.drop([49],inplace=True))

#taking the opening stock price
open_price=dataset.iloc[:,1].values

#plotting the open price
plt.plot(open_price)

#scaling the open price
scaler=MinMaxScaler(feature_range=(0,1))
open_price=scaler.fit_transform(open_price.reshape(-1,1))

plt.plot(open_price)

