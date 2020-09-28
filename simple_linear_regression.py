# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')                   #endpoint 1(input of .csv file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
l=len(X[1,:])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#taking user input for prediction
lst=[]
for i in range(0, l):                           
    ele = float(input())                          #endpoint 2(taking user input values)
    lst.append(ele)   

# Predicting the Test set results
y_pred = regressor.predict([lst])
print(y_pred)                                      #endpoint 3(output)
