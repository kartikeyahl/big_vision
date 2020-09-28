# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')                  #endpoint 1(input of .csv file)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
l=len(X[1,:])

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#taking user input for prediction
lst=[]
for i in range(0, l):                           
    ele = float(input())                          #endpoint 2(taking user input values)
    lst.append(ele)   

# Predicting a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform([lst])))
print(y_pred)                                      #endpoint 3(output)
