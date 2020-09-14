# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')     #endpoint 1(input of .csv file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
l=len(X[1,:])

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#taking user input for prediction
lst=[]
for i in range(0, l):                           
    ele = float(input("i/p :"))                    #endpoint 2(taking user input values)
    lst.append(ele)                       

# Predicting a new result
regressor.predict([lst])                           #endpoint 2(output)

