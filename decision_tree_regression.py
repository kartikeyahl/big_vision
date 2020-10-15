# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')    #endpoint 1(input of .csv file for model training)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lst = pd.read_csv('to_predict.csv')    #endpoint 2(input of .csv file for prediction)


# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict(lst)         #endpoint 3(output)
