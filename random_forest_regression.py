# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')          #endpoint 1(input of .csv file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

lst = pd.read_csv('to_predict.csv')    #endpoint 2(input of .csv file for prediction)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)            

# Predicting a new result
regressor.predict(lst)                   #endpoint 3(output)
                                  

