# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')               #endpoint 1(input of .csv file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
l=len(X[1,:])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#taking user input for prediction
lst=[]
for i in range(0, l):                           
    ele = float(input())                          #endpoint 2(taking user input values)
    lst.append(ele) 

# Predicting a new result
y_pred= classifier.predict([lst]))
print(y_pred)                                      #endpoint 3(output)
