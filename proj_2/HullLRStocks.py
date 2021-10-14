'''
Program:        Linear Regression on Stock Data Predicting the DOW 
Programmer:     Jacob Hull
Date:           Septemeber 29th 2021
Description:    This program uses the sklearn linear regression alorgithm with the normal equation to
                predict the future dow value after traning. The prediction is based on the inputs
                given by the user.
'''

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Setup minmax scaler and linear regession libraries
minmax = MinMaxScaler()
LR = LinearRegression()

# Get the data from the file
rawData = pd.read_csv('clean_data.csv')

# Printing and graphing data first
#print(rawData)
attributes = ["dow", "vix", "ford", "oil", "gold", "corn", "steel"]
scatter_matrix(rawData[attributes])
plt.show()

### Learning Curve ###
data = rawData.to_numpy()
error = np.zeros(len(data)-150)
i = -1
# This will go from a starting number to how long the data is
for sample in range(150, len(data)):
    #Sdata is the sampled set
    Sdata = resample(data, n_samples=sample) 

    # Normalize the data except the y column
    Sdata[:,1:] = minmax.fit_transform(Sdata[:,1:])
    
    # Split data into testing and training use 80/20 rule
    train, test = train_test_split(Sdata, test_size=0.2) # no seed, let it be random every time
   
    ### Training ###

    # Split training into features
    y = train[:,0]
    x1 = train[:,1]
    x2 = train[:,2]
    x3 = train[:,3]
    x4 = train[:,4]
    x5 = train[:,5]
    x6 = train[:,6]
    
    # Creating the data matrix "x" which contains all the features for training
    x = np.column_stack((x1,x2,x3,x4,x5,x6))
    
    x = np.c_[np.ones(x.shape[0]),x]

    LR.fit(x,y)

    # Get weights
    w0 = LR.intercept_
    w = LR.coef_
    w[0] = w0

    ### Testing ###

    # Split test into features
    y = test[:,0]
    x1 = test[:,1]
    x2 = test[:,2]
    x3 = test[:,3]
    x4 = test[:,4]
    x5 = test[:,5]
    x6 = test[:,6]
   
    # Combine the test xs into a matrix
    x = np.column_stack((x1,x2,x3,x4,x5,x6))
    
    # Add a column of ones
    x = np.c_[np.ones(x.shape[0]),x]
    
    # Prediction
    yhat = x.dot(w)

    # Error finding 
    result = yhat - y
    
    # This is the square
    result = np.dot(result.T, result)

    # Find the error we are off
    i+=1
    error[i] = np.sqrt((result)) / sample

# Graph the learning curve
fig, graph = plt.subplots()
graph.plot(error, ".")
plt.show()

# Inputs from zero
vix = float(input("vix closing: "))
ford = float(input("ford closing: "))
oil = float(input("crude oil closing: "))
gold = float(input("gold closing: "))
corn = float(input("corn futures closing: "))
steel = float(input("steel closing: "))

# Creating the list and transforming it
closings = [[vix, ford, oil, gold, corn, steel]]
closings = minmax.transform(closings)
closings = np.c_[np.ones(closings.shape[0]), closings]
yhat = closings.dot(w)
print("The predicted DOW value is: {}".format(yhat))
