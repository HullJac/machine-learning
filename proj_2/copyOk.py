'''
Program:        Linear Regression on Stock Data 
Programmer:     Jacob Hull
Date:           Septemeber 29th 2021
Description:    This program uses the linear regression alorgithm with the normal equation to
                predict the future dow value.
'''

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Setup scaler and linear regession libraries
minmax = MinMaxScaler()
LR = LinearRegression()

# Get the data from the file
rawData = pd.read_csv('clean_data.csv')

# Printing and graphing data first
#print(rawData)
attributes = ["dow", "vix", "ford", "oil", "gold", "corn", "steel"]
scatter_matrix(rawData[attributes])
plt.show()

### Learning curve ###
data = rawData.to_numpy()
error = np.zeros(len(data)-150)
i = -1
# This will go from a starting number to how long the data is
for sample in range(150, len(data)):
    #Sdata is the sampled set
    Sdata = resample(data, n_samples=sample) 
   
    # Split data into testing and training use 80/20 rule
    train, test = train_test_split(Sdata, test_size=0.2) # no seed, let it be random every time

    # Split training into features
    y = train[:,0]
    x1 = train[:,1]
    x2 = train[:,2]
    x3 = train[:,3]
    x4 = train[:,4]
    x5 = train[:,5]
    x6 = train[:,6]
    
    ''' 
    # Scaling using minmax
    x1 = (x1-minn[0])/(maxx[0]-minn[0])
    x2 = (x2-minn[1])/(maxx[1]-minn[1])
    x3 = (x3-minn[2])/(maxx[2]-minn[2])
    x4 = (x4-minn[3])/(maxx[3]-minn[3])
    x5 = (x5-minn[4])/(maxx[4]-minn[4])
    x6 = (x6-minn[5])/(maxx[5]-minn[5])
    '''
    
    # Creating the data matrix "x" which contains all the features
    x = np.column_stack((x1,x2,x3,x4,x5,x6))
    
    x = minmax.fit_transform(x)
    
    x = np.c_[np.ones(x.shape[0]),x]

    LR.fit(x,y)

    # Get weights
    w0 = LR.intercept_
    w = LR.coef_
    w[0] = w0

    '''
    # Get mins and maxs
    minn = minmax.data_min_
    maxx = minmax.data_max_
    '''

    # Split into feature columns
    y = test[:,0]
    x1 = test[:,1]
    x2 = test[:,2]
    x3 = test[:,3]
    x4 = test[:,4]
    x5 = test[:,5]
    x6 = test[:,6]
    
    '''
    # Scaling using minmax
    x1 = (x1-minn[0])/(maxx[0]-minn[0])
    x2 = (x2-minn[1])/(maxx[1]-minn[1])
    x3 = (x3-minn[2])/(maxx[2]-minn[2])
    x4 = (x4-minn[3])/(maxx[3]-minn[3])
    x5 = (x5-minn[4])/(maxx[4]-minn[4])
    x6 = (x6-minn[5])/(maxx[5]-minn[5])
    '''

    x = np.column_stack((x1,x2,x3,x4,x5,x6))
    
    x = minmax.fit_transform(x)
    
    x = np.c_[np.ones(x.shape[0]),x]
    
    # Prediction
    yhat = x.dot(w)
    #print("yhat: {}".format(yhat))

    result = yhat - y
    
    # This is the square
    result = np.dot(result.T, result)
    #print("result: {}".format(np.sqrt(result)))

    # Find the error we are off
    i+=1
    error[i] = np.sqrt((result)) / sample
    #print(error[i])

fig, graph = plt.subplots()
graph.plot(error, ".")
plt.show()

# Testing with some dummy data
ytest = [1, 20, 13, 70, 20, 500, 25]
print("w: {}".format(w))
print("yhat: {}".format(yhat))
print("w.dot(ytest): {} ".format(w.dot(ytest)))
print("result: {} ".format(np.sqrt(result)))

'''  For later when taking inputs
# Inputs
vix = int(input("vix: "))
ford = int(input("ford: "))
oil = int(input("crude oil: "))
gold = int(input("gold: "))
corn = int(input("corn futures: "))
steel = int(input("steel: "))
'''
