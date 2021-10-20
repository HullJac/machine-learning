# stuff with just a # at the end are from stochastic graident descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
t0 = 1.0 #
t1 = 50000 #

# this function gets the error for us
def L2(X, y, W):
    prediction = np.dot(X,W)

    ## difference of the observed
    v1 = y-prediction
    error = np.dot(v1.T, v1)

    return np.sqrt(error)

'''
X = 
W = weights
a = learning rate
n = max number of iterations
'''

def learningSchedule(t): #
    global t0,t1
    return t0/(t1+t)


def stochasticGradientDescent(X,y,W,n): #
    costHistory = np.zeros(n)
    m = len(y)
    for i in range(n):
        for j in range(m): # actual stochastic meandering around until we find something
            index = np.random.randint(m)
            xj = X[index,:]
            yj = y[index]

            prediction = np.dot(xj,W)
            t = yj - prediction
            tt = np.transpose(t)
            alpha = learningSchedule(i*m+j) #bigger the number the faster it goes down
            W = W+alpha*np.dot(tt,xj)

            costHistory[i] = L2(X,y,W)
    return W, costHistory

def batchGradientDecent(X,y,W,a,n):
    costHistory = np.zeros(n)

    sz = W.shape
    for i in range(n):
        prediction = np.dot(X,W)
        t = y-prediction # y - yhat
        tt = np.transpose(t) # transpose y - yhat
        
        # final equation this should get us closer
        W = W + a*np.dot(tt,X)
        cc = L2(X,y,W)
        if cc < 1000:
            costHistory[i] = cc
        
    return W, costHistory


rawData = pd.read_csv('BP.csv')

data = rawData.to_numpy()

y = data[:,0]
# gets the shape of whats left
dimension = rawData.shape[1]

X = data[:,1:dimension]

# Normalize here, but werenot going to here becasue this specific data does not need it
###
###
###

# then add column of ones
X = np.c_[np.ones(X.shape[0]),X]

W = np.random.randn(dimension)*10

#print(W)

'''
#alpha = 0.0000001/2/2           # learning rate  can do successive divide by two to get away from nan
#N = 10000000*3    # num steps
N = 1000

#W, cost  = batchGradientDecent(X,y,W,alpha,N)

searches = 15
minny = 10000000
wresult = []
for i in range(searches):
    W,cost = stochasticGradientDescent(X,y,W,N) # this is just one,we need to do bunch and take the smallest
    costFunction = cost[N-1]
    if costFunction < minny:
        wresult = W
        minny = costFunction


print(wresult)
#print(W)
#fig,graph = plt.subplots()
#graph.plot(cost)
#plt.show()

# goal of this is to find the weights that minimize the function
# know you have the right answer when you get the same y a lot
# look at the cost curve or learning curve and mark down value that it converged at
# take the lowest number of those 
'''

###------------------------Polynomial regression----------#
import copy
rawData = pd.read_csv('poly.csv')

data = rawData.to_numpy()

y = data[:,0]

dim = rawData.shape[1]

X = data[:,1:dim]

xx = copy.deepcopy(X) #xx contains one coulmn of data will be more if we have more data

p = 5

np.random.seed(10)

# create data matrix with column up to p = 2
for i in range(p-1):
    x_i = xx**(2+i)
    X = np.c_[X,x_i]

X = np.c_[np.ones(X.shape[0]),X]
dim = X.shape[1]
W = np.random.randn(dim)*10


alpha = 0.000001/2/2
N = 1000000*3

W, cost  = batchGradientDecent(X,y,W,alpha,N)


print(W)
#print(X)

