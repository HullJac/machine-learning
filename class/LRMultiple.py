import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

rawData = pd.read_csv('BP.csv')

# Printing and graphing data first
print(rawData)
attributes = ["BP", "Age", "Weight"]
scatter_matrix(rawData[attributes])
plt.show()

# Data prep
data = rawData.to_numpy()

# split to each column
y = data[:,0]
x1 = data[:,1]
x2 = data[:,2]

# Combine Xs then add the ones
# These are just a place holder for W0 bias
x = np.column_stack((x1,x2))
x = np.c_[np.ones(x.shape[0]), x]
print(x)

# ITs just the same as before, do the same equation

xt = np.transpose(x)

right = np.dot(xt, y)
temp = np.dot(xt, x)
left = np.linalg.inv(temp)

w = np.dot(left, right)

# Right now they are heavier on certain ws because the data is not normal
# So we need to normalize the data and put them on a scale from one to zero
print(w)

# Trying to see what y is 
ytest = [1.0, 51, 305]
print(w.dot(ytest))

# max-min normarlization
# in each feature, find the max and min.
# subtract the min 
# divide by (max - min)
# will get things around 1
# problem here lies when there are outliers

max1 = max(x1)
min1 = min(x1)
x1 = (x1-min1)/(max1-min1)
ytest[1] = (ytest[1]-min1)/(max1-min1)
print(x1)


max2 = max(x2)
min2 = min(x2)
x2 = (x2-min2)/(max2-min2)
ytest[2] = (ytest[2]-min2)/(max2-min2)
print(x2)

# ----------------------------------------------------------#
# Combine Xs then add the ones
# These are just a place holder for W0 bias
x = np.column_stack((x1,x2))
x = np.c_[np.ones(x.shape[0]), x]
print(x)

# ITs just the same as before, do the same equation

xt = np.transpose(x)

right = np.dot(xt, y)
temp = np.dot(xt, x)
left = np.linalg.inv(temp)

w = np.dot(left, right)

# Right now they are heavier on certain ws because the data is not normal
# So we need to normalize the data and put them on a scale from one to zero
print(w)

# From this, we can see that the scaling didnt really help us
print(w.dot(ytest))



# --------------------------------------
from sklearn.preprocessing import MinMaxScaler

x1 = data[:,1]
x2 = data[:,2]
x = np.column_stack((x1,x2))

minmax = MinMaxScaler()
X = minmax.fit_transform(x)
print(X)

X = np.c_[np.ones(X.shape[0]),X]

LR = LinearRegression()

LR.fit(X,y)

w0 = LR.intercept_
w = LR.coef_
w[0] = w0
# this is the coeficients and are the same as before.
print(w)


print(w.dot(ytest))
print("Here--------------------")


# ----------------training and testing

from sklearn.model_selection import train_test_split

train, test = train_test_split(rawData, test_size=.2, random_state=12)

data = train.to_numpy()

# separate data
y = data[:,0]
x1 = data[:,1]
x2 = data[:,2]

# make data matrix
X = np.column_stack((x1,x2))

# normalize the xs
X = minmax.fit_transform(X)
print(X)


X = np.c_[np.ones(X.shape[0]),X]

LR.fit(X,y)

# get weights
w0 = LR.intercept_
w = LR.coef_
w[0] = w0

print(w)

# ---------------prediction set
minn = minmax.data_min_
maxx = minmax.data_max_

testSet = test.to_numpy()

y = testSet[:,0]
x1 = testSet[:,1]
x2 = testSet[:,2]

x1 = (x1-minn[0])/(maxx[0]-minn[0])
x2 = (x2-minn[1])/(maxx[1]-minn[1])

X = np.column_stack((x1,x2))

X = np.c_[np.ones(X.shape[0]), X]

# yhat is the prediction
yhat = X.dot(w)
print("Here--------------------")

#L1 error function
# gives percentage that we are off
result = np.abs((yhat-y)/y*100)
# we want the mean of that though
print(np.mean(result))


### learning curve stuff
from sklearn.utils import resample

data = rawData.to_numpy()
error = np.zeros(len(data)-100)
i = -1
# this will go from 100 to 1000 cause thats how long the data is
for sample in range(100, len(data)):
    Sdata= resample(data, n_samples=sample) #Sdata is the sampled set
    
    train, test = train_test_split(Sdata, test_size=0.2) # no seed, let it be random every time

    y = train[:,0]
    x1 = train[:,1]
    x2 = train[:,2]
    
    X = np.column_stack((x1,x2))
    
    X = minmax.fit_transform(X)
    
    X = np.c_[np.ones(X.shape[0]),X]

    LR.fit(X,y)

    w0 = LR.intercept_
    w = LR.coef_
    w[0] = w0

    # evaluate how we did
    minn = minmax.data_min_
    maxx = minmax.data_max_

    y = test[:,0]
    x1 = test[:,1]
    x2 = test[:,2]
    
    x1 = (x1-minn[0])/(maxx[0]-minn[0])
    x2 = (x2-minn[1])/(maxx[1]-minn[1])
    # need to scale before these
    X = np.column_stack((x1,x2))

    X = minmax.fit_transform(X)
    
    X = np.c_[np.ones(X.shape[0]),X]
    
    # prediction
    yhat = X.dot(w)

    result = yhat - y
    # this is the square
    result = np.dot(result.T, result)
    i+=1
    error[i] = np.sqrt((result)) / sample

fig, graph = plt.subplots()
graph.plot(error, ".")
plt.show()
