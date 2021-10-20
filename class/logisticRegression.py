#this is a class template that we will use in class.
#no sense in re-inventing the wheel
#I will add to this as we go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler


def plot_learning_curves(model,X,y):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=10)

    train_errors,val_errors=[],[]

    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))

        val_errors.append(mean_squared_error(y_val,y_val_predict))


    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")

    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="validate")

    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)

    w0 = model.intercept_
    w = model.coef_
    w[0] = w0

# sigmoid function 
# this funciton assigns a probability to the argument
def sigmoid(x): # x stand for anything, not the data matrix
    fn = 1 + np.exp(-x)
    return 1/fn

# this is a self define error function to help us visualize the training 
# this is just how we want to visualize this and see the graph !!!!!!!!!
def errorF(X, y, w):
    z = np.dot(X,w)
    p = sigmoid(z)

    diff = p - y
    j = np.dot(diff, diff)
    
    return np.sqrt(j)

# logistic gradient decent based on our derivation in class
def BGD(X, y, w, a, n): # data matrix, waht we know, weights, alpha, numsteps
    m = len(y)
    cost = np.zeros(n)

    for i in range(n):
        dff = sigmoid(np.dot(X,w)) - y
        w = w - a * 1/m * np.dot(dff.T, X)
        cost[i] = errorF(X,y,w)
    
    return w, cost

rawData = pd.read_csv('socialnetwork.csv')
print(rawData)

data = rawData.to_numpy()

y = data[:,4]
x = data[:,2:4]

minmax = MinMaxScaler()
x = minmax.fit_transform(x)
x = np.c_[np.ones(x.shape[0]), x]

dim = x.shape[1]
np.random.seed(200) # take seed out to actually see how good you are doing
w = np.random.randn(dim) * 10

alpha = 0.1
n = 17000  # change this to stop earlier
w, cost = BGD(x,y,w,alpha,n)

print(w)

fig,graph = plt.subplots()
graph.plot(cost)
plt.show()

# plot acual data and look at it
import matplotlib.colors

fig = plt.figure()
colors = ["red", "blue"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x[:,1], x[:,2], c=color_indices, cmap=colormap)

# putting line in graph
x0 = np.zeros(40)
x1 = np.zeros(40)
pair= np.empty((1600, 3), float)
delx0 = 1/40 # delta x
delx1 = 1/40 # delta y
for i in range(40):
    x0[i] = i*delx0 # filling points to wehre thye are on the map
    x1[i] = i*delx1

k=0
for i in range(40):
    for j in range(40):
        pair[k][0] = 1
        pair[k][1] = x0[i]
        pair[k][2] = x1[j]
        k += 1


yhat = sigmoid(pair.dot(w))
plt.scatter(pair[:,1], pair[:,2],c=yhat, cmap=colormap, alpha=0.2)

plt.show()

# Then see how well these weights did
