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
import matplotlib.colors

def plot_learning_curves(model,X,y,mx,my):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)

    train_errors,val_errors=[],[]

    for m in range(1,mx):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))

        val_errors.append(mean_squared_error(y_val,y_val_predict))


    plt.plot(np.sqrt(train_errors),"r.",linewidth=2,label="train")

    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="validate")
    plt.xlim([0,mx])
    plt.ylim([0, my])
    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)

    wo=model.intercept_
    w=model.coef_
    w[0]=wo

    return w


rawData=pd.read_csv('iris.csv')
data = rawData.replace(to_replace=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], value=[0,1,2])

print(data)

data = data.to_numpy()

SL = data[:,1]
SW = data[:,2]
PL = data[:,3]
PW = data[:,4]
y = data[:,5]

fig = plt.figure()
colors = ["red", "blue", "green"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)

plt.scatter(SL, SW, c=color_indices, cmap=colormap)
plt.xlabel("sepal Length")
plt.ylabel("sepal Width")
plt.show()

fig = plt.figure()
plt.scatter(PL, PW, c=color_indices, cmap=colormap)
plt.xlabel("pedal Length")
plt.ylabel("pedal Width")
plt.show()

#functions we are going to use
# Softmax for matricies hacked to work
def softmax(z): # z is a matrix
    t1 = np.exp(z.T) # raise it to its transpose    numerator
    t2=np.sum(np.exp(z),axis=1) #denominator
    return (t1/t2).T


# argmax function
# choose the max column based on probability
def to_class(c):
    return c.argmax(axis=1)  # what column in that row is the max


# finds corss entropys foractual observes and the probability
def crossE(o,t):
    # how far off are we on our prediction.
    # how close is this related to this
    # punished bad probabilities or weights and values good ones
    return -np.sum(np.log(o)*(t),axis=1)


def cost(o,t):
    # cross entropies for each point
    return np.mean(crossE(o,t))


# take the pedal information
x = data[:,3:5]
#y is defined above

# normalize based on the variation in the data
# this minimizes the influence of outliers
# based on standard deviation
x[:,0] = (x[:,0] - x[:,0].mean())/x[:,0].std() # to normalize, divide by the standard deviation
x[:,1] = (x[:,1] - x[:,1].mean())/x[:,1].std() # to normalize, divide by the standard deviation
# moved the mean to around zero
#print(x)

# add the column of ones
x = np.c_[np.ones(x.shape[0]),x]

# time to do some one hot code on the output
leny = len(y)
y_ohc = np.empty((leny,3))

for i in range(leny):
    if y[i] == 0:
        y_ohc[i] = [1,0,0]
    elif y[i] == 1:
        y_ohc = [0,1,0]
    else:
        y_ohc = [0,0,1]

#pick a random seed
np.random.seed(24670)

w = np.random.randn(3,3) # has a weight for output, col1, and col2
#print(w)

eta = .001
n = 20000
costy = np.empty(n)

# downhill simplex in 9 dimensions!!

for i in range(n):
    # the softmax of the score S
    prediction = softmax(np.dot(x,w))
    # how close was our prediction
    cc = cost(prediction, y_ohc)
    
    # ten here is the number of random samples 
    # until I get a pat of lower cost
    for j in range(10):
        dw = np.random.randint(3,size=(3,3))-1 # this is a random point in the 9d space

        #print(dw)
        #sys.exit()
        # tw = guessed weights
        tw = w + dw * eta # adjusting current weights in nine dimensional surface
        # if weight decreases its a good step
        prediction = softmax(np.dot(x,tw))
        tc = cost(prediction, y_ohc)
        

        if tc < cc:
            w = tw
            cc = tc
            break

    costy[i] = cc


fig = plt.figure()
plt.plot(costy)
plt.show()
