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
dataR = rawData.replace(to_replace=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], value=[0,1,2])

print(dataR)

data = dataR.to_numpy()

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
x = copy.deepcopy(data[:,3:5])
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

x0set = x[:,1]
x1set = x[:,2]

x0min = np.amin(x0set)
x0max = np.amax(x0set)

x1min = np.amin(x1set)
x1max = np.amax(x1set)

delx0 = (x0max - x0min)/40.
delx1 = (x1max - x1min)/40.

# create an array of each so we can get a serries of samples
x0 = np.zeros(40)
x1 = np.zeros(40)

pair = np.empty((1600,3), float)

for i in range(40):
    x0[i] = x0min + i * delx0
    x1[i] = x1min + i * delx1

k = 0
for i in range(40):
    for j in range(40):
        pair[k][0] = 1
        pair[k][1] = x0[i]
        pair[k][2] = x1[j]
        k += 1


s = softmax(pair.dot(w))

yguess = to_class(s)

fig = plt.figure()
plt.scatter(pair[:,1], pair[:,2], c=yguess, cmap=colormap, alpha = 0.2)
plt.show()

# Normalize pedal data
PL[:] = (PL[:]-PL[:].mean())/PL[:].std()
PW[:] = (PW[:]-PW[:].mean())/PW[:].std()

plt.scatter(PL, PW, c=color_indices, cmap = colormap)
plt.show()


#############################################
# it is sciklearn brain on neutral time
# this is doing a whoel lot fo things
# we are doing a softmax linear regression

data = dataR.to_numpy()
x = copy.deepcopy(data[:,3:5])

# we will use the industry standard scaler normalization
# subtract mean and divide by unit variate deviation // only slightly differnt as it is normalized to 1
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# first normalize
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# do not add a column of ones for this because they do it internally for us

softmax_sci = LogisticRegression(multi_class="multinomial", solver="lbfgs",C=10) # well come back to this
# C here is like the alpha

softmax_sci.fit(x,y)

# try to test it out
xx = [[5,1]] # this is the value we want to predict

px = scaler.transform(xx)

print(softmax_sci.predict(px))
print(softmax_sci.predict_proba(px))

# Graphing
x0set = x[:,0]
x1set = x[:,1]

x0min = np.amin(x0set)
x0max = np.amax(x0set)

x1min = np.amin(x1set)
x1max = np.amax(x1set)

delx0 = (x0max - x0min)/40.
delx1 = (x1max - x1min)/40.

# create an array of each so we can get a serries of samples
x0 = np.zeros(40)
x1 = np.zeros(40)

pair = np.empty((1600,2), float)

for i in range(40):
    x0[i] = x0min + i * delx0
    x1[i] = x1min + i * delx1

k = 0
for i in range(40):
    for j in range(40):
        pair[k][0] = x0[i]
        pair[k][1] = x1[j]
        k += 1

yguess = softmax_sci.predict(pair)

fig = plt.figure()
plt.scatter(pair[:,0], pair[:,1], c=yguess, cmap=colormap, alpha = 0.2)
plt.show()
sys.exit()

# Normalize pedal data
PL[:] = (PL[:]-PL[:].mean())/PL[:].std()
PW[:] = (PW[:]-PW[:].mean())/PW[:].std()

plt.scatter(PL, PW, c=color_indices, cmap = colormap)
plt.show()
