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
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def plot_learning_curves(model,X,y, mx, my):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)

    train_errors,val_errors=[],[]

    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))


    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="validate")

    plt.ylim([0,my])
    plt.xlim([0,mx])

    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)
    w0 = model.intercept_
    w = model.coef_
    w[0] = w0

    return w

# Picked our features by looking at the scatter plot
rawData = pd.read_csv('regular.csv')
#print(rawData)
#attributes = ["Y", "X2", "X3", "X5", "X6"]
#scatter_matrix(rawData[attributes])
#plt.show()

# Put features together that we want
data = rawData.to_numpy()
y = data[:,0]
#dim = rawData.shape[1]
x = data[:,1:]
#print(x)

# Now we are doing the poynomializing
poly = PolynomialFeatures(degree=20, include_bias=False)
data = poly.fit_transform(x)
print(data.shape[1])

# scale it using minmax
minmax = MinMaxScaler()
x = minmax.fit_transform(data)
# add the column of ones
x = np.c_[np.ones(x.shape[0]), x]

model = LinearRegression()
model = Ridge(alpha = 0.25)
#model = Lasso(alpha = .001)

my = 2          # tell y where to stop at
mx = len(y) - .2*len(y) # Tell x where to stop at

w = plot_learning_curves(model,x,y, int(mx), my)
print(w)
plt.show()
# train to point of convergence and get your weights and you can then take inputs and get an output.
# Then to train and test and find the weights after this
# this is just like we did before
