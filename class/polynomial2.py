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


def plot_learning_curves(model,X,y, my, mx):
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

'''
testData = [[2,3],[5,6],[8,9]]
data = np.array(testData)
poly = PolynomialFeatures(degree = 3)
data = poly.fit_transform(data)
print(data)
'''

# Picked our features by looking at the scatter plot
rawData = pd.read_csv('RealEstate.csv')
print(rawData)
attributes = ["Y", "X2", "X3", "X5", "X6"]
scatter_matrix(rawData[attributes])
plt.show()

# Put features together that we want
data = rawData.to_numpy()
y = data[:,1]
x2 = data[:,3]
x3 = data[:,4]
x5 = data[:,6]
x6 = data[:,7]
x = np.column_stack((x2,x3,x5,x6))
print(x)

# Now we are doing the poynomializing
poly = PolynomialFeatures(degree=20, include_bias=False)
data = poly.fit_transform(x)
print(data.shape[1])

# scale it using minmax
minmax = MinMaxScaler()
x = minmax.fit_transform(data)
# add the column of ones
x = np.c_[np.ones(x.shape[0]), x]
#print(x)
my = 100
mx = len(y) - .2*len(y)

model = LinearRegression()
plot_learning_curves(model,x,y, int(mx), my)

plt.show()

# train to point of convergence and get your weights and you can then take inputs and get an output.
# Then to train and test and find the weights after this
# this is just like we did before
