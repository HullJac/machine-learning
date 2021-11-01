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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm


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


rawData=pd.read_csv('SvmtestMulti6.csv')

print(rawData)

scatter_matrix(rawData)
plt.show()

data = rawData.to_numpy()

y = data[:,4]
x1 = data[:,0]
x2 = data[:,1]
xdontCare = data[:,2]
x3 = data[:,3]

colors=["red", "green", "blue", "black"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)

fig = plt.figure()
threedee = fig.add_subplot(projection='3d')

threedee.scatter(x1, x2, x3, c=color_indices, cmap=colormap)

plt.show()

x = np.column_stack((x1,x2,x3))

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


linsvm = svm.LinearSVC(C=1) # bigger number here is a softer margin
linsvm.fit(x,y)

x4 = [[3.76,12.9,10.2]] #pick points to test
x1 = [[1.8,21.5,2.33]]
px = scaler.transform(x1) #scale them

#Then run the 80 20 here

print(linsvm.predict(px))

