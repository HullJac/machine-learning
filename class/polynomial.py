import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# x is data matrix
# model is the form we think it has 
def plot_learning_curves(model, x, y, mx, my):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_errors, val_errors = [],[]
    for m in range(1, len(x_train)): # Change the staring point here to play around
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_val)  # This is the yhat(s)

        # Finding errors for training and validation
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "validation")
    plt.ylim = ([0,my])
    plt.xlim = ([0,mx])
    plt.legend(loc = "upper right", fontsize = 14)
    plt.xlabel("Traning set size", fontsize = 14)
    plt.ylabel("RMSE", fontsize = 14)


rawData = pd.read_csv("poly.csv")
data = rawData.to_numpy()

y = data[:,0]

dim = rawData.shape[1] # ? ()

x = data[:,1:dim]

# 1 here is linear
p = 2
polyFeatures = PolynomialFeatures(degree = p) # without bias tag, it adds ones for you

xp = polyFeatures.fit_transform(x)

model = LinearRegression()
my = 10
mx = 

plot_learning_curves(model, xp, y, mx, my)
plt.show()


# Now just grab the weights and take input from the user to predict the output
