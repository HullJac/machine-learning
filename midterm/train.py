'''
Program:        Ridge Regression Model Training Using Polynomial Features on Traffic Data
Programmer:     Jacob Hull
Date:           10/19/21
Description:    This Program trains the Ridge Regression model to predict traffic flow 
                westbound on interstate 94 in Minneapolis Minnessota. It utilizes polynomial 
                features to minimize its error of prediction of traffic flow based on the 
                four features. The four features are temperature in kelvin, rain accumulation 
                within the last hour, snow accumulation within the last hour, and percentage 
                of cloud cover from the last hour. This model is trained using the 80/20 rule
                and the error is found using the root mean squared error method. 
                From the training we get our weight matrix, completed model, and scaler.
                I then save those to files to be used later to predict user inputs and 
                generate an output based on the trained model.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from pickle import dump

# Function that trains and validates the model using the 80/20 rule
# It returns the wights found from the training
def plot_learning_curves(model,X,y):
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

    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)
    w0 = model.intercept_
    w = model.coef_
    w[0] = w0
    
    return w

# Grabbing the data from the csv file I created of the column I would like to use
rawData = pd.read_csv('Traffic.csv')
attributes = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]

# Creating a scatter matrix of the raw data and displaying it
scatter_matrix(rawData[attributes])
plt.show()

# Turn data to a numpy array
data = rawData.to_numpy()
#print(data)

# Grabbing a subset of data at random to help with runtime and training
# Changing the second parameter will change the size of the data sampled
#data = data[np.random.choice(data.shape[0], 10000, replace=False), :]
#print(data)
#print(len(data))

# Separate the data columns 
y = data[:,0]   # traffic_volume
x1 = data[:,1]  # temp
x2 = data[:,2]  # rain_1h
x3 = data[:,3]  # snow_1h
x4 = data[:,4]  # clouds_all

# Create the data matrix
x = np.column_stack((x1,x2,x3,x4))

# Now we are doing the poynomializing
poly = PolynomialFeatures(degree=7, include_bias=False)
data = poly.fit_transform(x)

# Scale data using minmax
minmax = MinMaxScaler()
x = minmax.fit_transform(data)

# Add the column of ones
x = np.c_[np.ones(x.shape[0]), x]

# Set up the learning model
model = Ridge(alpha = 0.01) 

# Get Ws and check how good the learning is going
w = plot_learning_curves(model,x,y)

# Write the weights to a file to be read back later
with open('w.txt', 'w') as f:
    np.savetxt(f, w)

# Saving the scaler and model for later so we don't have to train again
dump(minmax, open('minmax.pkl', 'wb'))
dump(poly, open('poly.pkl', 'wb'))

# Show the graph from the training and testing
plt.show()


# Maybe export the model and then use it to do .predict on the weights and inputs
# Do this insted of returning the weights and stuff
