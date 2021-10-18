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


def plot_learning_curves(model,X,y):  #, mx, my):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)

    train_errors,val_errors=[],[]

    for m in range(1,len(X_train)): # I changed this from one to 25 and it seemed to do better
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))


    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="validate")

    #plt.ylim([0,my])
    #plt.xlim([0,mx])

    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)
    w0 = model.intercept_
    w = model.coef_
    w[0] = w0

    return w

# Picked our features by looking at the scatter plot
rawData = pd.read_csv('Traffic.csv')
#print(rawData)
#attributes = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]
#scatter_matrix(rawData[attributes])
#plt.show()


# Put features together that we want
#print(rawData)
for i in range(1):
    data = rawData.to_numpy()
    #print(data)
    # Grabbing a subset of data
    data = data[np.random.choice(data.shape[0], 30000, replace=False), :]
    #print(data)
    #print(len(data))
    
    y = data[:,0]   # traffic_volume
    x1 = data[:,1]  # temp
    x2 = data[:,2]  # rain_1h
    x3 = data[:,3]  # snow_1h
    x4 = data[:,4]  # clouds_all
    x = np.column_stack((x1,x2,x3,x4))
    
    # Now we are doing the poynomializing
    poly = PolynomialFeatures(degree=8, include_bias=False) # CHANGE DEGREE HERE
    data = poly.fit_transform(x)
    
    # Scale data using minmax
    minmax = MinMaxScaler()
    x = minmax.fit_transform(data)
    
    # Add the column of ones
    x = np.c_[np.ones(x.shape[0]), x]
    
    # Set up the learning model
    model = Ridge(alpha = .00001) # added two zeros here may be too much
    
    #my = 2500          # tell y where to stop at
    #mx = len(y) - .2*len(y) # Tell x where to stop at
    
    # Get Ws and check how good the learning is going
    w = plot_learning_curves(model,x,y)  #, int(mx), my)
    print(w)
    plt.savefig('1,30000,8,00001'+str(i)+'.png', dpi=300, bbox_inches ='tight')
#plt.show()
# train to point of convergence and get your weights and you can then take inputs and get an output.
# Then to train and test and find the weights after this
# this is just like we did before


'''
temp = input("average temp in kelvin: ")
rain = input("rain in mm for last : ")
snow = input("snow in mm for last hour: ")
cloud = input("cloud cover percentage: ")
'''
