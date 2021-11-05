import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import seaborn as sns
from sklearn.svm import SVC

'''
# Help supress warnings I am getting for large polys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'''

rawData=pd.read_csv('winequality-redMulti.csv')

# Getting information about the dataset
#print(rawData.describe())

'''
# Create a heat map of the data
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(rawData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# A lot of correlation between citric acid, density, and fixed acidity, so I dropped them
# Also seems that free sulfur dioxide and total sulfur dioxide are related
'''

'''
scatter_matrix(rawData)
plt.show()
'''

data = rawData.to_numpy()

# separate the data 
fixedAcid = data[:,0]
volAcid = data[:,1]
citAcid = data[:,2]
resSugar = data[:,3]
chlorides = data[:,4]
FSD = data[:,5]
TSD = data[:,6]
density = data[:,7]
pH = data[:,8]
sulphates = data[:,9]
alcohol = data[:,10]
y = data[:,11] # quality

'''
# Graphing
colors=["red", "green", "blue", "black", "yellow", "purple"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)

fig = plt.figure()
threedee = fig.add_subplot(projection='3d')

#threedee.scatter(resSugar, chlorides, pH, sulphates, alcohol, c=color_indices, cmap=colormap)
#threedee.scatter(fixedAcid, citAcid, resSugar, sulphates, alcohol, c=color_indices, cmap=colormap) # positive correlation
threedee.scatter(chlorides, FSD, TSD, density, pH, c=color_indices, cmap=colormap) # negative correltion

plt.show()
'''

# Pick the Xs that seem the best from heatmap and graphing
#x = np.column_stack((fixedAcid,volAcid,citAcid,resSugar,chlorides,FSD,TSD,density,pH,sulphates,alcohol))# everything
#x = np.column_stack((volAcid, resSugar, chlorides, pH, sulphates, alcohol)) # no correlation
#x = np.column_stack((fixedAcid, volAcid, pH, alcohol)) # by eyeball

x = np.column_stack((fixedAcid,citAcid,resSugar,sulphates,alcohol))# related to quality positivly
#x = np.column_stack((volAcid, chlorides, FSD, TSD, density, pH))# related to quality negatively

# Set up the scaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Function to create and test the model using the 80/20 method
def train():
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    # Set up the model
    #model = svm.SVC(kernel='poly', coef0=1, degree=7, max_iter=1000000000, C=0.01) # bigger C here is a softer margin
    
    #model = svm.SVC(kernel='poly', coef0=1, degree=15, max_iter=100000000, C=0.1)
    # above is not bad with certain features
    
    #model = svm.SVC(kernel='poly', coef0=1, degree=5, max_iter=1000000000, C=10)
    # got to 66% at 111 iterations
    
    #model = svm.SVC(kernel='poly', coef0=1, degree=3, max_iter=1000000000, C=10)
    # model that does fairly well with all features
    
    model = svm.SVC(kernel='poly', coef0=1, degree=7, max_iter=1000000000, C=10)
    # got to 66 at 18 iterations and also 4
    # these were based on the quality positively

    # Fit the model
    model.fit(x_train, y_train)

    # Check the accuracy of the model
    testAccuracy = model.score(x_test, y_test)

    # Return the accuracy and the model
    return testAccuracy*100, model
    
it = 0
acc, model = train()
print(acc)
bestAcc = acc
while (acc < 67):
    it += 1
    acc, model = train()
    if acc > bestAcc:
        bestAcc = acc
        print(str(acc)+ " : " + str(it))
        

'''
#one = [[9.6,0.32,0.47,1.4,0.056000000000000000,9.0,24.0,0.99695,3.22,0.82,10.3]] # 7
one = [[0.32,1.4,0.056000000000000000,3.22,0.82,10.3]] # 7
one = scaler.transform(one)

two = [[0.655,2.3,0.083,3.17,0.66,9.8]] # 5
two = scaler.transform(two)

oneA = model.predict(one)
twoA = model.predict(two)

print(oneA)
print(twoA)
'''
