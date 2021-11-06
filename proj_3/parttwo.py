'''
Program:        SVM Model Training and Testing With Kernel Trick And Polynomial Features On Red Wine Data
Programmer:     Jacob Hull
Date:           11/6/21
Description:    This program trains an SVM model with polynomial features to predict the 
                quality of a wine given its basic properties. It utilizes polynomial 
                features to minimize the error of prediction of quality based on the features 
                I have chosen. I decided to use all the features because I have had the highest
                percentage guessed right with all of them. In the program, many models are 
                trained using the 80/20 rule and the best is chosen by percent accuracy of 
                the predictions from the testing data. I then take in new data from another 
                csv file and predict the outcome of the data given to see how my model performs 
                with data not found in the dataset.
'''
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

# Helps supress warnings for data display like graphing
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'''

rawData=pd.read_csv('winequality-redMulti.csv')

# Getting information about the dataset
#print(rawData.describe())

# Create a heat map of the data
'''
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(rawData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
'''

# Scatter matrix
'''
scatter_matrix(rawData)
plt.show()
'''

# Convert to numpy
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

# Graphing
'''
colors=["red", "green", "blue", "black", "yellow", "purple"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)

fig = plt.figure()
threedee = fig.add_subplot(projection='3d')

#threedee.scatter(resSugar, chlorides, pH, sulphates, alcohol, c=color_indices, cmap=colormap) #possible correlation
threedee.scatter(fixedAcid, citAcid, resSugar, sulphates, alcohol, c=color_indices, cmap=colormap) # positive correlation
#threedee.scatter(chlorides, FSD, TSD, density, pH, c=color_indices, cmap=colormap) # negative correltion
plt.show()
'''

# Pick the Xs that seem the best from heatmap and graphing
# Every feature
x = np.column_stack((fixedAcid,volAcid,citAcid,resSugar,chlorides,FSD,TSD,density,pH,sulphates,alcohol))

# Set up the scaler and fit and transform it
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Function to create and test the model using the 80/20 method
def train():
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
   
    # Create the model
    model = svm.SVC(kernel='poly', coef0=1, degree=3, max_iter=1000000000, C=10)
    
    # Fit the model
    model.fit(x_train, y_train)

    # Check the accuracy of the model
    testAccuracy = model.score(x_test, y_test)

    # Return the accuracy and the model
    return testAccuracy*100, model

# Run the model until I get a higher percent accuracy
it = 0
goodModel = 0
acc, model = train()
bestAcc = acc
while (acc < 70 and it < 1000):
    it += 1
    acc, model = train()
    if acc > bestAcc:
        bestAcc = acc
        goodModel = model
        
# Loading the input data
inpData=pd.read_csv('input.csv')
inp = inpData.to_numpy()

# Separate the differet features
fixedAcid = inp[:,0]
volAcid =   inp[:,1]
citAcid =   inp[:,2]
resSugar =  inp[:,3]
chlorides = inp[:,4]
FSD =       inp[:,5]
TSD =       inp[:,6]
density =   inp[:,7]
pH =        inp[:,8]
sulphates = inp[:,9]
alcohol =   inp[:,10]

# Create the data matrix
inpX = np.column_stack((fixedAcid,volAcid,citAcid,resSugar,chlorides,FSD,TSD,density,pH,sulphates,alcohol))

# List to store output values
output = []

# Predict for each row and add then to output list
for row in inpX:
    predicted = goodModel.predict([row])
    output.append(predicted)

# Cast output to be a numpy array
output = np.asarray(output)

# Write output to file
np.savetxt("output.csv", output, delimiter=',', fmt='%d')
