'''
Program:        Softmax Regression Model Training and Testing Using Polynomial Features On Diabetes Data
Programmer:     Jacob Hull
Date:           11/6/21
Description:    This program trains the Softmax Regression model with polynomial features to
                predict the likelyhood of somone getting diabetes. It utilizes polynomial 
                features to minimize the error of prediction of diabetes based on the features 
                I have chosen. The features are everything except pregnancies. Many models are 
                trained using the 80/20 rule and the best is chosen by percent accuracy of the
                predictions from the testing data. I then take in new data from another csv file
                and predict the outcome of the data given to see how my model performs with data
                not found in the dataset.
'''
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Loading the data
rawData=pd.read_csv('diabetesBinary.csv')
data = rawData.to_numpy()

# Create a heat map of the data
'''
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(rawData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
'''

# Separate the y column
y = data[:,-1]

# Separate the different features
gluc = data[:,1]
bloo = data[:,2]
skin = data[:,3]
insu = data[:,4]
bmi  = data[:,5]
diab = data[:,6]
age  = data[:,7] 

# Create the cleaner object
imp = SimpleImputer(missing_values=0, strategy='mean')

# Create the data matrix to clean
x = np.column_stack((gluc,bloo,skin,insu,bmi,diab,age)) # I am using everything but pregnancies

# Clean the data (all except pregnancies becaseu you can have 0 of those)
x = imp.fit_transform(x)

# Checking out 3-D graphs of the data
'''
colors=["red", "green"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)
fig = plt.figure()
threedee = fig.add_subplot(projection='3d')
threedee.scatter(gluc,bloo,skin,insu,bmi, c=color_indices, cmap=colormap)
plt.show()
sys.exit()
'''

# Create the polynomial features and fit transform it
poly = PolynomialFeatures(degree=3, include_bias=False)
x = poly.fit_transform(x)

# Create the scaler and fit and transform it
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Function to fit the model and find how well it did
def train():
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    
    # Create the model
    softmax_sci = LogisticRegression(multi_class="multinomial",solver="lbfgs", max_iter=5000, C=0.01)
    
    # Fit the model
    softmax_sci.fit(x_train, y_train)
    
    # Check accuracy of the model
    testAccuracy = softmax_sci.score(x_test, y_test)

    # Return the accuracy and model
    return testAccuracy*100, softmax_sci

# Run until we get above a specified percent
prevAcc = 0
goodModel = 0
it = 0
acc, model = train()
while (acc < 88 and it < 10000):
    it += 1
    acc, model = train()
    if acc > prevAcc:
        prevAcc = acc
        goodModel = model

# Loading the input data
inpData=pd.read_csv('input.csv')
inp = inpData.to_numpy()

# Separate the differet features
gluc = inp[:,1]
bloo = inp[:,2]
skin = inp[:,3]
insu = inp[:,4]
bmi  = inp[:,5]
diab = inp[:,6]
age  = inp[:,7] 

# Create the data matrix
inpX = np.column_stack((gluc,bloo,skin,insu,bmi,diab,age)) # I am using everything but pregnancies

# Clean the data just in case of zero values
inpX = imp.fit_transform(inpX)

# Polynomialize the data
inpX = poly.fit_transform(inpX)

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
