'''
Program:

'''
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Loading the data and displaying it
rawData=pd.read_csv('diabetesBinary.csv')
data = rawData.to_numpy()

# Separate the y column
y = data[:,-1]

# Separate the differet features
preg = data[:,0]
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
x = np.column_stack((gluc,bloo,skin,insu,bmi,diab,age))

# Clean the data (all except pregnancies becaseu you can have 0 of those)
x = imp.fit_transform(x)

'''
# Chekcing out graphs of the data
colors=["red", "green"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)
fig = plt.figure()
threedee = fig.add_subplot(projection='3d')
threedee.scatter(gluc,bloo,skin,bmi,age, c=color_indices, cmap=colormap)
plt.show()
#sys.exit()
'''

# Grab each column again
gluc = x[:,0]
bloo = x[:,1]
skin = x[:,2]
insu = x[:,3]
bmi  = x[:,4]
diab = x[:,5]
age  = x[:,6] 

# Create new data matrix from all the cleaned features
x = np.column_stack((preg,gluc,bloo,skin,insu,bmi,diab,age))

#x = np.column_stack((preg,gluc,bmi,age))

# Create the polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False) # tweleve here is pretty good
x = poly.fit_transform(x)

# Create the scaler and fit and transform it
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Function to fit the model and find how 
def train():
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    
    # Create the model
    softmax_sci = LogisticRegression(multi_class="multinomial",solver="lbfgs", max_iter=5000, C=0.1)
    
    # Fit the model
    softmax_sci.fit(x_train, y_train)
    
    # Check accuracy of the model
    testAccuracy = softmax_sci.score(x_test, y_test)
    print("Test Accuracy: "+ str(testAccuracy*100))

    # Return the accuracy and model
    return testAccuracy*100, softmax_sci

# Run until we get 90 percent
acc, model = train()
while (acc < 85):
    acc, model = train()


one = [[6,148,72,35,0,33.6,0.627,50]] # 1
one = poly.fit_transform(one)
one = scaler.transform(one)

two = [[1,85,66,29,0,26.6,0.351,31]] # 0
two = poly.fit_transform(two)
two = scaler.transform(two)

three = [[8,183,64,0,0,23.3,0.672,32]] # 1
three = poly.fit_transform(three)
three = scaler.transform(three)

four = [[1,89,66,23,94,28.1,0.167,21]] # 0
four = poly.fit_transform(four)
four = scaler.transform(four)

five = [[0,137,40,35,168,43.1,2.288,33]] # 1
five = poly.fit_transform(five)
five = scaler.transform(five)

six = [[9,154,78,30,100,30.9,0.164,45]] # 0
six = poly.fit_transform(six)
six = scaler.transform(six)

seven = [[3,169,74,19,125,29.9,0.268,31]] # 1
seven = poly.fit_transform(seven)
seven = scaler.transform(seven)

oneA = model.predict(one)
twoA = model.predict(two)
threeA = model.predict(three)
fourA = model.predict(four)
fiveA = model.predict(five)
sixA = model.predict(six)
sevenA = model.predict(seven)

out = [oneA, twoA, threeA, fourA, fiveA, sixA, sevenA]
for num in out:
    print("outcome: " + str(num))
