import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import seaborn as sns

# Loading the data and displaying it
rawData=pd.read_csv('diabetesBinary.csv')
data = rawData.to_numpy()
y = data[:,-1]
x = data[:,0:-1]

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
#print(x)


#sns.pairplot(rawData, hue="Outcome", markers=["o", "s"])
#plt.show()


preg = x[:,0]
gluc = x[:,1]
bloo = x[:,2]
skin = x[:,3]
insu = x[:,4]
bmi  = x[:,5]
diab = x[:,6]
age  = x[:,7] 


#glucose bmi pregs and blood pressure 

fig = plt.figure()
colors = ["red", "blue"]
color_indices = y
colormap = matplotlib.colors.ListedColormap(colors)

plt.scatter(gluc, preg, c=color_indices, cmap=colormap)
plt.xlabel("gluc")
plt.ylabel("preg")
plt.show()

plt.scatter(gluc, bloo, c=color_indices, cmap=colormap)
plt.xlabel("gluc")
plt.ylabel("bloo")
plt.show()

plt.scatter(gluc, skin, c=color_indices, cmap=colormap) # doesn't look awful
plt.xlabel("gluc")
plt.ylabel("skin")
plt.show()

plt.scatter(gluc, insu, c=color_indices, cmap=colormap) # not awful either
plt.xlabel("glucose")
plt.ylabel("insu")
plt.show()

plt.scatter(gluc, bmi, c=color_indices, cmap=colormap)
plt.xlabel("gluc")
plt.ylabel("bmi")
plt.show()

plt.scatter(gluc, diab, c=color_indices, cmap=colormap)
plt.xlabel("gluc")
plt.ylabel("diab")
plt.show()

plt.scatter(gluc, age, c=color_indices, cmap=colormap)
plt.xlabel("gluc")
plt.ylabel("age")
plt.show()


# Actually working on the data



'''
# First normalize the data
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# do not add a column of ones for this because they do it internally for us

softmax_sci = LogisticRegression(multi_class="multinomial", solver="lbfgs",C=10) # well come back to this
# C here is like the alpha

softmax_sci.fit(x,y)

# try to test it out
#xx = [[5,1]] # this is the value we want to predict
#px = scaler.transform(xx)
#print(softmax_sci.predict(px))
#print(softmax_sci.predict_proba(px))

# Graphing
x0set = x[:,0]
x1set = x[:,1]

x0min = np.amin(x0set)
x0max = np.amax(x0set)

x1min = np.amin(x1set)
x1max = np.amax(x1set)

delx0 = (x0max - x0min)/40.
delx1 = (x1max - x1min)/40.

# create an array of each so we can get a serries of samples
x0 = np.zeros(40)
x1 = np.zeros(40)

pair = np.empty((1600,2), float)

for i in range(40):
    x0[i] = x0min + i * delx0
    x1[i] = x1min + i * delx1

k = 0
for i in range(40):
    for j in range(40):
        pair[k][0] = x0[i]
        pair[k][1] = x1[j]
        k += 1

yguess = softmax_sci.predict(pair)

fig = plt.figure()
plt.scatter(pair[:,0], pair[:,1], c=yguess, cmap=colormap, alpha = 0.2)
plt.show()
sys.exit()

# Normalize pedal data
PL[:] = (PL[:]-PL[:].mean())/PL[:].std()
PW[:] = (PW[:]-PW[:].mean())/PW[:].std()

plt.scatter(PL, PW, c=color_indices, cmap = colormap)
plt.show()
'''
