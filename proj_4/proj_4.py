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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

#####################################################

# Grab the data
rawData = pd.read_csv('StockTrainBin.csv')

# Convert to numpy 
data = rawData.to_numpy()

# Set some data aside some for extra testing and split into x and y
testData = data[:-30:,:]
testX = testData[:,1:]
testY = testData[:,0]

# Separate the x and y 
X = data[:-30,1:]
y = data[:-30,0]

# Scale all the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)

# Creating the classifiers to make the voitng classifier
svm_clf = SVC( # SVM
        C=5,
        kernel='poly',
        coef0=1,
        degree=6,
        probability=True,
        gamma="auto",
        max_iter=50000000)
soft_clf = LogisticRegression( # SoftMmax
        multi_class="multinomial",
        solver="lbfgs",
        C=15,
        max_iter=50000000)
rnd_clf = RandomForestClassifier( # RandomForest
        n_estimators=2000,
        bootstrap=True,
        max_samples=1.0,
        max_features=7,
        max_leaf_nodes=5000,
        max_depth=5000,
        n_jobs=-1)


# Create the voting classifier
stock_clf = VotingClassifier(
        estimators=[('rf', rnd_clf),  ('sm', soft_clf), ('sv', svm_clf)],
        voting='soft')


# Test the classifiers
for clf in (rnd_clf, soft_clf, svm_clf, stock_clf):
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fitting the models and predicting
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Test on data not in the set
y_pred = stock_clf.predict(testX)
acc = accuracy_score(testY, y_pred)
print("Test on new data: " + str(acc))


# The code below was used to check out the importance of the features
# to try and find some that may not be as important or some that are
# more important than others
'''
##########Random Forests###########
# same thing as bagging with decision trees
rnd_clf = RandomForestClassifier(
        n_estimators = 5000,
        min_samples_split = 5,
        max_samples = 0.4,
        max_features = 8,
        max_leaf_nodes = 5000,
        max_depth = 500,
        n_jobs = -1
        )

rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
print()
print(accuracy_score(y_test, y_pred))

for name, score in zip(rawData, rnd_clf.feature_importances_):
    print(name, score)
# with this information, you can then trim some data out and then see how it performs again
'''
