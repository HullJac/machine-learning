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

# Add a new column of data dn print to make sure it is there
#rawData["BST"] = np.sqrt(pow(rawData["DCP"], 2) + pow(rawData["CFP"], 2) + pow(rawData["VP"], 2))
#print(rawData)

# Convert to numpy 
data = rawData.to_numpy()
# Separate the data and set aside some for testing
testData = data[:-30:,:]
#testX1 = testData[:,1:4]
#testX2 = testData[:,6:]
#testX = np.concatenate((testX1,testX2), axis=1)
testX = testData[:,1:]
testY = testData[:,0]


#x1 = data[:-30,1:4]
#x2 = data[:-30,6:]
#X = np.concatenate((x1,x2), axis=1)
#print(X)
X = data[:-30,1:]
y = data[:-30,0]

# Polynomialize the data 
#poly = PolynomialFeatures(degree=1, include_bias=False)
#X = poly.fit_transform(X)

scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)
# Creating the classifiers to make the voitng classifier
#tree_clf = DecisionTreeClassifier( # DecisionTree
#        max_depth=1000,
#        max_features=6,
#        min_samples_leaf=0.15,
#        min_samples_split=0.15)
#bag_clf = BaggingClassifier( # Bagging
#        DecisionTreeClassifier(),
#        n_estimators=5000, # number of decision trees to train
#        max_samples=0.5, # higher number here decreases variance, but increases bias
#        bootstrap=True, # if this is false, it would do pasting And maks it so no repeats are chosen forthe specific node at focus right now  #########true and false do about the same
#        n_jobs=-1,
#        max_features=6,
#        bootstrap_features=True)

svm_clf = SVC( # SVM
        C=1,
        kernel='poly', 
        coef0=1,
        degree=3,
        probability=True, 
        gamma="auto", 
        max_iter=500000)
soft_clf = LogisticRegression( # SoftMmax
        multi_class="multinomial",
        solver="lbfgs", 
        C=30, 
        max_iter=500000)
rnd_clf = RandomForestClassifier( # RandomForest 
        n_estimators=2000,
        #min_samples_split=0.05,
        bootstrap=True,
        max_samples=0.5,
        max_features=7,
        max_leaf_nodes=5000,
        max_depth=5000,
        n_jobs=-1)


# Create the classifier
stock_clf = VotingClassifier(
        estimators=[('rf', rnd_clf),  ('sm', soft_clf), ('sv', svm_clf)],
        voting='soft')

#('bc', bag_clf),,('tc', tree_clf)
#, bag_clf tree_clf,

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

'''
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=5000, # number of decision trees to train
        max_samples=0.4, # higher number here decreases variance, but increases bias
        bootstrap=False, # if this is false, it would do pasting And maks it so no repeats are chosen forthe specific node at focus right now  #########true and false do about the same
        n_jobs=-1,
        random_state=42,
        max_features=7,
        bootstrap_features=True,
        #oob_score=True # if you want to use this, you do not have to use the 80/20 split for validataion, this is the validation
)

bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
#print(bag_clf.oob_score_)

'''
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
