'''
Program:        Prediciting Stock Data using A Voting Classifier
Programmer:     Jacob Hull
Date:           11/23/21
Description:    This program uses
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Separate the x and y 
X = data[90:-270,1:]
y = data[90:-270,0]

# Grab some subsets of data to test on
testData = data[-90:,:]
testX = testData[:,1:]
testY = testData[:,0]

testData2 = data[-180:-90,:]
testX2 = testData2[:,1:]
testY2 = testData2[:,0]

testData3 = data[-270:-180,:]
testX3 = testData3[:,1:]
testY3 = testData3[:,0]

testData4 = data[0:90,:]
testX4 = testData4[:,1:]
testY4 = testData4[:,0]

# Scale all the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)
testX2 = scaler.transform(testX2)
testX3 = scaler.transform(testX3)
testX4 = scaler.transform(testX4)

# Creating the classifiers to make the voitng classifier
svm_clf = SVC( # SVM
        C=100, # 100
#        kernel='poly',
#        coef0=1.0,
#        degree=3,
        probability=True,
#        gamma="auto",
        #max_iter=50000000
        )
soft_clf = LogisticRegression( # SoftMmax
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        C=200, #200
)
rnd_clf = RandomForestClassifier( # RandomForest
        n_estimators=1000, #1000
        bootstrap=True,
        max_samples=1.0, # 1.0
        max_features=7,     # have 7 features here
        max_leaf_nodes=100, #100
#        max_depth=75,     #5000
        min_impurity_decrease= 0.1, #0.1
        n_jobs=-1
)


# Create the voting classifier
stock_clf = VotingClassifier(
        estimators=[('rf', rnd_clf), ('sm', soft_clf),('sv', svm_clf)],  #  
        voting='soft') # , weights=[0.25, 0.50, 0.25])


# Test the classifiers
for i in range(5):
    for clf in (rnd_clf, soft_clf, svm_clf, stock_clf): #
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Fitting the models and predicting
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

    # Test on data not in the set
    y_pred = stock_clf.predict(testX)
    acc1 = accuracy_score(testY, y_pred)
    print("Test on new data1: " + str(acc1))
    
    y_pred = stock_clf.predict(testX2)
    acc2 = accuracy_score(testY2, y_pred)
    print("Test on new data2: " + str(acc2))
    
    y_pred = stock_clf.predict(testX3)
    acc3 = accuracy_score(testY3, y_pred)
    print("Test on new data3: " + str(acc3))
    
    y_pred = stock_clf.predict(testX4)
    acc4 = accuracy_score(testY4, y_pred)
    print("Test on new data4: " + str(acc4))
    print("avg test acc: {}".format((acc1+acc2+acc3+acc4)/4))
    print("---------------"+str(i)+"------------------")


###################
# MONEY STUFF HERE#
###################


# The code below was used to check out the importance of the features
# to try and find some that may not be as important or some that are
# more important than others
##########Random Forests###########
'''
rnd_clf = RandomForestClassifier(
        n_estimators = 5000,
        min_samples_split = 5,
        max_samples = 0.4,
        max_features = 7,
        max_leaf_nodes = 5000,
        max_depth = 500,
        n_jobs = -1
        )

rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

for name, score in zip(rawData, rnd_clf.feature_importances_):
    print(name, score)
# with this information, you can then trim some data out and then see how it performs again
'''
