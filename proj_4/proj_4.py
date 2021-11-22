'''
Program:        Prediciting Stock Data using A Voting Classifier
Programmer:     Jacob Hull
Date:           11/23/21
Description:    This program uses a voting classifier composed of an SVC, 
                SoftMax, and RandomForest classifiers. The voting classifier,
                called stock_clf, is trained on stock market data using an
                80/20 split and tested to check for accuracy through pulling 
                out an extra years worth of data before training. This score 
                is part of how I am basing the successfulness of my model, as 
                this shows more of how it can perform in a real life situation
                on new data. The other score that I am basing the successfulness
                of the model off is the accuracy on the 20% split. Together, these
                will tell me if the model is doing fairly well overall.
'''
import pandas as pd
import numpy as np
import random
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
#X = data[90:-270,1:]
#y = data[90:-270,0]
n = random.randint(0, 4000)
n2 = n + 365

#X = data[:-365,1:]
#y = data[:-365,0]

X1 = data[:n,1:]
y1 = data[:n,0]
X2 = data[n2:,1:]
y2 = data[n2:,0]

X=np.concatenate((X1,X2), axis=0)
y=np.concatenate((y1,y2), axis=0)

'''
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
'''
testData = data[n:n2,:]
testX = testData[:,1:]
testY = testData[:,0]

# Scale all the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)
#testX = scaler.transform(testX)
#testX2 = scaler.transform(testX2)
#testX3 = scaler.transform(testX3)
#testX4 = scaler.transform(testX4)

# Creating the classifiers to make the voitng classifier
svm_clf = SVC( # SVM
        C=100, # 100
        kernel='poly',
        coef0=1.0,
        degree=2,
        probability=True,
        gamma="auto",
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
        max_samples=0.5, # 1.0
        max_features=7,     # have 7 features here
        max_leaf_nodes=50, #100 or 50 50 probably
        n_jobs=-1
)


# Create the voting classifier
stock_clf = VotingClassifier(
        estimators=[('rf', rnd_clf), ('sm', soft_clf),('sv', svm_clf)],  #  
        voting='soft')


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
    print("---------------"+str(i)+"------------------")
''' 
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
'''

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
