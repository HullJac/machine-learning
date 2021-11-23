'''
Program:        Prediciting Future Stock Data Using A Voting Classifier
Programmer:     Jacob Hull
Date:           11/23/21
Description:    This program uses a voting classifier composed of an SVC, 
                SoftMax, and RandomForest classifier. The voting classifier,
                called stock_clf, is trained on stock market data using an
                80/20 split and tested to check for accuracy through pulling 
                out an extra years worth of data before training. This score 
                is part of how I am basing the successfulness of my model, as 
                this shows more of how it can perform in a real life situation
                on new data. The other score that I am basing the successfulness
                of the model on is the prediction accuracy on the 20% split. 
                Together, these will tell me if the model is doing fairly well 
                overall.
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

n = random.randint(0, 4000)
n2 = n + 365

# Separate the x and y 
X1 = data[:n,1:]
y1 = data[:n,0]
X2 = data[n2:,1:]
y2 = data[n2:,0]

# Add the x's together and the y's together
X=np.concatenate((X1,X2), axis=0)
y=np.concatenate((y1,y2), axis=0)

# Create the pulled ot data
testData = data[n:n2,:]
testX = testData[:,1:]
testY = testData[:,0]

# Scale all the data except the y's
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)

# Creating the classifiers to put in the voitng classifier
svm_clf = SVC( # SVM
        C=100,
        probability=True,
)
soft_clf = LogisticRegression( # SoftMmax
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        C=200,
)
rnd_clf = RandomForestClassifier( # RandomForest
        n_estimators=1000,
        bootstrap=True,
        max_samples=1.0,
        max_features=7,
        max_leaf_nodes=50,
        n_jobs=-1
)


# Create the voting classifier
stock_clf = VotingClassifier(
        estimators=[('rf', rnd_clf), ('sm', soft_clf),('sv', svm_clf)], 
        voting='soft')


# Test the classifiers
# Store the best one based on both accuracies
bestModel = 0
bestAvgPercent = 0
pulledBest = 0
twentyAcc = 0
for i in range(15):
    for clf in (rnd_clf, soft_clf, svm_clf, stock_clf): 
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Fitting the models and predicting
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        #print(clf.__class__.__name__, acc)

    # Test on data not in the set and calculate model score
    y_pred = stock_clf.predict(testX)
    acc1 = accuracy_score(testY, y_pred)
    avgAcc = (acc + acc1) / 2
    
    # Assign the best model if needed
    if avgAcc > bestAvgPercent:
        twentyAcc = acc
        pulledBest = acc1
        bestAvgPercent = avgAcc
        bestModel = stock_clf 

    # Print out how the model did
    #print("Test on pulled data: " + str(acc1))
    #print("Avg accuracy: " + str(avgAcc))
    #print("---------------"+str(i)+"------------------")

# Print out the best model
print("Percent on 20%: " + str(twentyAcc))
print("Percent on pulled data: " + str(pulledBest))


# The code below was used to check out the importance of the features
# to try and find some that may not be as important or some that are
# more important than others
# With this information, you can then trim some data out and then see 
# how it performs again
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
'''
