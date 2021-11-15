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

#################################################

rawData = pd.read_csv('StockTrainBin.csv')
data = rawData.to_numpy()

X = data[:,1:]
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y)

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
##########Random Forests###########
# same thing as bagging with decision trees
from sklearn.ensemble import RandomForestClassifier 
rnd_clf = RandomForestClassifier(
        n_estimators = 5000,
        max_samples = 0.4,
        max_features = 6,
        max_leaf_nodes = 160,
        n_jobs = -1
        )

rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

for name, score in zip(rawData, rnd_clf.feature_importances_):
    print(name, score)
# with this information, you can then trim some data out and then see how it performs again
'''
