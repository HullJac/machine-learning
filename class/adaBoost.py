#this is a class template that we will use in class.
#no sense in re-inventing the wheel
#I will add to this as we go
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
from sklearn.datasets import make_moons


from sklearn.ensemble import AdaBoostClassifier
# first testing data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# go throuhg and run it and then change corresponding values to better fit and see what else gets better
# alwasy care about generalization not the fitting of the data
maxx=0
np.random.seed(42)
for i in range(2,1000,1):
    ada_clf = AdaBoostClassifier (
        # Can put any classifier here that we want
        DecisionTreeClassifier(
            max_depth=2,
            max_leaf_nodes=i,
          #  max_features=2,
          #  min_samples_leaf=i #16
        #SVC(
        #    probability=True
           ),
        n_estimators=i,
        algorithm="SAMME.R",
        learning_rate=1.41,
        random_state=42
        ) # lr was 1.55
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score > maxx:
        print(str(i) + ":" + str(score))
        maxx=score
