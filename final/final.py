'''
Program:        Prediciting Heart Disease In Patients Using Bootsing
Programmer:     Jacob Hull
Date:           
Description:    This program 
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor

#####################################################

# Grab the data
rawData = pd.read_csv('heart.csv')

# Convert to numpy 
data = rawData.to_numpy()

# Create imputer to clean data with
imp = SimpleImputer(missing_values=-1, strategy='most_frequent')

# Separate the x and y
X = data[:,:-1]
y = data[:,-1]

# Clean the data - taking out all the "?" and replacing them with the mean
X = imp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

########random forest#######
rnd_clf=RandomForestClassifier(n_estimators=500, n_jobs=-1)

rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)

print("Random Forest")
print(accuracy_score(y_test,y_pred)*100)


ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(
            max_depth=1
           #,max_leaf_nodes=i
           #  ,max_features=2
           #,min_samples_leaf=i
            )
            , n_estimators=30,  # ada boost can do this one with less estimators
    algorithm="SAMME.R", learning_rate=1.0)
ada_clf.fit(X_train, y_train)
y_pred=ada_clf.predict(X_test)
score=accuracy_score(y_test,y_pred)

print("adaBoost")
print(score*100)


###############################################
#Gradient boost

#Takes weak regressors and makes them strong
# This is how you would fit a classification routine like this too
gb = GradientBoostingRegressor(
        max_depth=2, 
        n_estimators=10, 
        learning_rate=1.0, 
        )

gb.fit(X,y)

print("gradient boost")
print(gb.score(X_test, y_test)*100)

# this is all based on trees, whereas adaboost can use anything
# next time, wrtite a routine to find the perfect amount of trees
