import pandas as pd
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


# Make the data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rnd_clf=RandomForestClassifier(n_estimators=500, n_jobs=-1)

rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)

print(accuracy_score(y_test,y_pred))


plot_decision_boundary(rnd_clf, X, y)
plt.show()


from sklearn.ensemble import AdaBoostClassifier
maxx=0
np.random.seed(42)

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
print(score)

plot_decision_boundary(ada_clf, X, y)
plt.show()


###############################################
#Gradient boost

np.random.seed(42)
X = np.random.rand(100, 1) - .5

y = 3 * X[:,0] ** 2 + .05 * np.random.randn(100)

plt.scatter(X,y)
plt.show()

from sklearn.tree import DecisionTreeRegressor

###########################y1

tree1 = DecisionTreeRegressor(
            max_depth=2,
            random_state=42
            )

tree1.fit(X, y)
yguess1 = tree1.predict(X)

# Visualizing what a tree regressor looks like
plt.scatter(X, y)
plt.scatter(X, yguess1)
plt.show()

#############################y2
# y becomes the diff
y2 = y - yguess1

tree2 = DecisionTreeRegressor(
            max_depth=2,
            random_state=42
            )

tree2.fit(X, y2)
yguess2 = tree2.predict(X)

# Visualizing what a tree regressor looks like
plt.scatter(X, y2)
plt.scatter(X, yguess2)
plt.show()
# still not good, so lets go again

#########################y3
y3 = y2 - yguess2

tree3 = DecisionTreeRegressor(
            max_depth=2,
            random_state=42
            )

tree3.fit(X, y3)
yguess3 = tree3.predict(X)

# Visualizing what a tree regressor looks like
plt.scatter(X, y3)
plt.scatter(X, yguess3)
plt.show()



##### checking how we did overall
plt.scatter(X, yguess1 + yguess2 + yguess3)
plt.scatter(X, y)
plt.show()
# not very impressive



################3

from sklearn.ensemble import GradientBoostingRegressor
# this just automates what we were doing 

#Takes weak regressors and makes them strong
# This is how you woudl fit a classification routine like this too
gb = GradientBoostingRegressor(
        max_depth=2, 
        n_estimators=10, 
        learning_rate=1.0, 
        random_state=42
        )

gb.fit(X,y)
yguess = gb.predict(X)

plt.scatter(X,y)
plt.scatter(X,yguess)
plt.show()

# this is all based on trees, whereas adaboost can use anything
# next time, wrtite a routine to find the perfect amount of trees
