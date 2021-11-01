# this template is for us to explore the different
# non-linear techniques avaiable to us in sciklearn
# to do multi-classification problems that resist
# linear methods (which seem to be most)
# the pretty picture algirithms are borrowed from another source.

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import copy,sys
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)



def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


from sklearn.datasets import make_moons

x,y = make_moons(n_samples=100, noise=.15, random_state=42)

#plot_dataset(x,y,[-1.5,2.5,-1,1.5])
#plt.show()

# now going to apply a polynomial to this
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=6)),
    ("scaler",StandardScaler()),
    ("svm_clf",LinearSVC(C=10,loss="hinge",random_state=42))
    ]) # All have to have the same interface to pipeline

polynomial_svm_clf.fit(x,y)

#plot_predictions(polynomial_svm_clf, [-1.5,2.5,-1,1.5])
#plot_dataset(x,y,[-1.5,2.5,-1,1.5])
#plt.show()


# now going to use the kernel trick

poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_poly", SVC(kernel='poly', coef0=1, degree=10, C=1)) # coef0 limits the effect of higher degree of polynomials
    ])


poly_kernel_svm_clf.fit(x,y)
plot_predictions(poly_kernel_svm_clf, [-1.5,2.5,-1,1.5])
plot_dataset(x,y,[-1.5,2.5,-1,1.5])
plt.show()


# need a method to count the incursions
