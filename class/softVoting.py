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

def plot_learning_curves(model,X,y,mx,my):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)

    train_errors,val_errors=[],[]

    for m in range(1,mx):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))

        val_errors.append(mean_squared_error(y_val,y_val_predict))


    plt.plot(np.sqrt(train_errors),"r.",linewidth=2,label="train")

    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="validate")
    plt.xlim([0,mx])
    plt.ylim([0, my])
    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)

    wo=model.intercept_
    w=model.coef_
    w[0]=wo

    return w


#np.random.seed(4444444)
rawData=pd.read_csv('winequality-redMulti.csv')

data = rawData.to_numpy()

x = data[:,0:11]
y = data[:,11]

x_train, x_test, y_train, y_test = train_test_split(x, y)# leaving default will be 25 75%

from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# deliberatly making some of these bad
log_clf = LogisticRegression(C=1, max_iter=50)

tree_clf = DecisionTreeClassifier(max_depth=4)

svm_clf = SVC(gamma="auto")

voting_clf = VotingClassifier(
        estimators=[('lc', log_clf), ('tc', tree_clf), ('sc', svm_clf)],
        voting='hard')


for clf in (log_clf, tree_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# Soft Voting
svm_clf = SVC(probability=True, gamma = "auto")

voting_clf = VotingClassifier(
        estimators=[('lc', log_clf), ('tc', tree_clf), ('sc', svm_clf)],
        voting='soft')

for clf in (log_clf, tree_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
