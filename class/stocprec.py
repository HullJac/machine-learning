import pandas as pd
import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix


rawData=pd.read_csv('mnist_train.csv')
data =rawData.to_numpy()

X_train=data[:,1:]
y_train=data[:,0]

rawData=pd.read_csv('mnist_test.csv')
data =rawData.to_numpy()

X_test=data[:,1:]
y_test=data[:,0]

np.random.seed(420)

from sklearn.ensemble import AdaBoostClassifier
maxx=0
np.random.seed(42)


rawd = read_csv('numbers.csv')
d = rawd.to_numpy

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


from sklearn.neural_network import MLPClassifier
num_clf=MLPClassifier(
        activation='logistic',
        learning_rate='adaptive',
        max_iter=20000,
        hidden_layer_sizes=(100,),
        learning_rate_init=2,
        solver='sgd',
        warm_start=True
    )

#maxx=0
#for i in range(100):
num_clf.fit(X_train,y_train)
mm=num_clf.score(X_train,y_train)
y_hat=num_clf.predict(X_test)
ac=accuracy_score(y_test,y_hat)
print(str(mm)+":"+str(ac))

#    if mm>maxx:
#        maxx=mm
#print(maxx)
