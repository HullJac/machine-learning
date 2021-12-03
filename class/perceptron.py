import numpy as np
from sklearn.linear_model import Perceptron

#### AND perceptron
X = [[0,0],[0,1],[1,0],[1,1]]
'''
y = [0,0,0,1]

and_clf = Perceptron()
and_clf.fit(X,y)

print(and_clf.score(X,y))
print(and_clf.coef_)
print(and_clf.intercept_)
'''

### XOR perceptron
y = [0,1,1,0]

xor_clf = Perceptron()
xor_clf.fit(X,y)

print(xor_clf.score(X,y))
print(xor_clf.coef_)
print(xor_clf.intercept_)

from sklearn.neural_network import MLPClassifier

# run this many times to find a set of weigths that you can live with
xor_clf = MLPClassifier(
            activation='logistic',
            learning_rate='adaptive',
            max_iter=10000,
            hidden_layer_sizes=(2,),
            solver='sgd', 
            learning_rate_init=2,
            warm_start=True
        )
for i in range(10):
    xor_clf.fit(X,y)
    print(xor_clf.score(X,y))

# how to see what options there are
#print(xor_clf.get_params())


#print(xor_clf.coefs_)
#print(xor_clf.intercepts_)



