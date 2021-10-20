import pandas as pd

data = pd.read_csv('OldFaithfulData.txt', delimiter="\t", header=None)

print(data)

import matplotlib.pyplot as plt
import numpy as np

fig, graph = plt.subplots()
graph.plot(data[0], data[1], ".")
plt.show()

# The data matrix, we chose the first column
X = data[0].to_numpy()

X = np.c_[np.ones(X.shape[0]), X]

print(X)

Y = data[1].to_numpy()

# finding the transpose
Xt = np.transpose(X)

#equation = W = (Xt * X)^-1 * Xt*Y

# Pieces of the equation
right = np.dot(Xt, Y)
temp = np.dot(Xt, X)
left = np.linalg.inv(temp)

W = np.dot(left, right)

print(W)


from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X, Y)

print(LR.intercept_)
print(LR.coef_)
