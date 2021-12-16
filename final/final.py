'''
Program:        Prediciting Heart Disease In Patients Using Gradient Boost and Voting Classifiers
Programmer:     Jacob Hull
Date:           12/16/21
Description:    This program trains three different machine learning classifiers and tests their
                accuracy of predicting heart disease using learning curves. The heart disease 
                data comes from the Cleveland Clinic Foundation, and was provided by 
                Robert Detrano, M.D., Ph.D. and David Aha. The data is cleaned with this
                code by using a simple imputer and putting the most commly occuring value in 
                place of the missing values. The models used for classification in the program
                are logistic regression, ada boost classifier, and gradient boost classifier.
                These are trained using a 75/25 split of the data at random and are check for 
                accuracy based on the remaining 25 percent of data that was not used during 
                training. The accuracies of the training and testing of each model are printed 
                to the screen along with the learning curve for each. 


                To run the program, you need Python 3.9 or above and all the imported libraries
                listed below in the imports section installed on your machine or environment. 
                To execute the program, you can simple call "python3 final.py" as long as the 
                "heart.csv" data file is in the working directory.
'''

###########
# Imports #
###########
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

#####################################################

# Grab the data
rawData = pd.read_csv('heart.csv')

# Scatter matrix
'''
attributes = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','y']
scatter_matrix(rawData[attributes])
plt.show()
'''

# Block below gets info about the data and creates a heat map of the data
'''
print("-------------------")
rawData.info()
print("-------------------")
print(rawData.describe())
print("-------------------")

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(rawData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
'''

# Convert to numpy 
data = rawData.to_numpy()

# Create imputer to clean data with
imp = SimpleImputer(missing_values=-1, strategy='most_frequent')

# Separate the x and y
X = data[:,:-1]
y = data[:,-1]

# Clean the data = taking out all the "-1" and replacing them with the most occuring value
X = imp.fit_transform(X)

# Split the data for testing and training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



#######################
# Logistic Regression #
#######################
log_clf = LogisticRegression(
        max_iter = 1000000,
        multi_class='multinomial',
        C = 0.35,
        solver = 'newton-cg'
)

log_clf.fit(X_train, y_train)
train = log_clf.score(X_train, y_train)
y_pred = log_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Logistic Regression")
print("Train = " + str(train) + "\nTest  = " +str(acc))


# Code for learning curve below taken from 
# https://vitalflux.com/learning-curves-explained-python-sklearn-example/
# I just changed the parameters to fit my data
train_sizes, train_scores, test_scores = learning_curve(
            estimator=log_clf, 
            X=X,
            y=y,
            train_sizes=np.linspace(0.01, 1.0, 50), 
            n_jobs=-1
        )

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
#
# Plot the learning curve
#
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Logistic Regression Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()


# Lists and grid seach helped to find the best C value and solver method
# From these, I tuned them based on the learning curve to generalize better
'''
percents = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
#solves = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

log_params = {"C" : percents}
log_params2 = {"solver" : solves}

grid = GridSearchCV(estimator=log_clf, param_grid=log_params)
grid.fit(X_train, y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.C)
#print(grid.best_estimator_.solver)
'''



#############
# Ada Boost #
#############
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(
        max_depth = 1
    ), 
    n_estimators=26, 
    algorithm="SAMME.R", 
    learning_rate=.25
)

ada_clf.fit(X_train, y_train)
train = ada_clf.score(X_train, y_train)
y_pred = ada_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Ada Boost")
print("Train = " + str(train) + "\nTest  = " +str(acc))


# Code for learning curve below taken from 
# https://vitalflux.com/learning-curves-explained-python-sklearn-example/
# I just changed the parameters to fit my data
train_sizes, train_scores, test_scores = learning_curve(
            estimator=ada_clf, 
            X=X,
            y=y,
            train_sizes=np.linspace(0.01, 1.0, 50), 
            n_jobs=-1
        )

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
#
# Plot the learning curve
#
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Ada Boost Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()


##################
# Gradient Boost #
##################
gb_clf = GradientBoostingClassifier(
        n_estimators = 12,
        max_depth = 1,
        learning_rate = 0.33,
        min_samples_split = 0.7,
        min_samples_leaf = 0.15,
        max_features = 6
)

gb_clf.fit(X_train, y_train)
train = gb_clf.score(X_train, y_train)
y_pred = gb_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Gradient Boost")
print("Train = " + str(train) + "\nTest  = " +str(acc))


# Code for learning curve below taken from 
# https://vitalflux.com/learning-curves-explained-python-sklearn-example/
# I just changed the parameters to fit my data
train_sizes, train_scores, test_scores = learning_curve(
            estimator=gb_clf, 
            X=X,
            y=y,
            train_sizes=np.linspace(0.01, 1.0, 50), 
            n_jobs=-1
        )

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
#
# Plot the learning curve
#
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Gradient Boost Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()


# Code in comment block below was used to find the best parameters for gradient boost
# This was done with the grid search technique
# From these, I tuned them based on the learning curve to generalize better
'''
for i in range(1,10):
    est = i * 100
    print(est)
    depth = i * 10
    print(depth)
    est = 150
    depth = 25

# Lists of parameters to test
percents = [.3,.4,.5,.6,.7]
leafs = [.1,.2,.3,.4,.5]
ests = []
deps = []
lr = []

# Fill lists if needed
for i in range(1,11):
    ests.append(i*3)
    deps.append(i*2)
    lr.append(i/10)

# Turn lists to numpy arrays
deps = np.array(deps)
ests = np.array(ests)
lr = np.array(lr)
minSamplesLeaf = np.array(leafs)
percents = np.array(percents)

params = {
    "n_estimators" : ests,
    "max_depth" : deps,
    "learning_rate" : lr,
    "min_samples_split" : percents,
    "min_samples_leaf" : minSamplesLeaf
}

#print(ada_clf.get_params().keys())
grid = GridSearchCV(estimator=gb_clf, param_grid = params)
grid.fit(X_train, y_train)
#print(grid)
print(grid.best_score_)
print(grid.best_estimator_.n_estimators)
print(grid.best_estimator_.max_depth)
print(grid.best_estimator_.learning_rate)
print(grid.best_estimator_.min_samples_split)
print(grid.best_estimator_.min_samples_leaf)
'''
