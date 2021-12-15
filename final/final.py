'''
Program:        Prediciting Heart Disease In Patients Using Gradient Boost and Voting Classifiers
Programmer:     Jacob Hull
Date:           12/16/21
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

#####################################################

# Grab the data
rawData = pd.read_csv('heart.csv')

'''
# Get info about the data
print("-------------------")
rawData.info()
print("-------------------")
print(rawData.describe())
print("-------------------")

# Create a heat map of the data
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

# Clean the data - taking out all the "-1" and replacing them with the most occuring value
X = imp.fit_transform(X)

#for i in range(20):
####################
#np.random.seed(i)#########
##################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #, random_state = i)

'''
for i in range(1,10):
    est = i * 100
    print(est)
    depth = i * 10
    print(depth)
'''
#    est = 150
#    depth = 25

# lists of parameters to test
#percents = [.3,.4,.5,.6,.7]
#leafs = [.1,.2,.3,.4,.5]
#ests = []
#deps = []
#lr = []

# fill lists if needed
#for i in range(1,11):
    #ests.append(i*3)
    #deps.append(i*2)
#    lr.append(i/10)

# turn lists to numpy arrays
#deps = np.array(deps)
#ests = np.array(ests)
#lr = np.array(lr)
#minSamplesLeaf = np.array(leafs)
#percents = np.array(percents)

#######Voting classifier##########


'''
########random forest##############
rnd_clf=RandomForestClassifier( 
                    n_estimators = est,
                    max_depth = depth,
                    n_jobs = -1
)

rnd_clf.fit(X_train, y_train)
train = rnd_clf.score(X_train, y_train)
test = rnd_clf.predict(X_test, y_test)

print("Random Forest")
print(str(train) + ":" + str(test))
'''

#########Gradient boost###############
gb_clf = GradientBoostingClassifier(
        n_estimators = 12,
        max_depth = 2,
        learning_rate = 0.33,
        min_samples_split = 0.7,
        min_samples_leaf = 0.15,
)




#    params = {
        #"n_estimators" : ests,
        #"max_depth" : deps,
        #"learning_rate" : lr,
        #"min_samples_split" : percents,
        #"min_samples_leaf" : minSamplesLeaf
#            }
'''
gb_clf.fit(X_train, y_train)
train = gb_clf.score(X_train, y_train)
y_pred = gb_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("gradient boost")
print(str(train) + ":" + str(acc))
'''

# Code for learning curve below taken from 
# https://vitalflux.com/learning-curves-explained-python-sklearn-example/
# I just changed the parameters to fit my data
train_sizes, train_scores, test_scores = learning_curve(
            estimator=gb_clf, 
            X=X,
            y=y,
            #cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 25), 
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
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()


'''
# Used to tune parameters 
#print(ada_clf.get_params().keys())
grid = GridSearchCV(estimator=gb_clf, param_grid = params)
grid.fit(X_train, y_train)
#print(grid)
print(grid.best_score_)
#print(grid.best_estimator_.n_estimators)
#print(grid.best_estimator_.max_depth)
print(grid.best_estimator_.learning_rate)
#print(grid.best_estimator_.min_samples_split)
#print(grid.best_estimator_.min_samples_leaf)
'''

print("######################################")

'''
https://machinelearningmastery.com/improve-deep-learning-performance/ 
'''
