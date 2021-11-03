'''
Program:

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.metrics import accuracy_score

def plot_learning_curves(model,X,y):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)

    train_errors,val_errors=[],[]

    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))

        val_errors.append(mean_squared_error(y_val,y_val_predict))


    plt.plot(np.sqrt(train_errors),"r.",linewidth=2,label="train")

    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="validate")
    plt.legend(loc="upper right",fontsize=14)
    plt.xlabel("Training set size",fontsize=14)
    plt.ylabel("RMSE",fontsize=14)

    wo=model.intercept_
    w=model.coef_
    w[0]=wo

    return w

# Loading the data and displaying it
rawData=pd.read_csv('diabetesBinary.csv')
data = rawData.to_numpy()

# Separate the y column
y = data[:,-1]

# Separate the differet features
preg = data[:,0]
gluc = data[:,1]
bloo = data[:,2]
skin = data[:,3]
insu = data[:,4]
bmi  = data[:,5]
diab = data[:,6]
age  = data[:,7] 

# Create the cleaner object
imp = SimpleImputer(missing_values=0, strategy='mean')

# Create the data matrix to clean
x = np.column_stack((gluc,bloo,skin,insu,bmi,diab,age))

# Clean the data (all except pregnancies becaseu you can have 0 of those)
x = imp.fit_transform(x)

# Grab each column again
gluc = x[:,0]
bloo = x[:,1]
skin = x[:,2]
insu = x[:,3]
bmi  = x[:,4]
diab = x[:,5]
age  = x[:,6] 

# Create new data matrix from all the cleaned features
#x = np.column_stack((preg,gluc,bloo,skin,insu,bmi,diab,age))
x = np.column_stack((gluc,bloo,insu,bmi,diab,age))

# Create the polynomial features
poly = PolynomialFeatures(degree=12, include_bias=False) # tweleve here is pretty good
x = poly.fit_transform(x)

# Create the scaler and fit and transform it
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

#print(x)

#sns.pairplot(rawData, hue="Outcome", markers=["o", "s"])
#plt.show()

# Then normalize the data
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Create the model
softmax_sci = LogisticRegression(multi_class="multinomial",solver="lbfgs", max_iter=5000, C=50)
# well come back to this
# C here is like the alpha
# can add a max_iter variable here too

# put softmax fit back here
softmax_sci.fit(x,y)

w = plot_learning_curves(softmax_sci, x, y)
print(w)

def train():
    X_train,X_val,y_train,y_val=train_test_split(x,y,test_size=0.2)
    

    accuracy = accuracy_score(X_val, y_val)
    print("Accuracy: " + str(accuracy*100))
    return accuracy
'''
ac = train()
while ac < 94.0:
    ac = train()
'''
