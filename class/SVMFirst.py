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
from sklearn import svm

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


rawData=pd.read_csv('iris.csv')
dataR=rawData.replace(to_replace=['Iris-setosa','Iris-versicolor','Iris-virginica'],value=[0,1,2])

data=dataR.to_numpy()

SL=data[:,1]
SW=data[:,2]
PL=data[:,3]
PW=data[:,4]
y=data[:,5]

fig=plt.figure()
colors=["red","blue","purple"]
color_indices=y

colormap=matplotlib.colors.ListedColormap(colors)

plt.scatter(SL,SW,c=color_indices,cmap=colormap)
plt.xlabel("sepal Length")
plt.ylabel("sepal Width")
plt.show()
fig=plt.figure()
plt.scatter(PL,PW,c=color_indices,cmap=colormap)
plt.xlabel("pedal Length")
plt.ylabel("pedal Width")
plt.show()

# take the pedal info

X=copy.deepcopy(data[:,3:5])
# normalize based on the variation in the data
# this minimizes the influence of outliers
X[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

#print(X)
linsvm=svm.LinearSVC(C=1) # one is the default

linsvm.fit(X, y)
print(linsvm.predict([[0,0]]))

x0set=X[:,0]
x1set=X[:,1]

x0min=np.amin(x0set)
x0max=np.amax(x0set)

x1min=np.amin(x1set)
x1max=np.amax(x1set)

delx0=(x0max-x0min)/40.

delx1=(x1max-x1min)/40.

x0=np.zeros(40)
x1=np.zeros(40)

pair=np.empty((1600,2),float)

for i in range(40):
    x0[i]=x0min+i*delx0
    x1[i]=x1min+i*delx1

k=0

for i in range(40):
    for j in range(40):
        # used to be anoher here 
        pair[k][0]=x0[i]
        pair[k][1]=x1[j]
        k+=1

yguess=linsvm.predict(pair)

fig=plt.figure()

plt.scatter(pair[:,0],pair[:,1],c=yguess,cmap=colormap,alpha=.2)

#plt.show()

PL[:]=(PL[:]-PL[:].mean())/PL[:].std()
PW[:]=(PW[:]-PW[:].mean())/PW[:].std()

plt.scatter(PL,PW,c=color_indices,cmap=colormap)

plt.show()


# it is sciklearn brain on neutral time

data=dataR.to_numpy()

X=copy.deepcopy(data[:,3:5])

# we will use the industry standard scaler normaliztion
# subtract mean and divide by unit variate deviation

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler=StandardScaler()

scaler.fit(X)
X=scaler.transform(X)

softmax_sci=LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10)

softmax_sci.fit(X,y)

xx=[[5,1]]

px=scaler.transform(xx)

print(softmax_sci.predict(px))
print(softmax_sci.predict_proba(px))

x0set=X[:,0]
x1set=X[:,1]

x0min=np.amin(x0set)
x0max=np.amax(x0set)

x1min=np.amin(x1set)
x1max=np.amax(x1set)

delx0=(x0max-x0min)/40.

delx1=(x1max-x1min)/40.

x0=np.zeros(40)
x1=np.zeros(40)

pair=np.empty((1600,2),float)

for i in range(40):
    x0[i]=x0min+i*delx0
    x1[i]=x1min+i*delx1

k=0

for i in range(40):
    for j in range(40):
        
        pair[k][0]=x0[i]
        pair[k][1]=x1[j]
        k+=1


yguess=softmax_sci.predict(pair)


fig=plt.figure()

plt.scatter(pair[:,0],pair[:,1],c=yguess,cmap=colormap,alpha=.2)

plt.show()


sys.exit()

PL[:]=(PL[:]-PL[:].mean())/PL[:].std()
PW[:]=(PW[:]-PW[:].mean())/PW[:].std()

plt.scatter(PL,PW,c=color_indices,cmap=colormap)

plt.show()
