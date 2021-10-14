# Split into each column
y = data[:,0]
x1 = data[:,1]
x2 = data[:,2]
x3 = data[:,3]
x4 = data[:,4]
x5 = data[:,5]
x6 = data[:,6]

# Combine Xs to make data matrix 
x = np.column_stack((x1,x2,x3,x4,x5,x6))

# Normalize the Xs
x = minmax.fit_transform(x)
print(x)

# Add ones for W0 bias
x = np.c_[np.ones(x.shape[0]), x]
print(x)

LR.fit(x,y)

# Get the weights
w0 = LR.intercept_
w = LR.coef_
w[0] = w0
# This is the coeficients and are the same as before.
print(w)

# ---------------prediction set
minn = minmax.data_min_
maxx = minmax.data_max_

testSet = test.to_numpy()

y = testSet[:,0]
x1 = testSet[:,1]
x2 = testSet[:,2]
x3 = testSet[:,3]
x4 = testSet[:,4]
x5 = testSet[:,5]
x6 = testSet[:,6]

x1 = (x1-minn[0])/(maxx[0]-minn[0])
x2 = (x2-minn[1])/(maxx[1]-minn[1])
x3 = (x3-minn[2])/(maxx[2]-minn[2])
x4 = (x4-minn[3])/(maxx[3]-minn[3])
x5 = (x5-minn[4])/(maxx[4]-minn[4])
x6 = (x6-minn[5])/(maxx[5]-minn[5])

x = np.column_stack((x1,x2,x3,x4,x5,x6))

x = np.c_[np.ones(x.shape[0]), x]

# yhat is the prediction
yhat = x.dot(w)

#L1 error function
# gives percentage that we are off
result = np.abs((yhat-y)/y*100)
# we want the mean of that though
print(np.mean(result))

