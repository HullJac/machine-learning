from pickle import load
import numpy as np

# Grab the weights from the file I put them in
w = np.loadtxt('w.txt')

# Load the scaler and polynomial so we can transform the inputs
# to match the degree of polynomialization
poly = load(open('poly.pkl', 'rb'))
minmax = load(open('minmax.pkl', 'rb'))

'''
# Grabbing input from the user
temp = float(input("average temperature on the interstate in kelvin: "))
rain = float(input("rain in mm for last hour: "))
snow = float(input("snow in mm for last hour: "))
cloud = float(input("cloud cover percentage : "))
'''

test = [[288.28,0,0,40]]
# Create input matrix
#test = [[temp, rain, snow, cloud]]

# Transforming the input to match the degree of polynomialization and minmax it
test = poly.transform(test)
test = minmax.transform(test)

# Add the column of ones
test = np.c_[np.ones(test.shape[0]), test]

# Predict the value and print it out
yhat = test.dot(w)
print("The predicted traffic volume is: {}".format(yhat))
