'''
Program:        Showcase For Ridge Regression Using Polynomial Features To Traffic Data
Programmer:     Jacob Hull
Date:           10/20/21
Description:    This Program uses the polynomial features, scaler, and weights from the trained
                Ridge Regression model to predict the flow of traffic on interstate 94 in 
                Minneapolis Minnesota based on user input. It loads the previous mentioned
                information from files that were saved by the train.py file so we do not have to 
                train the model everytime we want to predict the traffic flow. I then transform the 
                user input into the proper form so that I can calculate a predicted value based on 
                the trained model. The predicted value is then shown to the user in an easily 
                readable format.

                To run this program, you need to have ran the train.py program as it creates
                the model and files needed to predict the traffic flow value.  The files created
                by the train.py file must be in the same directory as this program.
                Those files are w.txt, poly.pkl, and minmax.pkl.
'''
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
