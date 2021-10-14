'''
Program:    2-d Gradient Descent
Programmer: Jacob Hull
Date:       9/17/21
Overview:   This program will take a function with two variables that is provided by the user and 
            by using graident descent will find the global minimum of that function.
'''
import numpy as np
import matplotlib.pyplot as plt
from function import *
import math

# Surpresses the warning I get when receiving NaN values
# I do this because I am using the NaN values to see if I am close
# If I get a NaN, then I know to stop moving the corresponding x or y
import warnings
warnings.filterwarnings("ignore")

def main():
    # Variables to change the outcome of the gradient descent
    delta = 3
    eta = .01
    maxSteps = 30000

    # Holds the outputs of the function to make a cost curve
    # Commented out the run faster
    #cost = np.empty(maxSteps)
    
    # Setting the seed to allow for better testing
    #randomSeed = 100
    #np.random.seed(randomSeed)
    
    # Get a random (x, y)
    prevPosition = np.random.sample(2)*100-50

    # dSigma will be 0 or 1
    dSigma = np.random.randint(2)
   
    # Pick a random position delta away from the prevPosition
    currPosition = prevPosition + delta * (-1) ** dSigma

    # Assign better names to my curr and prev x and y 
    # so I can better keep track of these later when reassigning
    currX = currPosition[0]
    currY = currPosition[1]
    prevX = prevPosition[0]
    prevY = prevPosition[1]
   
    # Run the gradient descent function maxSteps times
    for i in range(maxSteps):
        # Get curret sigmas with starting spots
        sigmaX = F(currX, prevY)
        sigmaY = F(prevX, currY)
   
        # Get prev sigma for computation later
        prevSigma = F(prevX, prevY)
        
        # Add evaluated function to cost curve to see how good we are going
        # Commented out to run faster
        #cost[i] = F(currX, currY)

        # This block of code is the meat of gradient descent
        # It evaluates the x and y separately and checks for NaNs to stop
        nextPositionX = currX - eta * (sigmaX - prevSigma)/(currX - prevX) 
        if (math.isnan(nextPositionX)):
            pass 
        else:
            prevX = currX
            currX = nextPositionX
        nextPositionY = currY - eta * (sigmaY - prevSigma)/(currY - prevY) 
        if (math.isnan(nextPositionY)):
            pass 
        else:
            prevY = currY
            currY = nextPositionY

    # Sets up the graph and displays it
    # Commented for faster run time and no clutter
    #fig, graph=plt.subplots()
    #graph.plot(cost)
    #plt.show()

    return currX, currY

# Runs the gradient descent a number of times, I chose 50
# Then cretes an array of the fucntion evaluated at the outcome (x, y) coordinate
# Then finds the minimum of the evaluated outcomes and returns the corresponding coordinates
if __name__ == "__main__":
    xArray = []
    yArray = []
    outputs = []
    for i in range(50):
        currX, currY = main()
        xArray.append(currX)
        yArray.append(currY)
        outputs.append(F(currX, currY))
    ind = outputs.index(min(outputs))
    print("X Coordinate: {}\nY Coordinate: {}".format(xArray[ind], yArray[ind]))
