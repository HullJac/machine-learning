import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return x**4 + .5*x**3 - 35*x**2 - 16.5*x + 540

def Fs(x,y):
    t1 = .01*x**2 + 0.01*y**2
    t2 = np.exp(-x**2-y**2)
    t3 = -.5*np.exp(-(x-2)**2-(y-2)**2)
    t4 = -.75*np.exp(-(x+2)**2-(y-2)**2)

    return t1 + t2 + t3 + t4

delta = 10
eta = .00001

maxSteps = 5000

cost = np.empty(maxSteps)

randomSeed = 10

np.random.seed(randomSeed)

position = np.random.sample(1)*100-50

sigma = F(position[0])

dSigma = np.random.randint(3) # will be 0,1,or 2

tempPosition = position + delta * (-1) ** dSigma #This will be plus or minus 10 or 20

tempSigma = F(tempPosition[0])

for i in range(maxSteps):
    cost[i] = sigma
    
    hold = position # info to preserve

    position = position - eta * (-sigma + tempSigma)/np.abs(position - tempPosition) # this line is gradient decent tends to go to better numbers

    tempSigma = sigma
    sigma = F(position[0])
    tempPosition = hold # the old position

print(position)

# this is gradient decent in one dimension

# two dimesnions is similar, 
