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


# F signle dimension
delta = 0.01

epsilon = 0.001

maxSteps = 10000

randomSeed = 10

np.random.seed(randomSeed)


#position = np.random.sample(1)*100-50 # one here means give me one number

#sigma = F(position)

cost = np.empty(maxSteps) # mapping stuff

'''
for i in range(maxSteps):
    Dposition = np.random.randint(3,size=(1))-1

    cost[i] = sigma # mapping stuff

    tempPosition = position + delta * Dposition

    tempSigma = F(tempPosition[0])

    if np.abs(sigma-tempSigma) < epsilon and Dposition[0]!=0:
        print(i)
        break

    if tempSigma <= sigma:
        position = tempPosition
        sigma = tempSigma
'''
'''
# matplotlib stuff
fig, graph = plt.subplots()
graph.plot(cost)
plt.show()

print(position)
'''

# Fs multi dimensional
position = np.random.sample(2) # gives us an x,y

sigma = Fs(position[0], position[1])

for i in range(maxSteps):
    cost[i] = sigma
    dsigma = np.random.randint(3, size=(2)) -1 

    tempPosition = position + dsigma * delta
    tempSigma = Fs(tempPosition[0], tempPosition[1])
    if tempSigma <= sigma:
        position= tempPosition
        sigma = tempSigma

print(position)
