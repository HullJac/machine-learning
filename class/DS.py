import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return x**4+.5*x**3-35*x**2-16.5*x+540



delta=.01

epsilon=.01

maxSteps=100000

randomSeed=10

np.random.seed(randomSeed)

position=np.random.sample(1)*100-50
lim=0
sigma=F(position)
cost=np.empty(maxSteps)
for i in range(maxSteps):
    Dposition=np.random.randint(3,size=(1))-1
    cost[i]=sigma
    tempPosition=position+delta*Dposition

    tempSigma=F(tempPosition[0])

    
    if np.abs(sigma-tempSigma)<epsilon and Dposition[0]!=0:
        lim=i
        break
    
    if tempSigma<=sigma:
        position=tempPosition
        sigma=tempSigma

    

fig, graph=plt.subplots()
graph.plot(cost)
plt.xlim((0, i))
plt.show()

print(position)
