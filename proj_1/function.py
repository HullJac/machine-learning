'''
Program:    Functions for 2-d Gradient Descenit
Programmer: Jacob Hull
Date:       9/17/21
Overview:   This program holds the function definitions that I tested for my gradient descent code.
'''
import numpy as np

# (3, 1)
#def F(x, y):
#    return 2*x**2 + 3*y**2 - 12*x - 6*y + 9

# (-2, 2)
def F(x, y):
    t1 = .01*x**2 + 0.01*y**2
    t2 = np.exp(-x**2-y**2)
    t3 = -.5*np.exp(-(x-2)**2-(y-2)**2)
    t4 = -.75*np.exp(-(x+2)**2-(y-2)**2)
    return t1 + t2 + t3 + t4

# (-0.5, -0.5) 
#def F(x, y):
#    return 5*x**2 + 5*y**2 + 5*x + 5*y + 5

# (0,0)
#def F(x, y):
#    return x**2 + y**2
