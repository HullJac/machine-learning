# Linear algebra with numpy
import numpy as np

# Row vector
l1 = [40, 25, 6]
rowVector = np.array(l1)
print(rowVector)


# Column vector
l2 = [[3], [7], [8]]
columnVector = np.array(l2)
print(columnVector)

# Dot product returns a scalar 
print(np.dot(rowVector, columnVector))

# Reverse will work if you transpose them
print(np.dot(columnVector.T, rowVector.T))

print(rowVector.dot(columnVector))

# Matrix
m1 = [[6,-2,0], [1,0,4], [-8,6,2]]
mat1 = np.array(m1)
print(mat1)
# Matrix transpose
print(mat1.T)

# Determinate
print(np.linalg.det(mat1))

# Inverse
mat1i = np.linalg.inv(mat1)
print(mat1i)

# Matrix dotted with inverse
print(mat1.dot(mat1i).round(10))


