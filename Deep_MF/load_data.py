import numpy as np 


x = np.load('test.npy')
print(x.dtype)
print(x[1])

a = np.full((101), 2)
print(a)