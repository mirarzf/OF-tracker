import numpy as np 

A = np.array([
    [1, 0 ], 
    [0, 1]
    ])
B = np.array([
    [8, 2], 
    [3, 4]
    ])
C = np.array([1, 2])
D = np.array([
    [1, 1], 
    [1, 1]
    ])
E = np.ones((2,3,4), dtype = int)
E[1] = E[1]*3
result = np.sum(E, axis = 0)

print(
    E, "\n", 
    result
)