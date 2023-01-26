import numpy as np 

a = np.array([[0, 1, 2],
              [0, 2, 4],
              [0, 3, 6]])

print(np.sum(np.where(a == 4, 2, -1)))
