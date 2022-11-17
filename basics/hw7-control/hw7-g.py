import numpy as np
import control

A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, -1, -3, 3]
])

B = np.array([
    [2, 0], 
    [0, 0], 
    [0, 0],
    [0, 1]
])

F = control.place(A, B, [-1, -2, -1+2j, -1-2j])
print(f'F = \n {F}')