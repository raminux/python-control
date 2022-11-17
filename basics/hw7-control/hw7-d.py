import numpy as np
import control
from scipy.linalg import null_space

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

lamda1 = -1
AB1 = np.hstack((lamda1*np.eye(4)-A, B))

lamda2 = -2
AB2 = np.hstack((lamda2*np.eye(4)-A, B))

lamda3 = -1+2j
AB3 = np.hstack((lamda3*np.eye(4)-A, B))

nullAB1 = null_space(AB1)
nullAB2 = null_space(AB2)
nullAB3 = null_space(AB3)

M1 = nullAB1[0:4, :]
D1 = - nullAB1[4:, :]

M2 = nullAB2[0:4, :]
D2 = -nullAB2[4:, :]

M3 = nullAB3[0:4, :]
D3 = -nullAB3[4:, :]

Zita1 = np.array([
    M1[0, 1], 
    -M1[0, 0]
])

Zita2 = np.array([
    1, 1
])

Zita3 = np.array([
    M3[0,1],
    -M3[0, 0]
])

v1 = (M1@Zita1.T).reshape(4,1)
v2 = (M2@Zita2.T).reshape(4,1)
v3 = (M3@Zita3.T).reshape(4,1)
v4 = np.conj(v3)

q1 = (D1@Zita1.T).reshape(2,1)
q2 = (D2@Zita2.T).reshape(2,1)
q3 = (D3@Zita3.T).reshape(2,1)
q4 = np.conj(q3)

Q = np.hstack((q1, q2, q3, q4))
V = np.hstack((v1, v2, v3, v4))

F = Q@np.linalg.inv(V)
print(F)



