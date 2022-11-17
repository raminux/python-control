import numpy as np
import control

A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, -1, -3, 3]
])

B = np.array([
    [0], 
    [0], 
    [0],
    [1]
])


print(f'A = \n {A},\n B = \n {B}')
C = control.ctrb(A, B)
print(f'C = \n {C}')
print(f'rank(C) = {np.linalg.matrix_rank(C)}')

Cinv = np.linalg.inv(C)
print(f'Cinv = \n {Cinv}')
Q = np.array([
    Cinv[3, :], 
    Cinv[3, :]@A, 
    Cinv[3, :]@A@A, 
    Cinv[3, :]@A@A@A, 
])

print(f'Q = \n {Q}')

P = np.linalg.inv(Q)
Ac = Q@A@P
Bc = Q@B
Am = np.array([Ac[3, :]])
Bm = 1
Adm = np.array([
    [-10, -19, -13, -5]
])

print(f'Ac = \n {Ac}, \n Bc = \n {Bc},')

Fc = Bm*(Adm-Am)
print(f'Fc = \n {Fc}')
F = Fc@Q
print(f'F = \n {F}')

# desired_coeff = [1, 5, 13, 19, 10]
# alpha_dA = A**4 + 5*A**3 + 13*A**2 + 19*A + 10*np.eye(4) 
# print(f'alpha_dA = \n {alpha_dA}')
# e4T = np.array([[0, 0, 0, 1]])

# F = -e4T@Cinv@alpha_dA
# print(f'F = \n {F}')

print(f'closed loop eigen values: {np.linalg.eig(A+B@F)}')