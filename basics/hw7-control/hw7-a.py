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


print(f'A = \n {A},\n B = \n {B}')
C = control.ctrb(A, B)
print(f'C = \n {C}')
print(f'rank(C) = {np.linalg.matrix_rank(C)}')
# print(np.linalg.matrix_rank(C[:, 0:3]))
# print(np.linalg.matrix_rank(C[:, 0:4]))
# print(np.linalg.matrix_rank(C[:, 0:5]))
# print(np.linalg.matrix_rank(C[:, 0:6]))
# print(np.linalg.matrix_rank(C[:, 0:7]))
# print(np.linalg.matrix_rank(C[:, 0:8]))
mu1 = 1
mu2 = 3
Cbar = np.array([
    C[:, 0], C[:, 1], C[:, 3], C[:, 5]
])
print(f'Cbar: {Cbar}')
print(f'rank(Cbar) = {np.linalg.matrix_rank(Cbar)}')
Cbarinv = np.linalg.inv(Cbar)
print(f'Cbarinv = \n {Cbarinv}')
Q = np.array([
    Cbarinv[0, :],
    Cbarinv[3, :],
    Cbarinv[3, :]@A,
    Cbarinv[3, :]@A@A,
])

print(f'Q = \n {Q}')

P = np.linalg.inv(Q)
Ac = Q@A@P
Bc = Q@B
Am = np.array([
    Ac[0, :],
    Ac[3, :]
])
Bm = np.array([
    Bc[0, :],
    Bc[3, :]
])
print(f'Ac = \n {Ac}, \n Bc = \n {Bc},')
print(f'Am = \n {Am}, \n Bm = \n {Bm},')

desired_coeff = [1, 5, 13, 19, 10]
desired_roots = np.roots(desired_coeff)

Adm = np.array([
    [0, 1, 0, 0],
    [-10, -19, -13, -5]
])

Fc = np.linalg.inv(Bm)@(Adm-Am)
print(f'Fc = \n {Fc}')
F = Fc@Q
print(f'F = \n {F}')

print(f'closed loop eigen values: {np.linalg.eig(A+B@F)}')
