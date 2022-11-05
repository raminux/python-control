import cvxpy as cp
import numpy as np
import scipy.linalg as linalg
import control



A = np.array([
    [0.2, 0.7, 0, 0],
    [0, 0.4, 0.7, 0],
    [0, 0, 0.5, 0.8],
    [0.7, 0, 0, 0.4]
])
B = np.eye(4)
controllability = np.linalg.matrix_rank(control.ctrb(A, B))
print(f'controllability matrix rank: {controllability}')
Q = 1*np.eye(4)
R = 1*np.eye(4)
Gamma = linalg.block_diag(Q, R)
n, m = B.shape

H11 = cp.Variable((n,n), hermitian=False, symmetric=True)
H12 = cp.Variable((n,m))
H22 = cp.Variable((m, m), hermitian=False, symmetric=True)
W = cp.Variable((n,n), hermitian=False, symmetric=True)
H = cp.bmat([[H11, H12], [H12.T, H22]])
obj = cp.trace(W)
cons = [ cp.bmat([ [H11-W, H12], [H12.T, H22]]) >> 0 ]

# AB = cp.bmat([[A, B]])
AB = np.concatenate((A, B), axis=1)
print(f'AB: \n {AB}')
cons += [
    cp.bmat([ [AB.T@H11@AB-H+Gamma, AB.T@H12], [H12.T@AB, H22] ]) >> 0
]
cons += [ H22 >> 0 ]
cons += [H == cp.bmat([[H11, H12], [H12.T, H22]])]

prob = cp.Problem(cp.Maximize(obj), cons)
# 
print(cp.installed_solvers())
prob.solve(solver=cp.MOSEK, verbose=True)

H22_opt = H22.value
H12_opt = H12.value
K_opt = -np.matmul(np.linalg.inv(H22_opt), np.transpose(H12_opt))
print(f'H22: \n {H22_opt}, \n H12: \n {H12_opt}, \n K_opt: \n {K_opt}')
print(f'optimal cost: {prob.value}, W: {W.value}, H: {H.value}')
# print(vars(prob))