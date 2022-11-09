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
H = cp.Variable((n+m,n+m), hermitian=False, symmetric=True)
W = cp.Variable((n,n), hermitian=False, symmetric=True)

obj = cp.trace(W)
W_a = cp.bmat([[W, np.zeros((n,m))], [np.zeros((m, n+m))]])
cons = [H-W_a >> 0]

AB = cp.bmat([[A, B, np.zeros((n, m))]])
ZI = cp.bmat([[np.zeros((m, n+m)), np.eye(m)]])
ABI = cp.bmat([[AB], [ZI]])
GammaH = cp.bmat([[Gamma-H, np.zeros((m+n, m))], [np.zeros((m, 2*m+n))]])
cons += [ABI.T@H@ABI + GammaH >> 0 ]

prob = cp.Problem(cp.Maximize(obj), cons)
# 
print(cp.installed_solvers())
prob.solve(solver=cp.MOSEK, verbose=True)

H_opt = H.value
H22 = np.array(H_opt[n:, n:])
H12 = np.array(H_opt[0:n, n:])
print(f'H22: \n {H22} \n, H12: \n {H12}')
K_opt = -np.matmul(np.linalg.inv(H22), np.transpose(H12))
print(f'optimal cost: {prob.value}, \n W: \n {W.value}, \n H: \n {H.value}, \n K: \n {K_opt}')
# print(vars(prob))





