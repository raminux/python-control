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
Wx = 1*np.eye(4)
Wu = 1*np.eye(4)
n, m = B.shape

P = cp.Variable((n,n), hermitian=False, symmetric=True)
Y = cp.Variable((m,n))
L = cp.Variable((m, m), hermitian=False, symmetric=True)
gamma = cp.Variable((1, 1))

cons = [ cp.bmat([ [np.eye(n)-P, A@P+B@Y], [P@A.T+Y.T@B.T, -P] ]) << 0 ]
cons += [ P-np.eye(n) >> 0]
cons += [ cp.bmat([ [L, Y], [Y.T, P] ]) >> 0]
cons += [ cp.trace(Wx@P) + cp.trace(Wu@L) << gamma ]

obj = gamma

prob = cp.Problem(cp.Minimize(obj), cons)
prob.solve(solver=cp.MOSEK, verbose=True)

print(f'optimal cost: {prob.value}')
print(f'P: \n {P.value}')
print(f'Y: \n {Y.value}')
print(f'L: \n {L.value}')
print(f'gamma: \n {gamma.value}')

