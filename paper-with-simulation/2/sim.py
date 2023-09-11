import numpy as np
import cvxpy as cp


A = np.array([
    [4/5, 1/2],
    [-2/5, 6/5],
])
B = np.array([
    [0],
    [1]
])

n, m = B.shape

# F = np.array([
#     [1/5, 2/5],
#     [-1/5, -2/5], 
#     [-3/20, 1/5], 
#     [3/20, -1/5]
# ])

F = np.array([
    [1, 0],
    [-1, 0], 
    [0, 1], 
    [0, -1]
])

a = 16 # x1 limit
b = 3 # x2 limit

g = np.array([
    [a], 
    [a], 
    [b],
    [b]
])

epsilon = 0.5
epsilon1 = 0.1
epsilon2 = 0.1
epsilon3 = 0.1
epsilon4 = 0.1

Sigma = 0*np.array([
    [0.0100, 0.003],
    [0.003, 0.020]
])

l1 = np.sqrt((1-epsilon1)/epsilon1)*np.sqrt(F[0]@Sigma@F[0].T)
l2 = np.sqrt((1-epsilon2)/epsilon2)*np.sqrt(F[1]@Sigma@F[1].T)
l3 = np.sqrt((1-epsilon3)/epsilon3)*np.sqrt(F[2]@Sigma@F[2].T)
l4 = np.sqrt((1-epsilon4)/epsilon4)*np.sqrt(F[3]@Sigma@F[3].T)

l = np.array([
    [l1], [l2], [l3], [l4]
])

landa = 0.99

error = cp.Variable()

p11 = cp.Variable()
p12 = cp.Variable()
p13 = cp.Variable()
p14 = cp.Variable()
p21 = cp.Variable()
p22 = cp.Variable()
p23 = cp.Variable()
p24 = cp.Variable()
p31 = cp.Variable()
p32 = cp.Variable()
p33 = cp.Variable()
p34 = cp.Variable()
p41 = cp.Variable()
p42 = cp.Variable()
p43 = cp.Variable()
p44 = cp.Variable()
K = cp.Variable((m, n))

P = cp.bmat([
    [p11, p12, p13, p14], 
    [p21, p22, p23, p24],
    [p31, p32, p33, p34],
    [p41, p42, p43, p44]
    ])

obj = error
cons = [P@F-F@(A+B@K) >= -error]
cons += [P@F-F@(A+B@K) <= error]
cons += [P@g - landa*g + l <= 0]
cons += [error >= -0.1]
cons += [error <= 0.1]
cons += [p11 >=0]
cons += [p12 >=0]
cons += [p13 >=0]
cons += [p14 >=0]
cons += [p21 >=0]
cons += [p22 >=0]
cons += [p23 >=0]
cons += [p24 >=0]
cons += [p31 >=0]
cons += [p32 >=0]
cons += [p33 >=0]
cons += [p34 >=0]
cons += [p41 >=0]
cons += [p42 >=0]
cons += [p43 >=0]
cons += [p44 >=0]

prob = cp.Problem(cp.Minimize(obj), cons)
prob.solve(solver=cp.MOSEK, verbose=True)
K_safe = K.value
print(f'K: {K_safe}, \n P: {P.value}, \n error: {error.value}')
ABK = A+B@K_safe
eigs = np.linalg.eig(ABK)
print(eigs[0])
Landa = np.array([
    [eigs[0][0], 0],
    [0, eigs[0][1]]
])
print(f'Landa: {Landa}')
print(f'P*F: {P.value@F}')
print(f'F*Landa: {F@Landa}')
print(f'FTP*F: {F.T@P.value@F}')




