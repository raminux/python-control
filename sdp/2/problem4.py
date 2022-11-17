import cvxpy as cp
import numpy as np
import scipy.linalg as linalg
import control
import matplotlib.pyplot as plt


def p4sdp(XD, D, Gamma, n, m):
    H11 = cp.Variable((n,n), hermitian=False, symmetric=True)
    H12 = cp.Variable((n,m))
    H22 = cp.Variable((m, m), hermitian=False, symmetric=True)
    W = cp.Variable((n,n), hermitian=False, symmetric=True)
    H = cp.bmat([[H11, H12], [H12.T, H22]])
    obj = cp.trace(W)
    cons = [ cp.bmat([ [H11-W, H12], [H12.T, H22]]) >> 0 ]
    cons += [H == cp.bmat([[H11, H12], [H12.T, H22]])]
    cons += [ H22 >> 0 ]
    cons += [ cp.bmat([ [XD.T@H11@XD-D.T@(H-Gamma)@D, XD.T@H12], [H12.T@XD, H22] ]) >> 0 ]
    cons += [ XD.T@H11@XD-D.T@(H-Gamma)@D >> 0 ]
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False)
    H22_opt = H22.value
    H12_opt = H12.value
    K_opt = -np.matmul(np.linalg.inv(H22_opt), np.transpose(H12_opt))
    # print(f'H22: \n {H22_opt}, \n H12: \n {H12_opt}, \n K_opt: \n {K_opt}')
    # print(f'optimal cost: {prob.value}, W: {W.value}, H: {H.value}')
    return K_opt, H.value


A_unstable = np.array([
    [0.2, 0.7, 0, 0],
    [0, 0.4, 0.7, 0],
    [0, 0, 0.5, 0.8],
    [0.7, 0, 0, 0.4]
])
A_stable = np.array([
    [0.3, 0.4, 0, 0],
    [0, 0.3, 0.3, 0],
    [0, 0, 0.4, 0.4],
    [0.4, 0, 0, 0.4]
])

A = A_stable
# A = A_unstable

B = np.eye(4)
controllability = np.linalg.matrix_rank(control.ctrb(A, B))
# print(f'controllability matrix rank: {controllability}')
Q = 1*np.eye(4)
R = 1*np.eye(4)
Gamma = linalg.block_diag(Q, R)
n, m = B.shape
l = n+m+1

xk = [np.array([
            [1], [2], [3], [4]
        ])
    ]
uk = [ ]

D = []

# K = [np.array( [[-0.13663838, -0.40314018, -0.0113336,  -0.0119179 ]
#                         ,[-0.00246948, -0.24297699, -0.41892659, -0.02369211]
#                         ,[-0.02973127, -0.01084326, -0.31382405, -0.4853211 ]
#                         ,[-0.42141882, -0.02286701, -0.0204658,  -0.27138098]])]
K = []
for t in range(10):
    
    for i in range(l):
        print(f'---------------------------------{t*l+i}-----------------------------')
        if t == 0:
            u = np.random.normal(0, 1, size=m).reshape((m, 1))
            # u = np.matmul(K[-1], xk[-1]) + np.random.normal(0, 0.01, size=m).reshape((m, 1))
        else:
            u = K[0]@xk[-1] + np.random.normal(0, 0.01, size=m).reshape((m, 1))
            # print(f'u: \n {u}')
        d = np.append(xk[-1], u, axis=0)
        D.append(d)
        x = A@xk[-1] + B@u
        # print(f'x: \n {x}')
        xk.append(x)
        uk.append(u)
        

    # print(len(xk))
    XD = np.transpose(np.asarray(xk[-l:], order='C')[:, :, 0])
    # print(XD.shape)
    Dv = np.transpose(np.asarray(D[-l-1:], order='C')[:, :, 0])
    print(f'rank(D): \n {np.linalg.matrix_rank(Dv)}')
    # print(Dv.shape)
    try:
        _K, H = p4sdp(XD, Dv, Gamma, n, m)
        print(f'_K: \n {_K}')
        K.append(_K)
    except:
        pass

xk = np.asarray(xk)[:, :, 0]
print(xk.shape)
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(12)
plt.subplot(2,2,1)
plt.plot(xk[:, 0])
plt.xlabel('k')
plt.ylabel('x_1')
plt.subplot(2,2,2)
plt.plot(xk[:, 1])
plt.xlabel('k')
plt.ylabel('x_2')
plt.subplot(2,2,3)
plt.plot(xk[:, 2])
plt.xlabel('k')
plt.ylabel('x_3')
plt.subplot(2,2,4)
plt.plot(xk[:, 3])
plt.xlabel('k')
plt.ylabel('x_4')
plt.show()


# print(f'XD: \n {XD}')
# print(f'D: \n{D}')
# print(f'D.shape: {D.shape}')
# print(f'XD.shape: {XD.shape}')

# check the rank of D before solving the SDP for control

