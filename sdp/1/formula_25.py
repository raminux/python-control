import cvxpy as cp 
import numpy as np
import scipy.linalg as linalg
import control
import matplotlib.pyplot as plt


def f25(X1, X0, U0, Wx, Wu, alpha, n, m, T):
    '''
    Search Python Documentation @TODO
    '''

    P = cp.Variable((n, n), hermitian=False, symmetric=True)
    Q = cp.Variable((T, n), hermitian=False, symmetric=False)
    L = cp.Variable((m,m), hermitian=False, symmetric=True)
    V = cp.Variable((T, T), hermitian=False, symmetric=True)
    gamma = cp.Variable((1,1))

    obj = gamma

    cons = [ P - np.eye(n) >> 0 ]
    cons += [ X0@Q == P ]
    cons += [ cp.trace(Wx@P) + cp.trace(Wu@L) + alpha*cp.trace(V) << gamma ]
    cons += [ cp.bmat([ [np.eye(n)-P, X1@Q], [Q.T@X1.T, -P] ])  << 0]
    cons += [ cp.bmat([ [L, U0@Q], [Q.T@U0.T, P] ])  >> 0]
    cons += [ cp.bmat([ [V, Q], [Q.T, P] ])  >> 0]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False)

    P_opt = P.value
    Q_opt = Q.value
    K_opt = np.matmul(U0, np.matmul(Q_opt, np.linalg.inv(P_opt)))
    # print(f'K_opt: \n {K_opt}')
    return K_opt, P_opt


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
Wx = 1*np.eye(4)
Wu = 1*np.eye(4)
alpha = 0.1
n, m = B.shape
T = n+m+1

xk = [np.array([
            [1], [2], [3], [4]
        ])
    ]
uk = [  ] #np.array([ [0], [0], [0], [0] ])



D = []

K = [np.array( [[-0.13663838, -0.40314018, -0.0113336,  -0.0119179 ]
                        ,[-0.00246948, -0.24297699, -0.41892659, -0.02369211]
                        ,[-0.02973127, -0.01084326, -0.31382405, -0.4853211 ]
                        ,[-0.42141882, -0.02286701, -0.0204658,  -0.27138098]])]
# K = []

for t in range(12):
    
    for i in range(T):
        print(f'---------------------------------{t*T+i}-----------------------------')
        if t == 0:
            u = np.random.normal(0, 5, size=m).reshape((m, 1)) #
            # u = np.ones((4, 1)) #
            # u = 0.9*np.random.randn(m, 1)
            # u = np.matmul(K[-1], xk[-1]) + np.random.normal(0, 0.1, size=m).reshape((m, 1))
        else:
            u = K[1]@xk[-1] + np.random.normal(0, 0.0, size=m).reshape((m, 1))
            # print(f'u: \n {u}')
        d = np.append(xk[-1], u, axis=0)
        D.append(d)
        x = A@xk[-1] + B@u
        # print(f'x: \n {x}')
        xk.append(x)
        uk.append(u)
        

    # print(len(xk))
    # print(f'len(xk): \n {len(xk)}')
    X1 = np.asarray(xk[-T:], order='C')[:, :, 0].T
    X0 = np.asarray(xk[-T-1:-1], order='C')[:, :, 0].T
    U0 = np.asarray(uk[-T-1:], order='C')[:, :, 0].T
    # print(f'len(X1): {len(X1)}, len(X0): {len(X0)}, len(U0): {len(U0)}')
    print(f'shape(X1): {X1.shape}, shape(X0): {X0.shape}, shape(U0): {U0.shape}')
    
    # print(f'X1.shape: \n {X1.shape}')
    # print(f'X1: \n {X1}')
    # print(f'X0: \n {X0}')
    # print(f'D[-T]: \n {D[-T]}')
    # print(f'D[-T-1]: \n {D[-T-1]}')
    # print(XD.shape)
    Dv = np.asarray(D[-T:], order='C')[:, :, 0].T
    # print(f'Dv.shape: \n {Dv.shape}')
    # print(f'len(D): \n {len(D)}')
    # print(f'rank(D): \n {np.linalg.matrix_rank(Dv)}')
    # print(f'D: \n {Dv}')

    # print(f'xk: \n {xk}')

    AB = X1@np.linalg.pinv(Dv)
    # AB = X1@linalg.pinv(Dv)
    print(f'AB: \n {AB}')

    uu, ss, vv = np.linalg.svd(Dv)
    # print(f'ss: \n {ss}')

    # print(Dv.shape)
    try:
        _K, H = f25(X1, X0, U0, Wx, Wu, alpha, n, m, T)
        print(f'_K: \n {_K}')
        K.append(_K)
    except:
        pass

xk = np.asarray(xk)[:, :, 0]
# print(xk.shape)
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
