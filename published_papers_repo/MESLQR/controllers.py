import cvxpy as cp
import numpy as np
import scipy.linalg as linalg
import control
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cmath



def lqr(A, B, Q, R, Z, W, gamma):
    '''
    LQR with discount factor. This function solves the LQR problem using 
    primal-dual appraoch.
    '''

    # State and input dimensions
    n, m = B.shape

    Gamma = linalg.block_diag(Q, R)

    AB = np.concatenate((A, B), axis=1)

    H11 = cp.Variable((n, n), hermitian=False, symmetric=True)
    H12 = cp.Variable((n, m))
    H22 = cp.Variable((m, m), hermitian=False, symmetric=True)
    E = cp.Variable((n, n), hermitian=False, symmetric=True)
    H = cp.bmat([[H11, H12], [H12.T, H22]])

    obj = cp.trace(Z@H)+(gamma/(1-gamma))*cp.trace(W@E)

    cons = [cp.bmat([[H11-E, H12], [H12.T, H22]]) >> 0]
    cons += [
        cp.bmat([[gamma*AB.T@H11@AB-H+Gamma, AB.T@H12], [H12.T@AB, H22/gamma]]) >> 0
    ]
    cons += [H22 >> 0]
    cons += [H == cp.bmat([[H11, H12], [H12.T, H22]])]


    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False)

    H22_opt = H22.value
    H12_opt = H12.value
    H11_opt = H11.value
    F_opt = -np.matmul(np.linalg.inv(H22_opt), np.transpose(H12_opt))


    return F_opt, prob.value


def meslqr(A, B, Q, R, Z, W, Sb, gamma, alpha):
    '''
    MESLQR. This function solves the Maximum Entropy Satisficing LQR problem using 
    primal-dual appraoch.
    '''

    # State and input dimensions
    n, m = B.shape

    Gamma = linalg.block_diag(Q, R)

    AB = np.concatenate((A, B), axis=1)

    H11 = cp.Variable((n, n), hermitian=False, symmetric=True)
    H12 = cp.Variable((n, m))
    H22 = cp.Variable((m, m), hermitian=False, symmetric=True)
    E = cp.Variable((n, n), hermitian=False, symmetric=True)
    H = cp.bmat([[H11, H12], [H12.T, H22]])
    Y = cp.Variable((n+m, n+m), symmetric=True)


    obj = cp.trace(Z@H)+(gamma/(1-gamma))*cp.trace(W@E)-cp.trace(Y@Sb)
    obj += (gamma/(2*(1-gamma))) *alpha*cp.log_det(H22*2/alpha)-alpha*(gamma/(2*(1-gamma)))*(-1+m*np.log(2*np.pi*np.e))

    cons = [cp.bmat([[H11-E, H12], [H12.T, H22]]) >> 0]
    cons += [
        cp.bmat([[gamma*AB.T@H11@AB-H+Gamma+Y, AB.T@H12], [H12.T@AB, H22/gamma]]) >> 0
    ]
    cons += [H22 >> 0]
    cons += [H == cp.bmat([[H11, H12], [H12.T, H22]])]
    cons += [Y >> 0]


    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False)

    H22_opt = H22.value
    H12_opt = H12.value
    H11_opt = H11.value
    F_opt = -np.matmul(np.linalg.inv(H22_opt), np.transpose(H12_opt))
    Cov_opt = (alpha/2)*np.linalg.inv(H22_opt)
    print(f'Y: {Y.value}')
    # Find Primal Cost
    # Solve lyapunov Equation of S
    AF = np.hstack((A, B))
    FAB = np.hstack((F_opt@A, F_opt@B))
    AF = np.vstack((AF, FAB))
    Fbar = np.vstack((np.eye(n), F_opt))
    Lbar = np.vstack((np.zeros((n, m)), np.sqrt(Cov_opt)))
    LQ = Z + (gamma/(1-gamma))*(Fbar@W@Fbar.T +Lbar@Lbar.T)
    S_opt = linalg.solve_discrete_lyapunov(np.sqrt(gamma)*AF, LQ)
    Jp = np.trace(Gamma@S_opt) - alpha*(gamma/(2*(1-gamma)))*(np.log(np.linalg.det(Cov_opt))+m*np.log(2*np.pi*np.e))
    eig_S = np.linalg.eig(Sb-S_opt)
    # Jp = np.trace(Z@H.value)+(gamma/(1-gamma))*np.trace(W@E.value)-np.trace(Y.value@Sb)

    # return F_opt, Cov_opt, prob.value
    return F_opt, Cov_opt, (Jp, prob.value), S_opt, eig_S, Y.value


def safe_control(A, B, M, G, landa):
    '''
    lambda-contractive controller. This function return a safe control gain F for the given 
    polyhedral safe set as defined by Mx <= G.
    '''
    # State, input, and constraint dimensions
    n, m = B.shape
    c, _ = M.shape

    error = cp.Variable()
    F = cp.Variable((m, n))
    P = cp.Variable((c, c), symmetric=False)

    obj = error
    cons = [P@M-M@(A+B@F) >= -error]
    cons += [P@M-M@(A+B@F) <= error]
    cons += [P@G - landa*G <= 0]
    cons += [error >= -0.01]
    cons += [error <= 0.01]
    cons += [P >= 0]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False)
    F_safe = F.value

    return F_safe, prob.value







