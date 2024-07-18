"""
Ramin Esmzad, Hamidreza Modares, Mechanical Engineering Department, Michigan State University
Simulation file for the Automatica Paper
"Direct Data-Driven Discounted Infinite Horizon Linear Quadratic Regulator with Robustness Guarantees"
"""

import logging
import csv
import cvxpy as cp
import numpy as np
import scipy.linalg as linalg
import control
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy.stats as stats
import scipy.signal as signal



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# print(f'matplotlib styles: {matplotlib.style.available}')
matplotlib_styles = ['Solarize_Light2', '_classic_test_patch',
                     '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic',
                     'dark_background', 'fast', 'fivethirtyeight', 'ggplot',
                     'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright',
                     'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark',
                     'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid',
                     'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
                     'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster',
                     'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white',
                     'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

plt.rcParams['text.usetex'] = True
plt.style.use(matplotlib_styles[12])  # 12, 15, 17, 21, 26, 27


fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=120)
fig_u, ax_u = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=120)
# ax_u = fig_u.add_subplot(111)


def generate_gaussian_noise(snr, signal_length):
    """Generates Gaussian noise with a given SNR.

    Args:
      snr: The desired SNR in dB.
      signal_length: The length of the noise to generate.

    Returns:
      A NumPy array containing the generated noise.
    """

    # Calculate the standard deviation of the noise.
    noise_stddev = np.sqrt(10 ** (-snr / 10))

    # Generate random numbers from a Gaussian distribution with the given standard deviation.
    # noise = np.random.normal(0, noise_stddev, signal_length)

    # return noise, noise_stddev
    return noise_stddev**2


ms, mu = 240, 36  # kg
bs = 980  # N.s/m
ks, kt = 16000, 160000  # N/m

Ac = np.array([
    [0, 1, 0, -1],
    [-ks/ms, -bs/ms, 0, bs/ms],
    [0, 0, 0, 1],
    [ks/mu, bs/mu, -kt/mu, -bs/mu]
])
Bc = np.array([
    [0], [1/ms], [0], [-1/mu]
])


n, m = Bc.shape

Cc = np.eye(n)
Dc = np.zeros((1, 1))

Ts = 0.01  # seconds

num = 150  # number of data points
sims = 200  # number of simulations
t_final = 150
t = np.linspace(0, t_final, num)
logger.info(f't = {t.shape}')

T = 10   # number of collected data vector sets

initials = np.array([
    [0.3, -4, 0.1, -1],
])

X0_cov = 0.0006*np.eye(n)
x0 = np.array([[0.1, 0, 0.1, 0]])

alpha1 = 0.01
landa = 0.01
delta1 = 0.01
landa1 = 0.02
cd1 = 37.43

num_loop = 100

snr = 37
noise_cov = generate_gaussian_noise(snr, 1)
logger.info(f'noise covariance: {noise_cov}, SNR: {snr}')
# 0.007
W = 1*np.array([
    [0.0001, 0.0, 0.0, 0.0],
    [0.0, 0.00001, 0.0, 0.0],
    [0.0, 0.0, noise_cov, 0.0],
    [0.0, 0.0, 0.0, 0.001]
])

Ad, Bd, Cd, Dd, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), Ts)
logger.info(f'Ad: \n{Ad}, \n Bd: \n {Bd}')

delta = 0.9999

R = 0.0001*np.array([[1]])
Q = 1*np.array([
    [30000, 0, 0, 0],
    [0, 30, 0, 0],
    [0, 0, 20, 0],
    [0, 0, 0, 1]
])

logger.info(f'Open loop poles: {np.linalg.eig(Ad)[0]}')


def get_trajectories(A, B, F, W, t, sims, num, X0_cov, initials, Q, R):
    # State and input dimensions
    n, m = B.shape
    mean_cost = np.zeros((t.shape[0], 1))
    x_mean = np.zeros((t.shape[0], n))
    x_min = 10000000*np.ones((t.shape[0], n))
    x_max = -1000000*np.ones((t.shape[0], n))
    u_mean = np.zeros((t.shape[0], m))
    u_min = 100000000*np.ones((t.shape[0], m))
    u_max = -100000000*np.ones((t.shape[0], m))
    for j in range(sims):
        x = np.zeros((t.shape[0], n))
        w = np.zeros((t.shape[0], n))
        u = np.zeros((t.shape[0], m))
        cost = np.zeros((t.shape[0], 1))
        i = 0  # np.random.randint(0, initials.shape[0])
        x[0] = np.random.multivariate_normal(
            [initials[i][0], initials[i][1], initials[i][2], initials[i][3]], X0_cov)
        for k in range(0, num-1):
            u[k] = F@x[k]
            w[k] = np.random.multivariate_normal([0]*n, 1*W)
            cost[k] = x[k].T@Q@x[k] + u[k]*R*u[k]
            x[k+1] = A@x[k]+B@u[k]+w[k]
        x_mean = x_mean + x
        x_min = np.minimum(x_min, x)
        x_max = np.maximum(x_max, x)

        u_mean = u_mean + u
        u_min = np.minimum(u_min, u)
        u_max = np.maximum(u_max, u)

        mean_cost = mean_cost + cost

    return sum(mean_cost/sims)/num, x_mean/sims, x_min, x_max, u_mean/sims, u_min, u_max


def plot_trajectories(A, B, F, W, axs, axu, t, sims, num, X0_cov, initials, Q, R, color, label, trans_alpha):
    # State and input dimensions
    n, m = B.shape
    mean_cost = np.zeros((t.shape[0], 1))
    x_mean = np.zeros((t.shape[0], n))
    x_min = 10000000*np.ones((t.shape[0], n))
    x_max = -1000000*np.ones((t.shape[0], n))
    u_mean = np.zeros((t.shape[0], m))
    u_min = 100000000*np.ones((t.shape[0], m))
    u_max = -100000000*np.ones((t.shape[0], m))
    for j in range(sims):
        x = np.zeros((t.shape[0], n))
        w = np.zeros((t.shape[0], n))
        u = np.zeros((t.shape[0], m))
        cost = np.zeros((t.shape[0], 1))
        i = 0  # np.random.randint(0, initials.shape[0])
        x[0] = np.random.multivariate_normal(
            [initials[i][0], initials[i][1], initials[i][2], initials[i][3]], X0_cov)
        for k in range(0, num-1):
            cost[k] = x[k].T@Q@x[k] + u[k]*R*u[k]

            u[k] = F@x[k]

            if k >= 30 and k <= 60:
                w[k] = np.random.multivariate_normal([0]*n, 1*W)
            else:
                w[k] = [0]*n
            x[k+1] = A@x[k]+B@u[k]+w[k]
        x_mean = x_mean + x
        x_min = np.minimum(x_min, x)
        x_max = np.maximum(x_max, x)

        u_mean = u_mean + u
        u_min = np.minimum(u_min, u)
        u_max = np.maximum(u_max, u)

        x = x.T
        x1 = x[0][:]
        x2 = x[1][:]
        x3 = x[2][:]
        x4 = x[3][:]

        mean_cost = mean_cost + cost
    

    x_mean = x_mean/sims
    x_mean = x_mean.T
    x_min = x_min.T
    x_max = x_max.T
    axs.plot(t, x_mean[0][:], f'{color}', label=f'{label}')
    axs.fill_between(t, x_min[0][:], x_max[0][:],
                     color=f'{color}', alpha=trans_alpha)

    u_mean = u_mean/sims
    u_min = u_min.T
    u_max = u_max.T
    u_min = np.reshape(u_min, (t.shape[0], ))
    u_max = np.reshape(u_max, (t.shape[0], ))
    axu.plot(t, u_mean, f'{color}', label=f'{label}')
    axu.fill_between(t, u_min, u_max, color=f'{color}', alpha=trans_alpha)

    return sum(mean_cost/sims)/num


def collect_data(A, B, W, K, x0, T):
    # State and input dimensions
    n, m = B.shape
    x = np.zeros((T, n))
    w = np.zeros((T, n))
    u = np.zeros((T, m))
    x[0] = x0
    for k in range(0, T-1):
        u[k] = 10*np.random.rand() 
        w[k] = np.random.multivariate_normal([0]*n, 1*W)
        x[k+1] = A@x[k]+B@u[k]+w[k]

    X1 = x[1:].T
    X0 = x[0:-1].T
    U0 = u[0:-1].T
    XU0 = np.vstack([X0, U0])
    logger.info(f'Data rank: {np.linalg.matrix_rank(XU0)}')

    XU0 = np.vstack([X0, U0])
    AhBh = X1@np.linalg.pinv(XU0)
    Ah = AhBh[:, 0:n]
    Bh = AhBh[:, n:]

    Wh = np.zeros((n, n))
    for i in range(0, T-1):
        wh = x[i+1, :] - (Ah@x[i, :] + Bh@u[i, :])
        wh = wh.reshape(n, 1)
        Wh += wh@wh.T
    Wh = Wh/(T)
    return X1, X0, U0, Ah, Bh, Wh, AhBh


def estimate_ABh_D(X0, U0, W, delta=0.1, landa=0, cd=7.43):

    c2w = W[0, 0]
    n, T = X0.shape
    m, T = U0.shape
    I = np.eye(m+n)

    D = landa*I
    ABh = np.zeros((n, m+n))
    for i in range(0, T-1):
        Z = np.hstack([X0[:, i].T, U0[:, i].T]).T
        Z = Z.reshape(m+n, 1)
        D = D + Z@Z.T
        
    for i in range(0, T-1):
        Z = np.hstack([X0[:, i].T, U0[:, i].T]).T
        Z = Z.reshape(m+n, 1)
        ABh = ABh + X0[:, i+1].reshape(n, 1)@Z.T
    D = D/(cd*c2w)
    ABh = ABh@np.linalg.inv(D)

    return ABh, D

def indirect_single_trajectory_LQR(X0, U0, Q, R, W, ABhh, Wh):

    # State and input dimensions
    m, T = U0.shape
    n, T = X0.shape

    ABh, D = estimate_ABh_D(X0, U0, 1*W, delta=delta1, landa=landa1, cd=cd1)
    I = np.eye(n)
    Sigmaxx = cp.Variable((n, n), symmetric=True)
    Sigmaxu = cp.Variable((n, m))
    Sigmauu = cp.Variable((m, m), symmetric=True)
    Sigma = cp.bmat([[Sigmaxx, Sigmaxu], [Sigmaxu.T, Sigmauu]])
    t = cp.Variable((1, 1))
    
    obj = cp.trace(linalg.block_diag(Q, R)@Sigma)

    cons = [Sigma >> 0]
    cons += [t >= 0]
    cons += [Sigma == cp.bmat([[Sigmaxx, Sigmaxu], [Sigmaxu.T, Sigmauu]])]

    cons += [
        cp.bmat([
            [I, W, np.zeros((n, n+m))],
            [W, Sigmaxx-ABh@Sigma@ABh.T-t*I, ABh@Sigma],
            [np.zeros((n+m, n)), (ABh@Sigma).T, t*D-Sigma]]) >> 0
    ]

    prob = cp.Problem(cp.Minimize(obj), cons)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'problem status: {prob.status}')
        return None

    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None


    K = Sigmaxu.value.T@np.linalg.inv(Sigmaxx.value)
    logger.info(
        f'Data based Robust LQR closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K


def data_driven_low_complexity_lqr(X0, X1, U0, Q, R, alpha):
    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    I = np.eye(n)
    P = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))
    L = cp.Variable((m, m))
    V = cp.Variable((T, T), symmetric=True)
    gamma = cp.Variable((1, 1))

    obj = gamma

    cons = [gamma >= 0]
    cons += [P >> I]
    cons += [L >> 0]
    cons += [V >> 0]
    cons += [cp.bmat([
        [I-P, X1@F],
        [(X1@F).T, -P]
    ]) << 0]
    cons += [cp.bmat([
        [L, U0@F],
        [(U0@F).T, P]
    ]) >> 0]
    cons += [cp.bmat([
        [V, F],
        [F.T, P]
    ]) >> 0]
    cons += [X0@F == P]
    cons += [cp.trace(Q@P) + cp.trace(R@L) + alpha*cp.trace(V) <= gamma]
    # cons += [cp.trace(Q@P) + cp.trace(R@L) <= gamma]

    prob = cp.Problem(cp.Minimize(obj), cons)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'LCLQR problem status: {prob.status}')
        return None, None

    # "optimal_inaccurate",
    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None

    K = U0@F.value@np.linalg.inv(P.value),
    logger.info(
        f'Data based Low Complexity closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P.value


def data_driven_promoting_ce_lqr(X0, X1, U0, Q, R, landa):
    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    I = np.eye(n)

    P = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))
    L = cp.Variable((m, m))
    gamma = cp.Variable((1, 1))

    obj = gamma

    D0 = np.vstack([U0, X0])
    Pi = np.eye(T) - np.linalg.pinv(D0)@D0

    cons = [gamma >= 0]
    cons += [P >> I]
    cons += [L >> 0]
    cons += [cp.bmat([
        [I-P, X1@F],
        [(X1@F).T, -P]
    ]) << 0]
    cons += [cp.bmat([
        [L, U0@F],
        [(U0@F).T, P]
    ]) >> 0]

    cons += [X0@F == P]
    cons += [cp.trace(Q@P) + cp.trace(R@L) + landa*cp.norm(Pi@F) <= gamma]

    prob = cp.Problem(cp.Minimize(obj), cons)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'PCE problem status: {prob.status}')
        return None, None

    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None

    K = U0@F.value@np.linalg.inv(P.value),
    logger.info(
        f'Data based Promoting CE LQR closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P.value


def data_driven_regularized_lqr(X0, X1, U0, Q, R, landa, alpha):
    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    I = np.eye(n)

    P = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))
    L = cp.Variable((m, m))
    V = cp.Variable((T, T), symmetric=True)
    gamma = cp.Variable((1, 1))

    obj = gamma

    D0 = np.vstack([U0, X0])
    Pi = np.eye(T) - np.linalg.pinv(D0)@D0

    cons = [gamma >= 0]
    cons += [P >> I]
    cons += [L >> 0]
    cons += [cp.bmat([
        [I-P, X1@F],
        [(X1@F).T, -P]
    ]) << 0]
    cons += [cp.bmat([
        [L, U0@F],
        [(U0@F).T, P]
    ]) >> 0]
    cons += [V >> 0]
    cons += [cp.bmat([
        [V, F],
        [F.T, P]
    ]) >> 0]
    cons += [X0@F == P]
    cons += [cp.trace(Q@P) + cp.trace(R@L) + landa *
             cp.norm(Pi@F) + alpha*cp.trace(V) <= gamma]
    # cons += [cp.trace(Q@P) + cp.trace(R@L) <= gamma]

    prob = cp.Problem(cp.Minimize(obj), cons)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'Regularized problem status: {prob.status}')
        return None, None

    # "optimal_inaccurate",
    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None

    K = U0@F.value@np.linalg.inv(P.value),
    logger.info(
        f'Data based Regularized LQR closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P.value


def low_complexity_lqr(A, B, Q, R):
    # State and input dimensions
    n, m = B.shape
    I = np.eye(n)
    P = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    L = cp.Variable((m, m))
    gamma = cp.Variable((1, 1))

    obj = gamma

    cons = [gamma >= 0]
    cons += [P >> I]
    cons += [cp.bmat([
        [I-P, A@P+B@Y],
        [(A@P+B@Y).T, -P]
    ]) << 0]
    cons += [cp.bmat([
        [L, Y],
        [Y.T, P]
    ]) >> 0]
    cons += [cp.trace(Q@P) + cp.trace(R*L) <= gamma]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False)

    return Y.value@np.linalg.inv(P.value), P.value


def model_based_Lyapunov_LQR(A, B, W, Q, R, delta):

    # State and input dimensions
    n, m = B.shape
    I = np.eye(n)
    Y = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((m, n))
    obj = cp.trace(np.linalg.inv(W)@Y)
    # obj = cp.tr_inv(Y) #np.linalg.inv(W)@

    cons = [Y >> 0]
    cons += [
        cp.bmat([[-Y, Y, (F).T, (A@Y+B@F).T],
                 [Y, -np.linalg.inv(Q), np.zeros((n, m)), np.zeros((n, n))],
                 [F, np.zeros((m, n)), -1/R, np.zeros((m, n))],
                 [(A@Y+B@F), np.zeros((n, n)), np.zeros((n, m)), -Y/delta],

                 ]) << 0
    ]

    # prob = cp.Problem(cp.Minimize(obj), cons)
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'CE problem status: {prob.status}')
        return None, None

    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None
    P = np.linalg.inv(Y.value)
    K = F.value@P
    logger.info(f'CE Ad+BdK: \n {np.linalg.eig(Ad+Bd@K)[0]}')
    return K, P


def indirect_data_driven_dynamic_programming_LQR(X1, X0, U0, W, Q, R, delta):

    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    D0 = np.vstack([U0, X0])
    I = np.eye(n)
    I_T = np.eye(T)
    Y = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))

    obj = cp.trace(np.linalg.inv(W)@Y)

    cons = [Y >> 0]

    cons += [
        cp.bmat([[Y, Y, (U0@F).T, (X1@F).T],
                 [Y, np.linalg.inv(Q), np.zeros((n, m)),
                  np.zeros((n, n))],
                 [U0@F, np.zeros((m, n)), 1/R,
                  np.zeros((m, n))],
                 [X1@F, np.zeros((n, n)), np.zeros((n, m)),
                  Y/delta],
                 ]) >> 0
    ]

    cons += [X0@F == Y]

    cons += [(I_T-np.linalg.pinv(D0)@D0)@F == 0]

    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'problem status: {prob.status}')
        return None, None, None, None

    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None, None, None

    G = F.value@np.linalg.inv(Y.value)
    K = U0@G
    P = np.linalg.inv(Y.value)

    logger.info(
        f'Date based Dynamic Prog closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P, G, obj.value


def data_driven_dynamic_programming_LQR(X1, X0, U0, W, Q, R, delta):

    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    D0 = np.vstack([U0, X0])
    I = np.eye(n)
    Y = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))
    alpha = cp.Variable((1, 1))

    obj = alpha

    cons = [Y >> 0]

    cons += [
        cp.bmat([[-Y, Y, (U0@F).T, (X1@F).T, F.T],
                 [Y, -np.linalg.inv(Q), np.zeros((n, m)),
                  np.zeros((n, n)), np.zeros((n, T))],
                 [U0@F, np.zeros((m, n)), -1/R,
                  np.zeros((m, n)), np.zeros((m, T))],
                 [X1@F, np.zeros((n, n)), np.zeros((n, m)),
                  -Y/delta, np.zeros((n, T))],
                 [F, np.zeros((T, n)), np.zeros((T, m)), np.zeros(
                     (T, n)), -alpha*np.eye(T)/(delta)],
                 ]) << 0
    ]

    cons += [cp.trace(np.linalg.inv(1*W)@Y)-alpha*n*n >= 0]
    cons += [X0@F == Y]
    cons += [alpha >= 0.0]

    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'DDDP problem status: {prob.status}')
        return None, None, None, None, None

    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None, None, None, None

    G = F.value@np.linalg.inv(Y.value)
    K = U0@G
    P = np.linalg.inv(Y.value)

    # logger.info(f'Tr(PW): {np.trace(P@W)}, 1/alpha: {1/alpha1.value}')
    logger.info(
        f'Date based Dynamic Prog closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P, G, alpha.value, obj.value


def plot_run(t, u_mean, u_min, u_max, x_mean, x_min, x_max, color, label, alpha, i):

    u_mean = u_mean/i
    u_min = u_min.T
    u_max = u_max.T
    u_min = np.reshape(u_min, (t.shape[0], ))
    u_max = np.reshape(u_max, (t.shape[0], ))
    ax_u.plot(t, u_mean, f'{color}', label=f'{label}')
    ax_u.fill_between(t, u_min, u_max, color=f'{color}', alpha=alpha)

    x_mean = x_mean/i
    x_mean = x_mean.T
    x_min = x_min.T
    x_max = x_max.T
    axs.plot(t, x_mean[0][:], f'{color}', label=f'{label}')
    axs.fill_between(t, x_min[0][:], x_max[0][:],
                     color=f'{color}', alpha=alpha)


def run():

    i = 0
    i_our = 0
    i_low = 0
    i_robust = 0
    i_pce = 0
    i_regularized = 0
    i_ce = 0
    n, m = Bd.shape

    mean_cost_our = np.zeros((t.shape[0], 1))
    mean_cost_low = np.zeros((t.shape[0], 1))
    mean_cost_robust = np.zeros((t.shape[0], 1))
    mean_cost_pce = np.zeros((t.shape[0], 1))
    mean_cost_regularized = np.zeros((t.shape[0], 1))
    mean_cost_ce = np.zeros((t.shape[0], 1))

    fail_our = 0
    fail_low = 0
    fail_robust = 0
    fail_pce = 0
    fail_regularized = 0
    fail_ce = 0

    x_mean_our = np.zeros((t.shape[0], n))
    x_min_our = 10000000*np.ones((t.shape[0], n))
    x_max_our = -1000000*np.ones((t.shape[0], n))

    u_mean_our = np.zeros((t.shape[0], m))
    u_min_our = 10000*np.ones((t.shape[0], m))
    u_max_our = -10000*np.ones((t.shape[0], m))

    x_mean_low = np.zeros((t.shape[0], n))
    x_min_low = 10000000*np.ones((t.shape[0], n))
    x_max_low = -1000000*np.ones((t.shape[0], n))

    u_mean_low = np.zeros((t.shape[0], m))
    u_min_low = 10000*np.ones((t.shape[0], m))
    u_max_low = -10000*np.ones((t.shape[0], m))

    x_mean_robust = np.zeros((t.shape[0], n))
    x_min_robust = 10000000*np.ones((t.shape[0], n))
    x_max_robust = -1000000*np.ones((t.shape[0], n))

    u_mean_robust = np.zeros((t.shape[0], m))
    u_min_robust = 10000*np.ones((t.shape[0], m))
    u_max_robust = -10000*np.ones((t.shape[0], m))

    x_mean_pce = np.zeros((t.shape[0], n))
    x_min_pce = 10000000*np.ones((t.shape[0], n))
    x_max_pce = -1000000*np.ones((t.shape[0], n))

    u_mean_pce = np.zeros((t.shape[0], m))
    u_min_pce = 10000*np.ones((t.shape[0], m))
    u_max_pce = -10000*np.ones((t.shape[0], m))

    x_mean_regularized = np.zeros((t.shape[0], n))
    x_min_regularized = 10000000*np.ones((t.shape[0], n))
    x_max_regularized = -1000000*np.ones((t.shape[0], n))

    u_mean_regularized = np.zeros((t.shape[0], m))
    u_min_regularized = 10000*np.ones((t.shape[0], m))
    u_max_regularized = -10000*np.ones((t.shape[0], m))

    x_mean_ce = np.zeros((t.shape[0], n))
    x_min_ce = 10000000*np.ones((t.shape[0], n))
    x_max_ce = -1000000*np.ones((t.shape[0], n))

    u_mean_ce = np.zeros((t.shape[0], m))
    u_min_ce = 10000*np.ones((t.shape[0], m))
    u_max_ce = -10000*np.ones((t.shape[0], m))

    eK_our = 0
    eK_low = 0
    eK_robust = 0
    eK_pce = 0
    eK_regularized = 0
    eK_ce = 0

    alpha_sum = 0
    cost_sum_our = 0
    cost_sum_low = 0
    cost_sum_robust = 0
    cost_sum_pce = 0
    cost_sum_regularized = 0
    cost_sum_ce = 0

    while i <= num_loop:

        logger.info(f'i: {i}')
        logger.info(f'i_our: {i_our}')
        logger.info(f'i_low: {i_low}')
        logger.info(f'i_robust: {i_robust}')
        logger.info(f'i_pce: {i_pce}')
        logger.info(f'i_regularized: {i_regularized}')
        logger.info(f'i_ce: {i_ce}')
        # collect data
        X1, X0, U0, Ah, Bh, Wh, ABh = collect_data(Ad, Bd, W, _K, x0, T)

        random.seed(i)

        # desgin controller
        K_ce, P_ce = model_based_Lyapunov_LQR(Ah, Bh, W, Q, R, delta)

        if K_ce is None:
            fail_ce += 1
        else:
            i_ce += 1
            cost, x, _x_min, _x_max, u, _u_min, _u_max = get_trajectories(
                Ad, Bd, K_ce, W, t, sims, num, X0_cov, initials, Q, R)

            cost_sum_ce = cost_sum_ce + cost
            x_mean_ce = x_mean_ce + x
            x_min_ce = np.minimum(x_min_ce, _x_min)
            x_max_ce = np.maximum(x_max_ce, _x_max)

            u_mean_ce = u_mean_ce + u
            u_min_ce = np.minimum(u_min_ce, _u_min)
            u_max_ce = np.maximum(u_max_ce, _u_max)

            eK_ce = eK_ce + np.linalg.norm(K_ce-_K)

        K_our, P, G, alpha_dd, obj = data_driven_dynamic_programming_LQR(
            X1, X0, U0, W, Q, R, delta)

        K_low, P = data_driven_low_complexity_lqr(X0, X1, U0, Q, R, alpha1)

        K_pce, P = data_driven_promoting_ce_lqr(X0, X1, U0, Q, R, landa)

        if K_pce is None:
            fail_pce += 1
        else:
            i_pce += 1
            cost, x, _x_min, _x_max, u, _u_min, _u_max = get_trajectories(
                Ad, Bd, K_pce, W, t, sims, num, X0_cov, initials, Q, R)

            cost_sum_pce = cost_sum_pce + cost
            x_mean_pce = x_mean_pce + x
            x_min_pce = np.minimum(x_min_pce, _x_min)
            x_max_pce = np.maximum(x_max_pce, _x_max)

            u_mean_pce = u_mean_pce + u
            u_min_pce = np.minimum(u_min_pce, _u_min)
            u_max_pce = np.maximum(u_max_pce, _u_max)

            eK_pce = eK_pce + np.linalg.norm(K_pce-_K)

        K_regularized, P = data_driven_regularized_lqr(
            X0, X1, U0, Q, R, landa, alpha1)

        if K_regularized is None:
            fail_regularized += 1
        else:
            i_regularized += 1
            cost, x, _x_min, _x_max, u, _u_min, _u_max = get_trajectories(
                Ad, Bd, K_regularized, W, t, sims, num, X0_cov, initials, Q, R)

            cost_sum_regularized = cost_sum_regularized + cost
            x_mean_regularized = x_mean_regularized + x
            x_min_regularized = np.minimum(x_min_regularized, _x_min)
            x_max_regularized = np.maximum(x_max_regularized, _x_max)

            u_mean_regularized = u_mean_regularized + u
            u_min_regularized = np.minimum(u_min_regularized, _u_min)
            u_max_regularized = np.maximum(u_max_regularized, _u_max)

            eK_regularized = eK_regularized + np.linalg.norm(K_regularized-_K)

        K_robust = indirect_single_trajectory_LQR(X0, U0, Q, R, W, ABh, Wh)

        #
        if K_low is None:
            fail_low += 1
        else:
            i_low += 1
            cost, x, _x_min, _x_max, u, _u_min, _u_max = get_trajectories(Ad, Bd, K_low, W, t, sims,
                                                                          num, X0_cov, initials, Q, R)

        # ifU = u > 10000
        # ifUmin = _u_min < -10000
        # ifUmax = _u_max > 10000
        # if ifU.any() or ifUmin.any() or ifUmax.any():
        #     fail_low += 1
        #     continue

            cost_sum_low = cost_sum_low + cost
            x_mean_low = x_mean_low + x
            x_min_low = np.minimum(x_min_low, _x_min)
            x_max_low = np.maximum(x_max_low, _x_max)

            u_mean_low = u_mean_low + u
            u_min_low = np.minimum(u_min_low, _u_min)
            u_max_low = np.maximum(u_max_low, _u_max)

            eK_low = eK_low + np.linalg.norm(K_low-_K)

        # simulate
        if K_our is None:
            fail_our += 1
        else:
            i_our += 1
            cost, x, _x_min, _x_max, u, _u_min, _u_max = get_trajectories(Ad, Bd, K_our, W, t, sims,
                                                                          num, X0_cov, initials, Q, R)
            cost_sum_our = cost_sum_our + cost
            x_mean_our = x_mean_our + x
            x_min_our = np.minimum(x_min_our, _x_min)
            x_max_our = np.maximum(x_max_our, _x_max)

            u_mean_our = u_mean_our + u
            u_min_our = np.minimum(u_min_our, _u_min)
            u_max_our = np.maximum(u_max_our, _u_max)

            eK_our = eK_our + np.linalg.norm(K_our-_K)
            alpha_sum = alpha_sum + alpha_dd

        if K_robust is None:
            fail_robust += 1
        else:
            i_robust += 1
            cost, x, _x_min, _x_max, u, _u_min, _u_max = get_trajectories(Ad, Bd, K_robust, W, t, sims,
                                                                          num, X0_cov, initials, Q, R)
            cost_sum_robust = cost_sum_robust + cost
            x_mean_robust = x_mean_robust + x
            x_min_robust = np.minimum(x_min_robust, _x_min)
            x_max_robust = np.maximum(x_max_robust, _x_max)

            u_mean_robust = u_mean_robust + u
            u_min_robust = np.minimum(u_min_robust, _u_min)
            u_max_robust = np.maximum(u_max_robust, _u_max)

            eK_robust = eK_robust + np.linalg.norm(K_robust-_K)

        i = i + 1

    if i_our > 0:
        plot_run(t, u_mean_our, u_min_our, u_max_our, x_mean_our,
                 x_min_our, x_max_our, 'g', 'Our method', 0.4, i_our)
        logger.info(f'Our LQR alpha: {alpha_sum/i_our}')
        logger.info(f'Our LQR  K-K*: {eK_our/i_our}')
        logger.info(f'Our LQR  cost: {cost_sum_our/i_our}')
        logger.info(f'Our LQR  fail: {fail_our}')

    if i_ce > 0:
        plot_run(t, u_mean_ce, u_min_ce, u_max_ce, x_mean_ce,
                 x_min_ce, x_max_ce, 'orange', 'CE method', 0.7, i_ce)
        logger.info(f'CE  K-K*: {eK_ce/i_ce}')
        logger.info(f'CE  cost: {cost_sum_ce/i_ce}')
        logger.info(f'CE  fail: {fail_ce}')

    if i_low > 0:
        # tex = r'LCLQR, $\alpha$' + f'\,=\,{alpha1}'
        tex = f'LCLQR, alpha={alpha1}'
        plot_run(t, u_mean_low, u_min_low, u_max_low, x_mean_low,
                 x_min_low, x_max_low, 'b', tex, 0.3, i_low)
        logger.info(f'LCLQR  K-K*: {eK_low/i_low}')
        logger.info(f'LCLQR  cost: {cost_sum_low/i_low}')
        logger.info(f'LCLQR  fail: {fail_low}')

    if i_robust > 0:
        # tex = r'RLQR, $c_\delta$' + f'\,=\,{cd1}'
        tex = f'RLQR, cdelta = {cd1}'
        plot_run(t, u_mean_robust, u_min_robust, u_max_robust, x_mean_robust,
                 x_min_robust, x_max_robust, 'r', tex, 0.3, i_robust)
        logger.info(f'RLQR  K-K*: {eK_robust/i_robust}')
        logger.info(f'RLQR  cost: {cost_sum_robust/i_robust}')
        logger.info(f'RLQR  fail: {fail_robust}')

    if i_pce > 0:
        # tex = r'PCELQR, $\lambda$' + f'\,=\,{landa}'
        tex = f'PCELQR, lambda={landa}'
        plot_run(t, u_mean_pce, u_min_pce, u_max_pce, x_mean_pce,
                 x_min_pce, x_max_pce, 'black', tex, 0.3, i_pce)
        logger.info(f'PCE  K-K*: {eK_pce/i_pce}')
        logger.info(f'PCE  cost: {cost_sum_pce/i_pce}')
        logger.info(f'PCE  fail: {fail_pce}')

    if i_regularized > 0:
        # tex = r'Regularized, $\lambda$' + \
            # f'\,=\,{landa},' + r'$\alpha$' + f'\,=\,{alpha1}'
        tex = f'Regularized, lambda' + \
            f'={landa}, alpha={alpha1}'
        plot_run(t, u_mean_regularized, u_min_regularized, u_max_regularized, x_mean_regularized,
                 x_min_regularized, x_max_regularized, 'purple', tex, 0.3, i_regularized)
        logger.info(f'Regularized  K-K*: {eK_regularized/i_regularized}')
        logger.info(f'Regularized  cost: {cost_sum_regularized/i_regularized}')
        logger.info(f'Regularized  fail: {fail_regularized}')


_K, _P, _ = control.dlqr((np.sqrt(0.99999))*Ad, (np.sqrt(0.99999))*Bd, Q, R)
_K = -_K


run()




logger.info(f'Model Based LQR K: {_K}')

cost = plot_trajectories(Ad, Bd, _K, 1*W, axs, ax_u, t, sims,
                         num, X0_cov, initials, Q, R, 'r', "Model-Based LQR", 0.2)
logger.info(f'Model Based LQR cost: {cost}')

plt.show()




