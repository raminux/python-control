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
# fig_u, ax_u = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=120)


# Define system parameters
# np.random.seed(42)  # Seed for reproducibility
Rm = 8.4        # Resistance (Ohm)
kt = 0.042      # Torque constant (N-m/A)
km = 0.042      # Back-emf constant (V-s/rad)
mr = 0.095      # Rotary arm mass (kg)
r = 0.085       # Rotary arm length (m)
Jr = mr * r**2 / 3  # Rotary arm inertia (kg-m^2)
br = 1e-3       # Rotary arm damping (N-m-s/rad)
mp = 0.024      # Pendulum mass (kg)
Lp = 0.129      # Pendulum length (m)
l = Lp / 2      # Center of mass (m)
Jp = mp * Lp**2 / 3  # Pendulum inertia (kg-m^2)
bp = 5e-5       # Pendulum damping (N-m-s/rad)
g = 9.81        # Gravity (m/s^2)
Jt = Jp * Jr - (mp * l * r)**2  # Total inertia

# Continuous-time A and B matrices
Ac = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (mp*mp)*(l*l)*r*g/Jt,  -br*Jp/Jt- km*km/Rm*Jp/Jt,   -mp*l*r*bp/Jt],
    [0, mp*g*l*Jr/Jt,    -mp*l*r*br/Jt- km*km/Rm*mp*l*r/Jt,   -Jr*bp/Jt]
])

Bc = np.array([
    [0],
    [0],
    [km * Jp / (Jt * Rm)],
    [km * mp * l * r / (Jt * Rm)]
])

# Discretize the system
Ts = 0.05 # Sampling time

u_dist = 0

system_cont = control.ss(Ac, Bc, np.eye(4), np.zeros((4, 1)))
system_disc = control.c2d(system_cont, Ts, method='zoh')

Ad = np.array(system_disc.A)
Bd = np.array(system_disc.B)



# ms, mu = 240, 36  # kg
# bs = 980  # N.s/m
# ks, kt = 16000, 160000  # N/m

# Ac = np.array([
#     [0, 1, 0, -1],
#     [-ks/ms, -bs/ms, 0, bs/ms],
#     [0, 0, 0, 1],
#     [ks/mu, bs/mu, -kt/mu, -bs/mu]
# ])
# Bc = np.array([
#     [0], [1/ms], [0], [-1/mu]
# ])


n, m = Bc.shape

# Cc = np.eye(n)
# Dc = np.zeros((4, 1))
# Ad, Bd, Cd, Dd, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), Ts)



W = 0.001*np.array([
    [0.0001, 0.0, 0.0, 0.0],
    [0.0, 0.0001, 0.0, 0.0],
    [0.0, 0.0, 0.0001, 0.0],
    [0.0, 0.0, 0.0, 0.0001]
])

V = 0.002*np.array([
    [0.01, 0.0, 0.0, 0.0],
    [0.0, 0.01, 0.0, 0.0],
    [0.0, 0.0, 0.01, 0.0],
    [0.0, 0.0, 0.0, 0.01]
])

Ivw = np.eye(n)

Wh = np.zeros((n, n))
def collect_data(A, B, W, V, K, x0, N, Wh):
    # State and input dimensions
    n, m = B.shape
    x = np.zeros((N, n))
    y = np.zeros((N, n))
    w = np.zeros((N, n))
    u = np.zeros((N, m))
    x[0] = x0
    y[0] = x0 + np.random.multivariate_normal([0]*n, 1*V)
    Y1 = np.zeros((n, N-1))
    Y0 = np.zeros((n, N-1))
    U0 = np.zeros((m, N-1))
    X1 = np.zeros((n, N-1))
    X0 = np.zeros((n, N-1))
    TT = 1
    for j in range(0, TT):
        x = np.zeros((N, n))
        y = np.zeros((N, n))
        w = np.zeros((N, n))
        u = np.zeros((N, m))
        x[0] = x0
        y[0] = x0 + np.random.multivariate_normal([0]*n, 1*V)
        for k in range(0, N-1):
            u[k] = K@x[k] + 0.45*np.random.randn() 
        
            w[k] = np.random.multivariate_normal([0]*n, 0*W)
            x[k+1] = A@x[k]+B@u[k]+w[k]
            y[k+1] = x[k+1] +  np.random.multivariate_normal([0]*n, 0*V)

        Y1 += y[1:].T
        Y0 += y[0:-1].T
        U0 += u[0:-1].T
        X1 += x[1:].T
        X0 += x[0:-1].T
    Y1 = Y1/TT
    Y0 = Y0/TT
    U0 = U0/TT
    X1 = X1/TT
    X0 = X0/TT
    print(f'U0: {U0}')
    print(f'X0: {X0}')
    XU0 = np.vstack([Y0, U0])
#     logger.info(f'Data rank: {np.linalg.matrix_rank(XU0)}')

#     XU0 = np.vstack([X0, U0])
    AhBh = Y1@np.linalg.pinv(XU0)
    Ah = AhBh[:, 0:n]
    Bh = AhBh[:, n:]
    print(f'Ah: {Ah}')
    print(f'Bh: {Bh}')
    print(f'Ad: {Ad}')
    print(f'Bd: {Bd}')
    # exit()

    # Wh = np.zeros((n, n))
    for i in range(0, N-1):
        wh = y[i+1, :] - (Ah@y[i, :] + Bh@u[i, :])
        wh = wh.reshape(n, 1)
        Wh += wh@wh.T
    Wh = Wh/(N)
    print(f'Wh: {Wh}')
    return Y1, Y0, U0, X1, X0, Ah, Bh, Wh


def data_driven_stability_from_noisy_measurements(Y1, Y0, U0, W, V):

    # State and input dimensions
    m, N = U0.shape
    n, N = Y1.shape
    D0 = np.vstack([U0, Y0])
    I = np.eye(n)
    Y = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((N, n))
    alpha = cp.Variable((1, 1))

    obj = 0 #alpha

    cons = [Y >> 0]

    cons += [
        cp.bmat([[-Y, (Y1@F).T, F.T],
                 [Y1@F, -Y, np.zeros((n, N))],
                 [F, np.zeros((N, n)), -alpha*np.eye(N)],
                 ]) << 0
    ]

    cons += [cp.trace(np.linalg.inv(W+V)@Y)-alpha*n*n >= 0]
    cons += [Y0@F == Y]
    cons += [alpha >= 0.0]

    prob = cp.Problem(cp.Maximize(0), cons)
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
        f'Date based closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P, G, alpha.value, obj




def data_driven_H2_from_noisy_measurements(Y1, Y0, U0, W, V, Q, R):

    # State and input dimensions
    m, N = U0.shape
    n, N = Y1.shape
    D0 = np.vstack([U0, Y0])
    I = np.eye(n)
    Im = np.eye(m)
    # print(f'Im: {Im}')
    Sigma = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((N, n))
    L = cp.Variable((m, m))
    H = cp.Variable((N, N))
    M = cp.Variable((N, m))
    Z = cp.Variable((N, N))
    E = cp.Variable((N, N))
    S = cp.Variable((n, n))


    beta = cp.Variable((1, 1))

    obj = 1*beta# + cp.trace(H) #+ cp.trace(E)
    # 
    # obj = cp.trace(Q@Sigma) + cp.trace(R@L) + cp.trace(R@U0@E@U0.T)  

    D0 = np.vstack([U0, Y0])
    Pi = np.eye(N) - np.linalg.pinv(D0)@D0

    cons = [Sigma >> 0]
    # cons += [H >> 0]
    # cons += [S >> 0]
    # cons += [L >> 0]
    # cons += [E >> 0]

    cons += [
        cp.bmat([[E, F],
                 [F.T, S],
                 ]) >> 0
    ]

    # cons += [ L - U0@H@U0.T >> 0]
    
    # cons += [cp.bmat([[L, U0@F],
    #              [F.T@U0.T, Sigma],
    #              ]) >> 0]
    cons += [cp.bmat([[H, F],
                 [F.T, Sigma],
                 ]) >> 0]
    cons += [cp.bmat([[S, Sigma],
                 [Sigma, V],
                 ]) >> 0]

    cons += [cp.trace(Q@Sigma) + 1*cp.trace(R@U0@H@U0.T) + cp.trace(R@U0@E@U0.T) +0*cp.trace(H) <= beta] #+  1*cp.trace(H)
    cons += [Y0@F == Sigma]
    # cons += [Pi@F == 0]
#     cons += [U0@M == I]
    # cons += [U0@M << 1.01*Im]
    cons += [U0@M == Im]
    cons += [Y0@M == np.zeros((n, m))]
    cons += [Y1@(H+E)@Y1.T + cp.trace(H+E)*(W+V) - Sigma + W << 0]
    # cons += [U0@(Z-E)@U0.T == 0]
    cons += [beta >= 0]


    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        logger.info(f'DDDP problem status: {prob.status}')
        return None, None, None

    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None, None

    G = F.value@np.linalg.inv(Sigma.value)
    K = U0@G
    

    # logger.info(f'Tr(PW): {np.trace(P@W)}, 1/alpha: {1/alpha1.value}')
    logger.info(
        f'H2 noisy measurements closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')
    # logger.info(f'M : {M.value}')

    return K, G, beta.value





def data_driven_dynamic_programming_LQR(X1, X0, U0, W, Q, R, delta):

    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    D0 = np.vstack([U0, X0])
    I = np.eye(n)
    Y = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))
    alpha = cp.Variable((1, 1))

    obj = 1*alpha

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




def data_driven_low_complexity_lqr(X1, X0, U0, W, Q, R, alpha):
    # State and input dimensions
    m, T = U0.shape
    n, T = X1.shape
    I = np.eye(n)
    P = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((T, n))
    L = cp.Variable((m, m))
    V = cp.Variable((T, T), symmetric=True)
    gamma = cp.Variable((1, 1))

    obj = 1*gamma

    cons = [gamma >= 0]
    # cons += [P >> I]
    cons += [L >> 0]
    cons += [V >> 0]
    cons += [cp.bmat([
        [W-P, X1@F],
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
        return None, None, None

    # "optimal_inaccurate",
    if prob.status in ["infeasible", "unbounded", "unbounded_inaccurate", "ILL_POSED", "UNKNOWN"]:
        return None, None, None

    K = U0@F.value@np.linalg.inv(P.value),
    logger.info(
        f'Data based Low Complexity closed-loop Ad+Bd@K : \n {np.linalg.eig(Ad+Bd@K)[0]}')

    return K, P.value, gamma.value



def simulate(K, sim_time, sim_num):
    i = 0

    x_mean = np.zeros((sim_time, n))
    x_min = 10000000*np.ones((sim_time, n))
    x_max = -1000000*np.ones((sim_time, n))

    u_mean = np.zeros((sim_time, m))
    u_min = 10000*np.ones((sim_time, m))
    u_max = -10000*np.ones((sim_time, m))

    cost_mean = np.zeros((sim_time, 1))
    cost_min= 100000*np.ones((sim_time, 1))
    cost_max = -100000*np.ones((sim_time, 1))

    while i < sim_num:
        random.seed(i)
        x = np.zeros((sim_time, n))
        y = np.zeros((sim_time, n))
        u = np.zeros((sim_time, m))
        cost = np.zeros((sim_time, 1))
        x[0] = x0
        y[0] = x0 + np.random.multivariate_normal([0]*n, 1*V)
        for k in range(0, sim_time-1):
            u[k] = K@y[k]
            cost[k] = x[k].T@Q@x[k] + u[k]@R@u[k]
            x[k+1] = Ad@x[k]+Bd@u[k]+np.random.multivariate_normal([0]*n, 1*W)
            y[k+1] = x[k+1]  +  np.random.multivariate_normal([0]*n, 1*V)
        
        x_mean = x_mean + x
        x_min = np.minimum(x_min, x)
        x_max = np.maximum(x_max, x)

        cost_mean = cost_mean + cost
        cost_min = np.minimum(cost_min, cost)
        cost_max = np.maximum(cost_max, cost)

        u_mean = u_mean + u
        u_min = np.minimum(u_min, u)
        u_max = np.maximum(u_max, u)
        i += 1

    x_mean /= sim_num
    u_mean /= sim_num
    cost_mean /= sim_num

    return x_mean, u_mean, cost_mean, x_min, x_max, u_min, u_max, cost_min, cost_max



# x0 = np.array([0.3, -4, 0.1, -1])
x0 = np.array([0.2, -0.2, 0, 0])
# 
N = 10# 7 for rotary pendulum  
# Q = np.array([
#     [10000, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])
# R = 0.000001*np.eye(m)
# Q = np.array([
#     [30000, 0, 0, 0],
#     [0, 30, 0, 0],
#     [0, 0, 20, 0],
#     [0, 0, 0, 1]
# ])
# R = 0.0001*np.eye(m)
# for rotary pendulum
Q = np.array([
    [1, 0, 0, 0],
    [0, 100, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
R = 10*np.eye(m)
# Q = np.array([
#     [30000, 0, 0, 0],
#     [0, 30, 0, 0],
#     [0, 0, 20, 0],
#     [0, 0, 0, 1]
# ])
# R = 0.0001*np.eye(m)
Klqr, S, E = control.dlqr(Ad, Bd, Q, R)
Klqr = -Klqr
K = -0.*Klqr
print(f'Klqr: {Klqr}')
print(f'A+BKlqr: {np.linalg.eigvals(Ad+Bd@Klqr)}')

desired_poles = [0.9, 0.8, -0.6,-0.5]
# K = control.place(Ad, Bd, desired_poles)
K, _, _ = control.dlqr(Ad, Bd, np.diag([1000, 1, 100, 1]), 0.001*R)
# print(f'K_place: {-K}')
Y1, Y0, U0, X1, X0, Ah, Bh, Wh= collect_data(Ad, Bd, W, V, -K, x0, N, Wh)
# print(f'Y1: {Y1}')
# print(f'Y0: {Y0}')
# K, P, G, alpha, objective_ = data_driven_stability_from_noisy_measurements(Y1, Y0, U0, W, 1*V)
# print(f'alpha: {alpha}')

Knoisy, G, beta = data_driven_H2_from_noisy_measurements(Y1, Y0, U0, W, V, Q, R)
print(f'Knoisy: {Knoisy}')
# print(f'G: {G}')
print(f'beta: {beta}')


K_auto, P, G, alpha, cost = data_driven_dynamic_programming_LQR(Y1, Y0, U0, W, Q, R, delta=0.99)
print(f'K_auto:{K_auto}')

Klow, P, gamma = data_driven_low_complexity_lqr(Y1, Y0, U0, W, Q, R, alpha=0)
print(f'Klow: {Klow}')
print(f'gamma: {gamma}')
print(f'Klqr: {Klqr}')
print(f'A+BKlqr: {np.linalg.eigvals(Ad+Bd@Klqr)}')
Khlqr, S, E = control.dlqr(Ah, Bh, Q, R)
Khlqr = -Khlqr
print(f'Khlqr: {Khlqr}')
print(f'A+BKhlqr: {np.linalg.eigvals(Ad+Bd@Khlqr)}')
# exit()

# Number of iterations
num_trials = 150

# Storage for results
success_count = {"stability": 0, "H2": 0, "dynamic_LQR": 0, "low_complexity": 0,
                  "dynamic_LQR_x": 0, "low_complexity_x": 0}
eigenval_count = {"stability": 0, "H2": 0, "dynamic_LQR": 0, "low_complexity": 0, 
                  "dynamic_LQR_x": 0, "low_complexity_x": 0}
K_sums = {"stability": np.zeros_like(Klqr), "H2": np.zeros_like(Klqr),
          "dynamic_LQR": np.zeros_like(Klqr), "low_complexity": np.zeros_like(Klqr),
          "dynamic_LQR_x": np.zeros_like(Klqr), "low_complexity_x": np.zeros_like(Klqr)}




sim_time = 20
sim_num = 50
costs = {"H2": {"mean": np.zeros((sim_time, 1)), "min": 10000000*np.zeros((sim_time, 1)), "max": -100000000*np.zeros((sim_time, 1)), },
         "dynamic_LQR": {"mean": np.zeros((sim_time, 1)), "min": 10000000*np.zeros((sim_time, 1)), "max": -100000000*np.zeros((sim_time, 1)), },
         "low_complexity": {"mean": np.zeros((sim_time, 1)), "min": 10000000*np.zeros((sim_time, 1)), "max": -100000000*np.zeros((sim_time, 1)), }}
# Run multiple trials
for i in range(num_trials):
    # Generate data
    random.seed(i)
    Wh = np.zeros((n,n))
    Y1, Y0, U0, X1, X0, Ah, Bh , Wh= collect_data(Ad, Bd, W, V, -K, x0, N, Wh)

    # Stability-based method
    K_stability, _, _, alpha, _ = data_driven_stability_from_noisy_measurements(Y1, Y0, U0, W, 10*V)
    if K_stability is not None:
        success_count["stability"] += 1
        eigenvals = np.linalg.eigvals(Ad + Bd @ K_stability)
        if np.all(np.abs(eigenvals) < 1):
            eigenval_count["stability"] += 1
            K_sums["stability"] += K_stability

    # H2 method
    K_H2, _, beta = data_driven_H2_from_noisy_measurements(Y1, Y0, U0, W, 1*V, Q, R)
    if K_H2 is not None:
        success_count["H2"] += 1
        eigenvals = np.linalg.eigvals(Ad + Bd @ K_H2)
        if np.all(np.abs(eigenvals) < 1):
            eigenval_count["H2"] += 1
            K_sums["H2"] += K_H2
            x_mean, u_mean, cost_mean, x_min, x_max, u_min, u_max, cost_min, cost_max = simulate(K_H2, sim_time, sim_num)
            if eigenval_count["H2"] == 1:
                costs["H2"]["mean"] += cost_mean
                costs["H2"]["min"] = cost_min
                costs["H2"]["max"] = cost_max
            else:
                costs["H2"]["mean"] += cost_mean
                costs["H2"]["min"] = np.minimum(costs["H2"]["min"], cost_min)
                costs["H2"]["max"] = np.maximum(costs["H2"]["max"], cost_max)

    # Dynamic Programming LQR method
    K_auto, _, _, _, _ = data_driven_dynamic_programming_LQR(Y1, Y0, U0, 1000*W, Q, R, delta=0.999)
    if K_auto is not None:
        success_count["dynamic_LQR"] += 1
        eigenvals = np.linalg.eigvals(Ad + Bd @ K_auto)
        if np.all(np.abs(eigenvals) < 1):
            eigenval_count["dynamic_LQR"] += 1
            K_sums["dynamic_LQR"] += K_auto
            x_mean, u_mean, cost_mean, x_min, x_max, u_min, u_max, cost_min, cost_max = simulate(K_auto, sim_time, sim_num)
            if eigenval_count["dynamic_LQR"] == 1:
                costs["dynamic_LQR"]["mean"] += cost_mean
                costs["dynamic_LQR"]["min"] = cost_min
                costs["dynamic_LQR"]["max"] = cost_max
            else:
                costs["dynamic_LQR"]["mean"] += cost_mean
                costs["dynamic_LQR"]["min"] = np.minimum(costs["dynamic_LQR"]["min"], cost_min)
                costs["dynamic_LQR"]["max"] = np.maximum(costs["dynamic_LQR"]["max"], cost_max)

    # Low Complexity LQR method
    K_low, _, gamma = data_driven_low_complexity_lqr(Y1, Y0, U0, W, Q, R, alpha=0.1)
    if K_low is not None:
        success_count["low_complexity"] += 1
        eigenvals = np.linalg.eigvals(Ad + Bd @ K_low)
        if np.all(np.abs(eigenvals) < 1):
            eigenval_count["low_complexity"] += 1
            K_sums["low_complexity"] += K_low[0]
            x_mean, u_mean, cost_mean, x_min, x_max, u_min, u_max, cost_min, cost_max = simulate(K_low, sim_time, sim_num)
            if eigenval_count["low_complexity"] == 1:
                costs["low_complexity"]["mean"] += cost_mean
                costs["low_complexity"]["min"] = cost_min
                costs["low_complexity"]["max"] = cost_max
            else:
                costs["low_complexity"]["mean"] += cost_mean
                costs["low_complexity"]["min"] = np.minimum(costs["low_complexity"]["min"], cost_min)
                costs["low_complexity"]["max"] = np.maximum(costs["low_complexity"]["max"], cost_max)

    # Dynamic Programming LQR method with X1, X0
    # K_auto_x, _, _, _, _ = data_driven_dynamic_programming_LQR(X1, X0, U0, 1000000*W, Q, R, delta=0.999)
    # if K_auto_x is not None:
    #     success_count["dynamic_LQR_x"] += 1
    #     eigenvals = np.linalg.eigvals(Ad + Bd @ K_auto_x)
    #     if np.all(np.abs(eigenvals) < 1):
    #         eigenval_count["dynamic_LQR_x"] += 1
    #         K_sums["dynamic_LQR_x"] += K_auto_x

    # # Low Complexity LQR method
    # K_low_x, _, gamma = data_driven_low_complexity_lqr(X1, X0, U0, W, Q, R, alpha=0.1)
    # if K_low_x is not None:
    #     success_count["low_complexity_x"] += 1
    #     eigenvals = np.linalg.eigvals(Ad + Bd @ K_low_x)
    #     if np.all(np.abs(eigenvals) < 1):
    #         eigenval_count["low_complexity_x"] += 1
    #         K_sums["low_complexity_x"] += K_low_x[0]

x_mean, u_mean, cost_mean, x_min, x_max, u_min, u_max, cost_min, cost_max = simulate(Klqr, sim_time, sim_num)
t = np.linspace(0, sim_time-1, sim_time-1)
axs.plot(t, cost_mean[0:-1], 'black',  linestyle='-', linewidth=2, marker='o', markersize=5, label=r'NSMLQR', zorder=3)
axs.fill_between(t, 
                 np.array(cost_min[0:-1]).ravel() , 
                 np.array(cost_max[0:-1]).ravel() , color='black', alpha=0.2, zorder=1) 

axs.plot(t, costs["H2"]["mean"][0:-1]/eigenval_count['H2'], 'red',  linestyle='-', linewidth=2, marker='o', markersize=5, label=r'DDNSMLQR', zorder=3)
axs.fill_between(t, 
                 np.array(costs["H2"]["min"][0:-1]).ravel() , 
                 np.array(costs["H2"]["max"][0:-1]).ravel() , color='red', alpha=0.2, zorder=1) 

axs.plot(t, costs["dynamic_LQR"]["mean"][0:-1]/eigenval_count['dynamic_LQR'], 'blue', linestyle='--', linewidth=2, marker='s', markersize=5, label='DDDPLQR', zorder=4)
axs.fill_between(t, 
                 np.array(costs["dynamic_LQR"]["min"][0:-1]).ravel() , 
                 np.array(costs["dynamic_LQR"]["max"][0:-1]).ravel() , color='blue', alpha=0.15, zorder=2)

axs.plot(t, costs["low_complexity"]["mean"][0:-1]/eigenval_count['low_complexity'], 'green', linestyle='-.', linewidth=2, marker='d', markersize=5, label='DDLCLQR', zorder=5)
axs.fill_between(t, 
                 np.array(costs["low_complexity"]["min"][0:-1]).ravel() , 
                 np.array(costs["low_complexity"]["max"][0:-1]).ravel() , color='green', alpha=0.1, zorder=1)   


# Formatting
axs.set_xlabel("k", fontsize=14, fontweight='bold')
axs.set_ylabel("Average Cost", fontsize=14, fontweight='bold')
axs.set_title("Comparison of Cost Functions Over Time", fontsize=16, fontweight='bold')

axs.grid(True, linestyle="--", alpha=0.6)  # Add dashed grid lines
axs.legend(loc="upper right", fontsize=12, frameon=True, bbox_to_anchor=(1.05, 1))  # Legend outside the plot

plt.tight_layout()  # Adjust layout for better fit

# Compute averages
K_avg = {key: (K_sums[key] / eigenval_count[key]) if eigenval_count[key] > 0 else None
         for key in K_sums.keys()}

# Compute norms with respect to LQR
K_norms = {key: (np.linalg.norm(K_avg[key] - Klqr) if K_avg[key] is not None else None)
           for key in K_avg.keys()}

# Display results
print(f'Klqr: {Klqr}')
print(f'K_place: {-K}')
print(f'A+BKplace: {np.linalg.eigvals(Ad-Bd@K)}')
print(f'Wh: {Wh}')
print("Success Count:", success_count)
print("Eigenvalues Inside Unit Circle:", eigenval_count)
print("Average K Matrices:", K_avg)
print("Norms of Average K with respect to Klqr:", K_norms)


plt.show()