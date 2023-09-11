import logging
import cvxpy as cp
import numpy as np
import scipy.linalg as linalg
import control
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from controllers import lqr, meslqr, safe_control
from helper import phase_portrait, plot_trajectories

plt.rcParams['text.usetex'] = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define figures for controllers
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
fig1, axs1 = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))

fig_lqr = plt.figure(figsize=(6, 5), tight_layout=True, dpi=120)
fig_meslqr = plt.figure(figsize=(6, 5), tight_layout=True, dpi=120)

ax_lqr = fig_lqr.add_subplot(111)
ax_meslqr = fig_meslqr.add_subplot(111)

# Set the desired style
# plt.style.use('ggplot')
plt.style.use('bmh')

# Change the color palette using a predefined color map
# ax_lqr.set_prop_cycle('color', plt.cm.Set2.colors)

# ax_lqr.set_aspect('equal')

# Define safe set
x1_min, x1_max = -18, 18
x2_min, x2_max = -5, 5
x1_g, x2_g = 2, 2

# Simulation parameters 
Ts = 0.005
num = 500 # number of data points
sims = 100 # number of simulations
t_final = 100
t = np.linspace(0,t_final, num)

# Define the matrices of the system
# m = 0.1
# k = 0.001
# g = 9.81
# a = 0.05
# L0 = 0.01
# L1 = 0.02
# R = 1
# r = 0.05
# Lr = L1 + L0*a/(a+r)

# x03 = np.sqrt(2*g*m*((a+r)**2)/(L0*a))
# A = np.eye(3) + Ts*np.array([
#     [0, 1, 0],
#     [2*g/(a+r), -k/m, -L0*a*x03/(m*(a+r)**2)],
#     [0, L0*a*x03/(Lr*(a+r)**2), -R/Lr]
# ])
# B = Ts*np.array([
#     [0],
#     [0],
#     [1/Lr]
# ])

A = np.array([
    [1, 0.01, 0.0001, 0],
    [0, 0.9982, 0.0267, 0.0001],
    [0.0, 0.0, 1.0016, 0.01],
    [0.0, -0.0045, 0.3122, 1.0016]
])
B = np.array([
    [0.0001], [0.0182], [0.0002], [0.0454]
])
# logger.info(f"A: {A}")
# logger.info(f"B: {B}")
# logger.info(f"Controlability: {np.linalg.matrix_rank(control.ctrb(A, B))}")
# logger.info(f"eigenvalues: {np.linalg.eig(A)}")
# State and input dimensions
n, m = B.shape

# Noise covariance
# W = np.array([
#     [0.001, 0, 0],
#     [0, 0.002, 0],
#     [0, 0, 0.001]
# ])
W = 1*np.array([
    [0.0006, 0.0003, 0.0001, 0.0006],
    [0.0003, 0.0008, 0.0003, 0.0004],
    [0.0001, 0.0003, 0.0007, 0.0006],
    [0.0006, 0.0004, 0.0006, 0.0031]
])
# LQR parameters
R = 1
Q = 1*np.array([
    [100, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 100, 0],
    [0, 0, 0, 1]
    ])

# Sum of initial conditions
Z = 0.01*np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0], 
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])
gamma = 0.96

Klqr, Slqr, _ = control.dlqr((np.sqrt(gamma))*A, (np.sqrt(gamma))*B, Q, R)
logger.info(f'Klqr = {-Klqr}')
logger.info(f'Slqr = {Slqr}')
F_lqr = -Klqr
logger.info(f'A+BK = {np.linalg.eig(A+B@F_lqr)[0]}')

# exit()
# Discount factor


# Design LQR 
# F_lqr, Cost_lqr = lqr(A, B, Q, R, Z, W, gamma)
# logger.info(f'F_lqr = {F_lqr}')
# logger.info(f'Cost_lqr = {Cost_lqr}')
# logger.info(f'A+BK = {np.linalg.eig(A+B@F_lqr)[0]}')


# Design MESLQR
alpha = 10
sb = 1
Sb = sb*np.array([
    [0.9, 0, 0, 0, 0],
    [0, 10, 0, 0, 0], 
    [0, 0, 10, 0, 0], 
    [0, 0, 0, 1000, 0], 
    [0, 0, 0, 0, 1000]
])
logger.info(f'Sb: {Sb}')
# exit()
F_meslqr, Cov_meslqr, Cost_meslqr, S_meslqr, eig_S, Y = meslqr(A, B, Q, R, Z, W, Sb, gamma, alpha)
logger.info(f'F_meslqr = {F_meslqr}')
logger.info(f'Cov_meslqr = {Cov_meslqr}')
logger.info(f'Cost_meslqr = {Cost_meslqr}')
logger.info(f'S_meslqr = {(S_meslqr-Sb)@Y}')
logger.info(f'eig_S = {eig_S[0]}')
logger.info(f'A+BK = {np.linalg.eig(A+B@F_meslqr)[0]}')

# exit()

########### Plot trajectories #################
initials = np.array([
    [2., 2., 0.5, 2], 
    
])
X0_cov = 0.0006*np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 1, 0], 
    [0, 0, 0, 1]
    ])
X0_cov = W

# LQR
cost_lqr = plot_trajectories(A, B, F_lqr, 1*Cov_meslqr, 1*W, axs1, ax_lqr, t, sims, num, X0_cov, initials, Q, R)

# MESLQR
cost_meslqr = plot_trajectories(A, B, F_meslqr, 1*Cov_meslqr, W, axs, ax_meslqr, t, sims, num, X0_cov, initials, Q, R)

###############################################


# Show the results
ax_lqr.set_xlabel(r'$k$', fontsize=16)
ax_lqr.set_ylabel(r'$u$', fontsize=16)
ax_lqr.set_title(f'LQR, Cost={cost_lqr}', fontsize=14, fontweight='bold', loc='left')

ax_meslqr.set_xlabel(r'$k$', fontsize=16)
ax_meslqr.set_ylabel(r'$u$', fontsize=16)
tex = r'MESLQR, $\alpha$' + f'\,=\,{alpha}' + r', Cost' + f'\,=\,{cost_meslqr}'
ax_meslqr.set_title(tex, fontsize=14, fontweight='bold', loc='left')
fig.suptitle('MESLQR', fontsize=14, fontweight='bold')
fig1.suptitle('LQR', fontsize=14, fontweight='bold')

axs[0][0].set_xlabel(r'$k$', fontsize=16)
axs[0][0].set_ylabel(r'$x_1 [m]$', fontsize=16)

axs[0][1].set_xlabel(r'$k$', fontsize=16)
axs[0][1].set_ylabel(r'$x_2 [m/s]$', fontsize=16)

axs[1][0].set_xlabel(r'$k$', fontsize=16)
axs[1][0].set_ylabel(r'$x_3 [rad]$', fontsize=16)

axs[1][1].set_xlabel(r'$k$', fontsize=16)
axs[1][1].set_ylabel(r'$x_4 [rad/s]$', fontsize=16)


axs1[0][0].set_xlabel(r'$k$', fontsize=16)
axs1[0][0].set_ylabel(r'$x_1 [m]$', fontsize=16)

axs1[0][1].set_xlabel(r'$k$', fontsize=16)
axs1[0][1].set_ylabel(r'$x_2 [m/s]$', fontsize=16)

axs1[1][0].set_xlabel(r'$k$', fontsize=16)
axs1[1][0].set_ylabel(r'$x_3 [rad]$', fontsize=16)

axs1[1][1].set_xlabel(r'$k$', fontsize=16)
axs1[1][1].set_ylabel(r'$x_4 [rad/s]$', fontsize=16)





ax_lqr.legend(fontsize=16)
ax_meslqr.legend(fontsize=16)

fig_lqr.savefig('lqr-mag-sus.png')
fig_meslqr.savefig(f'meslqr-{alpha}-mag-sus.png')

plt.show()
