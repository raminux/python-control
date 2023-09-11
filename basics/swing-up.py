import matplotlib.pyplot as plt
from matplotlib import rc, animation, patches
rc('animation', html='html5')

import numpt as np
import scipy.integrate as integrate

from scipy.linalg import solve_continuous_are


# System parameters
g = 9.81
l = 1.0
m_1 = 1.0
m_c = 1.0

def make_deriv(f=lambda state, t, args: 0):

    def deriv(state, t, args):
        dydx = np.zeros_like(state)

        x, theta1, x_dot, theta1_dot = state

        dydx[0] = x_dot
        dydx[1] = theta1_dot

        cos_theta1 = np.cos(theta1)
        sin_theta1 = np.sin(theta1)

        ft = f(state, t, args)
        