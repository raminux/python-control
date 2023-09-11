
import numpy as np
import math
import random


def phase_portrait(A_cl, x1_min, x1_max, x2_min, x2_max, ax, num):
    # Define the system equations
    def system_eqns(X):
        x1, x2 = X
        x1_next = A_cl[0][0]*x1 + A_cl[0][1]*x2
        x2_next = A_cl[1][0]*x1 + A_cl[1][1]*x2
        return np.array([x1_next, x2_next])

    # Create a grid of values for x1 and x2
    x1_vals = np.linspace(x1_min, x1_max, 20)
    x2_vals = np.linspace(x2_min, x2_max, 20)

    # Create a meshgrid of all possible (x1,x2) pairs
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Apply the system equations to each (x1,x2) pair
    X_next = np.apply_along_axis(system_eqns, 0, np.array([X1, X2]))
    ax.set_xlim([x1_min, x1_max])
    ax.set_ylim([x2_min, x2_max])
    widths = np.linspace(2, 4, X1.size)
    q = ax.quiver(X1, X2, X_next[0]-X1, X_next[1]-X2, linewidths=widths)
    # ax.quiverkey(q, X=0.8, Y=1.05, U=10, label='Phase portrait', labelpos='E')



def plot_trajectories(A, B, F, Cov, W, axs, axu, t, sims, num, X0_cov, initials, Q, R):
    # State and input dimensions
    n, m = B.shape
    mean_cost = np.zeros((t.shape[0], 1))
    for j in range(sims):
        x = np.zeros((t.shape[0], n))
        w = np.zeros((t.shape[0], n))
        u = np.zeros((t.shape[0], m))
        cost = np.zeros((t.shape[0], 1))
        i = 0#np.random.randint(0, initials.shape[0])
        x[0] = np.random.multivariate_normal([initials[i][0], initials[i][1], initials[i][2], initials[i][3]], X0_cov)
        for k in range(0,num-1):
            cost[k] = x[k].T@Q@x[k] + u[k]*R*u[k]      
            u[k] = random.gauss(F@x[k], math.sqrt(Cov))
            w[k] = np.random.multivariate_normal([0, 0, 0, 0], W)
            x[k+1] = A@x[k]+B@u[k]+w[k]
            

        x = x.T
        x1 = x[0][:]
        x2 = x[1][:]
        x3 = x[2][:]
        x4 = x[3][:]

        # if j == 0:
        #     axs[0][0].plot(t, x1, label='State trajectories')
        #     axs[][1].plot(t, x2, label='State trajectories')
        #     axs[1][0].plot(t, x3, label='State trajectories')
        #     axs[1][1].plot(t, x4, label='State trajectories')
        #     axu.plot(t, u)

        # else:0
        axs[0][0].plot(t, x1)
        axs[0][1].plot(t, x2)
        axs[1][0].plot(t, x3)
        axs[1][1].plot(t, x4)
        axu.plot(t, u)
        mean_cost = mean_cost + cost
    # axu.plot(t, mean_cost/sims)
    return sum(mean_cost/sims)/num