import numpy as np
import control
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sqrt


A = [
    [0, 1],
    [-1, -1],
]

B = [
    [0], 
    [1]
]

C = [[1, 0]]
D = [[0]]

ro = 1

F = [[1-sqrt(1+1/ro), 1-sqrt(1+2*sqrt(1+1/ro))]]
# print(F, F.shape)
x0 = [1, 1]

# function that returns dz/dt

def model(x, t, u):
    x1 = x[0]
    x2 = x[1]
    dx1dt = x2
    dx2dt = -x1-x2+u
    dxdt = [dx1dt,dx2dt]
    return dxdt

n = 101
t = np.linspace(0,6,n)
x1 = np.empty_like(t)
x2 = np.empty_like(t)
u = np.empty_like(t)
# record initial conditions
x1[0] = 1
x2[0] = 1

# solve ODE
ro = 1
done = True
while done:
    # ro = ro - 0.01
    print(f'ro: {ro}')
    x1 = np.empty_like(t)
    x2 = np.empty_like(t)
    u = np.empty_like(t)
    # record initial conditions
    x1[0] = 1
    x2[0] = 1
    for i in range(1,n):
        # span for next time step
        tspan = [t[i-1], t[i]]
        # solve for next step
        u[i] = (1-sqrt(1+1/ro))*x1[i-1]+(1-sqrt(1+2*sqrt(1+1/ro)))*x2[i-1]
        # if abs(u[i]) > 2: 
            # done = False
        xt = odeint(model, x0, tspan, args=(u[i], ))
        x1[i] = xt[1][0]
        x2[i] = xt[1][1]
        x0 = xt[1]
    done = False

# plot results
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(12)
plt.subplot(3,1,1)
plt.title(f'Result for ro = {ro}')
plt.ylabel('u')
plt.plot(t,u,'g:',label='u(t)')
plt.subplot(3,1,2)
plt.ylabel('x1')
plt.plot(t,x1,'b-',label='x1(t)')
plt.subplot(3,1,3)
plt.ylabel('x2')
plt.plot(t,x2,'r--',label='x2(t)')
plt.xlabel('time (s)')
# plt.legend(loc='best')
plt.show()