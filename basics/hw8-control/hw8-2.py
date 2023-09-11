import numpy as np
import control
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sqrt


A = [
    [0, 1],
    [0, -1],
]

B = [
    [0], 
    [1]
]

C = [[1, 0]]
D = [[0]]


F_b = [-1, -1]
N = 1
K = [9, 16]
F_c = [-3, -2, 1]

x0 = [0, 0]
xh0 = [0, 0]
z0 = 0

# function that returns dz/dt
alpha = 0.2


def integrator(z, t, r, x):
    dzdt = r-x
    return dzdt

def model(x, t, u):
    x1 = x[0]
    x2 = x[1]
    dx1dt = x2
    dx2dt = -x2+u - alpha*x1
    dxdt = [dx1dt,dx2dt]
    return dxdt

def observer(xh, t, u, y):
    xh1 = xh[0]
    xh2 = xh[1]
    e = y-xh1
    dxh1dt = xh2
    dxh2dt = -xh2+u + K[0]*e
    dxhdt = [dxh1dt,dxh2dt]
    return dxhdt

n = 201
t = np.linspace(0,20,n)
x1 = np.empty_like(t)
x2 = np.empty_like(t)
u = np.empty_like(t)
# record initial conditions
x1[0] = 0
x2[0] = 0

# solve ODE
x1 = np.empty_like(t)
x2 = np.empty_like(t)
xh1 = np.empty_like(t)
xh2 = np.empty_like(t)

u = np.empty_like(t)
# record initial conditions
x1[0] = 0
x2[0] = 0

# reference
r = 1

for i in range(1,n):
    # span for next time step
    tspan = [t[i-1], t[i]]
    # solve for next step
    z = odeint(integrator, z0, tspan, args=(r, x1[i-1]))
    z0 = z[1]
    u[i] = F_c[0]*xh1[i-1]+F_c[1]*xh2[i-1]  + F_c[2]*(z0)
    xt = odeint(model, x0, tspan, args=(u[i], ))
    x1[i] = xt[1][0]
    x2[i] = xt[1][1]
    x0 = xt[1]
    xht = odeint(observer, xh0, tspan, args=(u[i], x1[i]))
    xh1[i] = xht[1][0]
    xh2[i] = xht[1][1]
    xh0 = xht[1]

# plot results
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(12)
plt.subplot(3,1,1)
plt.ylabel('u')
plt.title(f'Integral Output Feedback Controller with alpha = 0.2')
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