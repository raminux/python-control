import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


K = 3
T = 4
u = 1

t_start = 0
t_stop = 25
step = 1

y0 = 0

t = np.arange(start=t_start, stop=t_stop, step=step)

def sys(y, t, K, T, u):
    dydt = (1/T)*(-y+K*u)
    return dydt

# Solve ODE
yt = odeint(sys, y0, t, args=(K, T, u))
print(yt)
plt.plot(t, yt)
plt.title('First Order Dynamic System Step Response Using odeint')
plt.xlabel('x [s]')
plt.ylabel('y(t)')
plt.grid()
plt.show()
