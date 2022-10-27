import numpy as np
import matplotlib.pyplot as plt


# Model Parameters
K = 3
T = 4
a = -(1/T)
b = K/T

# Simulation Parameters
Ts = 0.1 # sampling time
Tstop = 20 # simulation time
N = int(Tstop/Ts)
y = np.zeros(N+2)
y[0] = 0

# PI controller settings
Kp = 0.5
Ti = 5

r = 5 # reference signal
e = np.zeros(N+2)
u = np.zeros(N+1)

for k in range(N+1):
    e[k] = r - y[k]
    u[k] = u[k-1] + Kp*(e[k] - e[k-1]) + (Kp/Ti)*e[k]
    y[k+1] = (1+Ts*a)*y[k] + Ts*b*u[k]

t = np.arange(0, Tstop+2*Ts, Ts)
print(f'len(t): {len(t[0:-1])}')
print(f'len(u): {len(u)}')
print(t[0:-2])

plt.figure(1)
plt.plot(t, y)
plt.title('Output Response of a Controlled System')
plt.xlabel('t [s]')
plt.ylabel('y(t)')
xmin = 0
xmax = Tstop
ymin = 0
ymax = 10
plt.axis([xmin, xmax, ymin, ymax])
plt.grid()

plt.figure(2)
plt.plot(t[0:-1], u, '-*', markersize=3, color='red')
plt.title('Control Signal')
plt.xlabel('t [s]')
plt.ylabel('u [V]')
plt.grid()
plt.show()

