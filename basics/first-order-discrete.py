import numpy as np
import matplotlib.pyplot as plt

# Model parameters
K = 3
T = 4
a = -1/T
b = K/T

# Simulation Parameters
Ts = 0.1
Tstop = 30
uk = 1
N = int(Tstop/Ts)
yk = [0]
for k in range(N):
    y = (1+a*Ts)*yk[-1]+Ts*b*uk
    yk.append(y)

k = np.arange(0, Tstop+Ts, Ts)

plt.plot(k, yk)
plt.title('First Order Discrete Time Step Response')
plt.xlabel('k')
plt.ylabel('y_k')
plt.grid()
plt.show()