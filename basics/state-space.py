import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np


# Simulation parameters
x0 = [0, 0]
start = 0
stop = 30
step = 1
t = np.arange(start, stop, step)

K = 3
T = 4

# State space model
A = [[-1/T, 0], [0, 0]]
B = [[K/T], [0]]
C = [[1, 0]]
D = 0

sys = sig.StateSpace(A, B, C, D)
t, xt = sig.step(sys, x0, t)
H = sys.to_tf()
print(f'H: {H}')

plt.plot(t, xt)
plt.title('Step Response of a State Space Model')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid()
plt.show()
