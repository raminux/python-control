import numpy as np
import matplotlib.pyplot as plt

# Transfer function --> Y(s)/U(s) = K/(Ts+1), 
# ODE --> dydt = (1/T)*(-y+Ku), 
# Step response --> y(t) = KU(1-exp(-t/T))

K = 3
T = 4

start = 0
stop = 30
step = 0.1

t = np.arange(start=start, stop=stop, step=step)

yt = K*(1-np.exp(-t/T))
plt.plot(t, yt)
plt.title('First Order Dynamic System Step Response')
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.grid()
plt.show()
