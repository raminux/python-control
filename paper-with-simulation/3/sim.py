import numpy as np
import matplotlib.pyplot as plt


def f(xk):
    p1 = 0.5
    p2 = 0.5
    vb = 0.01
    if not xk.shape:
        vk = -vb + 2*vb*np.random.rand()
    else:
        vk = -vb + 2*vb*np.random.rand(xk.shape[0])
    
    yk = p1*np.exp(-xk) + p2*np.exp(-xk)*np.sin(xk) + vk
    return yk

k0 = 0
kf = 10000
xL = 0
xH = 2
step = (xH-xL)/(kf-k0)
xk = np.arange(xL, xH+step, step)
yk = f(xk)
q = 2
n = 1
P = 2
rank = min(q, P)
M = np.zeros((q, P))
Y = np.zeros((n, P))
Theta = np.zeros((q, xk.shape[0]))
Theta[3][:] = np.array([[1],[2]])
print(Theta)
exit()
tg = 0.1
tc = 0.01
TG = tg*np.eye(q)
TC = tc*np.eye(q)
i = 0
for k in xk:
    y = f(x[k])
    if np.linalg.matrix_rank(M) < rank:
        if i == 2: i=0
        Y[0][i] = y
        M[0][i] = np.exp(-x[k])
        M[1][i] = np.exp(-x[k])*np.sin(x[k])
        i += 1
    if np.linalg.matrix_rank(M) >= rank:
        yhk = 5

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
axs[0][0].plot(xk, yk)
# plt.show()



