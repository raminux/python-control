import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import cos, sin, exp, log2


'''
Simulating the example in the paper Using Theorem 2

- the algorithm does not converge to any stable point even though I changed the parameters 
of the algorithm and also the basis functions. I couldn't be able to tune the actor-critic networks
proposed in the paper. 

- The other problem is that the barrier function is undefined when the state of the system passes the 
safe set. There is no guarantee to keep the system states in the safe zone during the learning phase. 

- There is no discussion on how to choose the initial policy for the system. 

- Without an initial stabilizing controller, the algorihtm diverges!

- The simulation done in the paper is wrong, because the simulaton was done on the transformed system, which is not correct!

- No information is provided for the parameters of the algorithm used in the simulation. 

'''

# state constraints
a1 = -1.3*1
A1 = 0.5*1
a2 = -3.1*1
A2 = 0.5*1

# Simulation parameters
n = 5001
t = np.linspace(0,200,n)

# State variables and control signals
x1 = np.empty_like(t)
x2 = np.empty_like(t)
u1 = np.empty_like(t)
u2 = np.empty_like(t)
s1 = np.empty_like(t)
s2 = np.empty_like(t)
B = 0.1
A = 2

# record initial conditions
x0 = [-1, -3]
x1[0] = -1
x2[0] = -3
s0 = [-2.57, -5.38]
s1[0] = -2.57
s2[0] = -5.38

# Cost Parameters
Q1 = 2*np.eye(2)
Q2 = np.eye(2)
R11 = 2*np.eye(1)
R12 = 2*np.eye(1)
R21 = np.eye(1)
R22 = np.eye(1)

# Actor parameters
aa1 = 0.93
aa2 = 0.94
wa1 = [0.6, 0.1, 0.8, 0.9, 0.2, 0.2, 0.8, 0.9]
wa2 = [0.2, 0.2, 0.9, 0.3, 0.6, 0.6, 0.6, 0.8]

# Critic parameters
ac1 = 0.95
ac2 = 0.95
wc1 = [0.6, 0.2, 0.9, 0.1, 0.4, 0.8, 0.4, 0.6]
wc2 = [0.5, 0.2, 0.2, 0.3, 0.7, 0.7, 0.5, 0.6]

# Experience Replay Parameters
sigma_a1_k = []
e_a_c1_k = []
sigma_a2_k = []
e_a_c2_k = []
p = 10


# System Dynamics
def sys_dynamic(x, t, u1, u2):
    x1 = x[0]
    x2 = x[1]
    fx = -x2-x1/2+(x2*(cos(2*x1)+2)**2)/4+(x2*(sin(4*x1*x1)+2)**2)/4
    g1x = cos(2*x1)+2
    g2x = sin(4*x1*x1)+2
    dx1dt = x2
    dx2dt = fx + g1x*u1 + g2x*u2
    dxdt = [dx1dt, dx2dt]
    return dxdt


# Transformed System Dynamics
# def s_dynamic(s, t, u1, u2, x):
#     s1, s2 = s[0], s[1]
#     # x1, x2 = x[0], x[1]
#     x1 = a1*A1*(exp(s1/2)-exp(-s1/2))/(a1*exp(s1/2)-A1*exp(-s1/2))
#     x2 = a2*A2*(exp(s2/2)-exp(-s2/2))/(a2*exp(s2/2)-A2*exp(-s2/2))
#     fx = -x2-x1/2+(x2*(cos(2*x1)+2)**2)/4+(x2*(sin(4*x1**2)+2)**2)/4
#     g1x = cos(2*x1)+2
#     g2x = sin(4*x1**2)+2
#     ds1dt = (a2*A2*(exp(s2/2)-exp(-s2/2))*(A1*A1*exp(-s1)-2*a1*A1+a1*a1*exp(s1)))/((a2*exp(s2/2)-A2*exp(-s2/2))*(A1*a1*a1-a1*A1*A1))
#     ds2dt = (fx + g1x*u1 + g2x*u2)*(A2*A2*exp(-s2)-2*a2*A2+a2*a2*exp(s2))/(A2*a2*a2-a2*A2*A2)
#     dsdt = [ds1dt, ds2dt]
#     return dsdt


# Barrier Function
def S(x):
    x1 = x[0]
    x2 = x[1]
    s1 = log2((A1/a1)*((a1-x1)/(A1-x1)))
    s2 = log2((A2/a2)*((a2-x2)/(A2-x2)))
    return s1, s2


# Critic Basis Vector
def phi_c(x): 
    s1, s2 = S(x)
    return np.array([
    [s1**4],
    [(s1**3)*s2],
    [(s1**2)*(s2**2)],
    [s1*(s2**3)],
    [s2**4],
    [s1**2], 
    [s1*s2], 
    [s2**2]
    ])

# Critic Basis Gradient Matrix
def gradient_phi_c(x):
    s1, s2 = S(x)
    return np.array([
    [4*(s1**3), 0],
    [3*s2*(s1**2), s1**3],
    [2*s1*(s2**2), 2*s2*(s1**2)],
    [s2**3, 3*s1*(s2**2)],
    [0, 4*(s2**3)], 
    [2*s1, 0], 
    [s2, s1], 
    [0, 2*s2]
    ])

# Actor Basis Vector
def phi_a(x):
    s1, s2 = S(x)
    return np.array([
    [s1**4],
    [(s1**3)*s2],
    [(s1**2)*(s2**2)],
    [s1*(s2**3)],
    [s2**4], 
    [s1**2], 
    [s1*s2], 
    [s2**2]
    ])


# Actor Basis Gradient Matrix
def gradient_phi_a(x):
    s1, s2 = S(x)
    return np.array([
    [4*(s1**3), 0],
    [3*s2*(s1**2), s1**3],
    [2*s1*(s2**2), 2*s2*(s1**2)],
    [s2**3, 3*s1*(s2**2)],
    [0, 4*(s2**3)], 
    [2*s1, 0], 
    [s2, s1], 
    [0, 2*s2]
    ])


# F(s)
def F(x):
    x1, x2 = x[0], x[1]
    s1, s2 = S(x)
    F1 = (a2*A2*(exp(s2/2)-exp(-s2/2))/(a2*exp(s2/2)-A2*exp(-s2/2)))*((A2*A2*exp(-s1)-2*a1*A1+a2*a2*exp(s1))/(A1*a1*a1-a1*A1*A1))
    F2 = (-x2-x1/2+(x2*(cos(2*x1)+2)**2)/4 + (x2*(sin(4*x1*x1)+2)**2)/4)*((A2*A2*exp(-s2)-2*a2*A2+a2*a2*exp(s2))/(A2*a2*a2-a2*A2*A2))
    return np.array([
        [F1],
        [F2]
    ])


#G1(s)
def G1(x):
    x1, x2 = x[0], x[1]
    s1, s2 = S(x)
    return np.array([
        [0], 
        [(cos(2*x1)+2)*((A2*A2*exp(-s2)-2*a2*A2+a2*a2*exp(s2))/(A2*a2*a2-a2*A2*A2))]
    ])



#G2(s)
def G2(x):
    x1, x2 = x[0], x[1]
    s1, s2 = S(x)
    return np.array([
        [0],
        [(sin(4*x1*x1)+2)*((A2*A2*exp(-s2)-2*a2*A2+a2*a2*exp(s2))/(A2*a2*a2-a2*A2*A2))]
    ])


# Dynamics of the first player Critic weights
def wc1_dynamic(wc, t, x, u1, u2, Qi, Ri1, Ri2, ac):
    s1, s2 = S(x)
    sigma_a = gradient_phi_c(x)@(F(x)+G1(x)*u1+G2(x)*u2)
    sigma_a1_k.append(sigma_a)
    r_a = np.array([[s1], [s2]]).T@Qi@np.array([[s1], [s2]]) + u1*Ri1*u1 + u2*Ri2*u2
    e_a_c = r_a + np.array(wc).T@sigma_a
    e_a_c1_k.append(e_a_c)
    
    l = len(sigma_a1_k)
    pp = l
    if l > p:
        pp = p
    
    sigmaK = sigma_a1_k[l-pp:l]
    eacK = e_a_c1_k[l-pp:l]
    dwc = -ac*(sigma_a*e_a_c)/(1+sigma_a.T@sigma_a)**2
    for j in range(len(eacK)):
        dwc += -ac*sigmaK[j]@eacK[j]/(1+sigmaK[j].T@sigmaK[j])**2

    return dwc.reshape(1, 8)[0].tolist()


# Dynamics of the second player Critic weights
def wc2_dynamic(wc, t, x, u1, u2, Qi, Ri1, Ri2, ac):
    s1, s2 = S(x)
    sigma_a = gradient_phi_c(x)@(F(x)+G1(x)*u1+G2(x)*u2)
    sigma_a2_k.append(sigma_a)
    r_a = np.array([[s1], [s2]]).T@Qi@np.array([[s1], [s2]]) + u1*Ri1*u1 + u2*Ri2*u2
    e_a_c = r_a + np.array(wc).T@sigma_a
    e_a_c2_k.append(e_a_c)
    l = len(sigma_a2_k)
    pp = l
    if l > p:
        pp = p
    
    sigmaK = sigma_a2_k[l-pp:l]
    eacK = e_a_c2_k[l-pp:l]
    dwc = -ac*(sigma_a*e_a_c)/(1+sigma_a.T@sigma_a)**2
    for j in range(len(eacK)):
        dwc += -ac*sigmaK[j]@eacK[j]/(1+sigmaK[j].T@sigmaK[j])**2

    return dwc.reshape(1, 8)[0].tolist()

# Dynamics of the first player Actor weights
def wa1_dynamic(wa, t, x, wc, R11, aa1):
    r = np.array(wa)@phi_a(x) + (0.5/R11)*G1(x).T@gradient_phi_c(x).T@np.array(wc)
    dwa = -aa1*phi_a(x)@r.T
    return dwa

# Dynamics of the second player Actor weights
def wa2_dynamic(wa, t, x, wc, R22, aa2):
    r = np.array(wa)@phi_a(x) + (0.5/R22)*G2(x).T@gradient_phi_c(x).T@np.array(wc)
    dwa = -aa2*phi_a(x)@r.T
    return dwa


# eta function for robustifying
def eta(x,):
    s1, s2 = S(x)
    return -B*((s1*s1+s2*s2)**1)/(A+(s1*s1+s2*s2))


def safe_RL_policy(x, u1, u2, wc1, wc2, wa1, wa2):
    wc1t = odeint(wc1_dynamic, wc1, tspan, args=(x, u1, u2, Q1, R11, R12, ac1))
    wc2t = odeint(wc2_dynamic, wc2, tspan, args=(x, u1, u2, Q2, R21, R22, ac2))
    wa1t = odeint(wa1_dynamic, wa1, tspan, args=(x, wc1, R11, aa1))
    wa2t = odeint(wa2_dynamic, wa2, tspan, args=(x, wc2, R22, aa2))
    wc1 = wc1t[1]
    wc2 = wc2t[1]
    wa1 = wa1t[1]
    wa2 = wa2t[1]
    u1 = np.array(wa1).T@phi_a(x) + eta(x)
    u2 = np.array(wa2).T@phi_a(x) + eta(x)
    # print(f'wc1: {wc1}')
    # print(f'wc2: {wc2}')
    # print(f'wa1: {wa1}')
    # print(f'wa2: {wa2}')
    return u1, u2, wc1, wc2, wa1, wa2


def optimal_policy(x):
    u1 = -2*(cos(2*x[0])+2)*x[1]
    u2 = -(sin(4*x[0]**2)+2)*x[1] 
    return u1, u2


for i in range(1,n):
    # span for next time step
    tspan = [t[i-1], t[i]]


    # RL policy
    u1[i], u2[i], wc1, wc2, wa1, wa2 = safe_RL_policy(x0, u1[i-1], u2[i-1], wc1, wc2, wa1, wa2)


    # Theoretical Optimal Policy for uncontrained system 
    # u1[i], u2[i] = optimal_policy(x0)
    
    
    # solve for next step
    xt = odeint(sys_dynamic, x0, tspan, args=(u1[i], u2[i]))
    x1[i] = xt[1][0]
    x2[i] = xt[1][1]
    x0 = xt[1]

    print(f'i-------->>>>>>>{i}_________\t\t x1: {x0[0]}, x2: {x0[1]}')



# plot results
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(16)

plt.subplot(2,2,1)
plt.ylabel('u1')
plt.plot(t,u1,'g:',label='u1(t)')

plt.subplot(2,2,2)
plt.ylabel('u2')
plt.plot(t,u2,'g:',label='u2(t)')
plt.xlabel('time (s)')

plt.subplot(2,2,3)
plt.ylabel('x1')
plt.plot(t,x1,'b-',label='x1(t)')

plt.subplot(2,2,4)
plt.ylabel('x2')
plt.plot(t,x2,'r--',label='x2(t)')
plt.xlabel('time (s)')

fig0 = plt.figure()
fig0.set_figheight(8)
fig0.set_figwidth(8)
plt.plot(x1,x2,'g:')
plt.xlabel('x1')
plt.ylabel('x2')

plt.plot([-1.3, 0.5], [-3.1, -3.1], 'b')
plt.plot([-1.3, -1.3], [-3.1, 0.5], 'b')
plt.plot([-1.3, 0.5], [0.5, 0.5], 'b')
plt.plot([0.5, 0.5], [-3.1, 0.5], 'b')
plt.legend(['x1-x2', 'Safe set'])




plt.show()
