import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import sdeint
import os

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(curr_dir, "graphics")

############### parameters, tau from Ghil (1976) 
T1 = 280
T2 = 285
T3 = 290
C = 2.1e8
eps = 235
tau = 6.4
gamma = 0.0005
mu = 1
period = 10e5
omega = 2*np.pi/period
sigma = np.sqrt(0.14)

############## beta comp, from eq. 8 in Berti et al. (1982)
r = eps/C*31557000 # seconds in a year
beta = -T3/(tau * r * ((1-T3/T1)*(1-T3/T2)))

def g(T):
    return (1-T/T1)*(1-T/T2)*(1-T/T3)

def det_drift(T, mu):
    return r*(mu/(1 + beta*g(T)) -1)

def F(T, t):
    return det_drift(T, 1 + gamma*np.cos(omega*t))

def phi(T_grid, mu):
    d = det_drift(T_grid, mu)
    phi = - np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) / 2*np.diff(T_grid))])
    return phi - phi[np.argmin(np.abs(T_grid - T3))]

############### det solution
num_starts = 20
t_start = np.linspace(270, 300, num_starts)
sols = []
def det_fun(t, y):
    return [F(y[0], t)]
tot_time = 55
t_eval = np.arange(0, tot_time, 2)
for i in t_start:
    sols.append(solve_ivp(det_fun, (0, 1.5e6), [i], t_eval=t_eval, method='LSODA', rtol=1e-10, atol=1e-10))

############## noisy solution
np.random.seed(82)
t_span = np.arange(0.0, 7500000, 0.5)
def det_drift_fun(y, t):
    return np.array([F(y[0], t)])
def noise_fun(y, t):
    return np.array([[sigma]])
noisy_sol = sdeint.itoEuler(det_drift_fun, noise_fun, np.array([T3]), t_span)

############### Plots

# pseudo potential
T_space = np.linspace(277, 293, 10000)
Phi = phi(T_space, mu)
dPhi = - det_drift(T_space, mu)

fig1, ax1 = plt.subplots()
ax1.plot(T_space, phi(T_space, mu), color='black')
ax1.plot(T1, 0.005, "o", ms = 9, color='black')
ax1.plot(T2, 0.49, "o", ms = 9, color='white', mec='black')
ax1.plot(T3, 0.005, "o", ms = 9, color='black')
ax1.set_ylabel(r"$\Phi$", rotation = 'horizontal', labelpad = 10)
ax1.set_xlabel(r"$T$ (K)")
ax1.set_xticks([T1,T2,T3], [r"$T_1 = 280$", r"$T_2 = 285$", r"$T_3 = 290$"])
fig1.savefig(os.path.join(save_dir, "dw_potential.pdf"), bbox_inches="tight")

# Noiseless convergence
fig2, ax2 = plt.subplots()
for i in range(num_starts):
    ax2.plot(sols[i].t, sols[i].y[0])
ax2.set_yticks([T1,T2,T3], [r"$T_1 = 280$", r"$T_2 = 285$", r"$T_3 = 290$"])
ax2.hlines([T1,T3], colors=['black', 'black'], xmin=[0, 0], xmax=[tot_time-1, tot_time-1], 
           linestyles=['--', '--'])
fig2.savefig(os.path.join(save_dir, "det_conv.pdf"), bbox_inches="tight")

# Noisy time
fig3, ax3 = plt.subplots()
mid, amp = 0.5*(T1+T3), 0.5*(T3-T1)
ax3.plot(t_span, noisy_sol[:, 0], lw=0.5, color='grey', label="T(t)")
ax3.plot(t_span, mid + amp*np.cos(omega*t_span),
         color='black', alpha=0.6, label="forcing oscillation")
#ax3.set_aspect(1)
fig3.savefig(os.path.join(save_dir, "noise_timecourse.pdf"), bbox_inches="tight")

plt.show()
