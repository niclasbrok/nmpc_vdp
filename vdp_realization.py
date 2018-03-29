import numpy as np
import vdp_utils
import sim_utils
import random
import matplotlib
try:
    matplotlib.use('PyQt4')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


# Simulate non-stiff realization (deterministic + stochastic) for u = 0

mu = 1.0
t0 = 0.0
t1 = 24.0
nt = 20000
nsim = 3
t = np.linspace(t0, t1, nt)
x0 = np.zeros(2) + 2.0
u = 0

# Re-define function and make reade for function calls


def _f(_x, _t):
    return vdp_utils.f(_x, u, _t, mu)


def _jac(_x, _t):
    return vdp_utils.fdx(_x, u, _t, mu)


# Run simulations

# Deterministic

dsol = sim_utils.ode_sim(x0, t, _f)

# Stochastic (Euler-Maruyama scheme)

sig = 1 / 3
random.seed(1)


def _g(_x, _t):
    _g = np.zeros(2)
    _g[1] = sig
    return _g


ssol = np.zeros((nt, 2, nsim))
for k in range(0, nsim):
    ssol[:, :, k] = sim_utils.sde_sim(x0, t, _f, _g, 1)


# Plot
cmap_blue = cmx.get_cmap('Blues')
cmap_green = cmx.get_cmap('Greens')
cmap_red = cmx.get_cmap('Reds')
lw = 4.0
fs = 15
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.rc('axes', titlesize=fs + 15)
matplotlib.rc('axes', labelsize=fs + 5)
matplotlib.rc('legend', fontsize=fs)

fig = plt.figure(figsize=(18, 7))
ax1 = fig.add_subplot(211)
ax1.plot(t, ssol[:, 0, 0], label='$x_{1,SDE}$', linewidth=lw - 2, color=cmap_blue(0.80), linestyle='-')
ax1.plot(t, ssol[:, 1, 0], label='$x_{2,SDE}$', linewidth=lw - 2, color=cmap_green(0.80), linestyle='-')
for k in range(1, nsim):
    ax1.plot(t, ssol[:, 0, k], linewidth=lw - 2, color=cmap_blue(0.80), linestyle='-')
    ax1.plot(t, ssol[:, 1, k], linewidth=lw - 2, color=cmap_green(0.80), linestyle='-')
ax1.plot(t, dsol[:, 0], label='$x_{1,ODE}$', linewidth=lw, color=cmap_blue(1.00))
ax1.plot(t, dsol[:, 1], label='$x_{2,ODE}$', linewidth=lw, color=cmap_green(1.00))
a1_ylim = ax1.get_ylim()
ax1.set_ylim(ymax=a1_ylim[1] + 0.75)
ax1.set_title('Stochastic Realizations')
ax1.set_xlabel('Time', labelpad=-15)
ax1.set_ylabel('$x_1$, $x_2$')
ax1.grid()
ax1.legend(loc='upper left', ncol=4)

ax2 = fig.add_subplot(212)
tidf = 7000
ax2.plot(ssol[0:tidf, 0, 0], ssol[0:tidf, 1, 0], label='SDE', linewidth=lw - 1, color=cmap_blue(0.80))
for k in range(1, nsim):
    ax2.plot(ssol[0:tidf, 0, k], ssol[0:tidf, 1, k], linewidth=lw - 1, color=cmap_blue(0.80))
ax2.plot(dsol[0:tidf, 0], dsol[0:tidf, 1], label='ODE', linewidth=lw - 1, color=cmap_red(0.80))
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.grid()
ax2.legend(loc='upper left')

fig.savefig('/Users/nlbr/Dropbox/DTU Niclas Laursen Brok/papers/NMPC 2018/latex/fig/results/vdp_realization.eps',
            format='eps', dpi=1000)
