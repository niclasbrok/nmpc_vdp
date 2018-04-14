import scipy.integrate as sp_int
import scipy.optimize as sp_opt
import numpy as np
import pyipopt as pyip
import vdp_utils
import sim_utils
import ipopt_utils
import ekf_utils
import matplotlib
try:
    matplotlib.use('PyQt4')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cmx
import pickle


def log_likelihood(par, args):
    yobs, nobs, x0, gekf, tsim, vobs, tobs = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    zest = np.append(x0, par)
    vest = np.zeros((3, 3))
    log_lik = 0.0
    for k in range(0, nobs):
        y0 = np.append(zest, np.squeeze(np.reshape(vest, (9, 1), order='F')))
        y = sp_int.odeint(lambda _y, _t: ekf_utils.fekf(_y, lambda __t: 0, gekf, _t), y0, tobs[k] + tsim)
        zpred = y[-1, 0:3]
        vpred = np.reshape(y[-1, 3:], (3, 3), order='F')
        hzdz = np.zeros((2, 3))
        hzdz[0, 0] = 1.0
        hzdz[1, 1] = 1.0
        _tmp1 = vpred.dot(np.transpose(hzdz))
        _tmp2 = hzdz.dot(_tmp1) + vobs
        kgain = _tmp1.dot(np.linalg.inv(_tmp2))
        errk = yobs[k, :] - hzdz.dot(zpred)
        zest = zpred + kgain.dot(errk)
        # Joseph update
        _tmp3 = kgain.dot(hzdz)
        _tmp4 = np.eye(3) - _tmp3
        _tmp5 = np.dot(_tmp4, vpred).dot(np.transpose(_tmp4))
        vest = _tmp5 + kgain.dot(vobs).dot(np.transpose(kgain))
        # Compute log-likelihood
        rkk = hzdz.dot(vpred).dot(np.transpose(hzdz)) + vobs
        log_lik += np.log(np.linalg.det(rkk)) + np.transpose(errk).dot(np.linalg.inv(rkk)).dot(errk)
    return log_lik

vobs = np.zeros((2, 2, 2))
sigobs = 1 / 10000
vobs = np.eye(2) * sigobs
nobs = 100
x0 = np.zeros(2) + 1.0
t0 = 0
t1 = 20
nt = 10000
mu = np.array([1, 10])
ts = np.linspace(t0, t1, nt)
tobs = np.linspace(t0, t1, nobs)
tsim = np.linspace(tobs[0], tobs[1], nt)
gekf = np.zeros((3, 3))
sigmod = 0.20
gekf[1, 1] = sigmod

muvec_ns = np.linspace(0.80 * mu[0], 1.20 * mu[0], 100)
muvec_s = np.linspace(0.80 * mu[1], 1.20 * mu[1], 100)
loglikval = np.zeros((muvec_ns.size, 2))
np.random.seed(1)

xr = sim_utils.sde_sim(x0, ts,
                       lambda _x, _t: vdp_utils.f(_x, 0, _t, mu[0]),
                       lambda _x, _t: gekf[0:2, 0:2], 2)
ytrue = xr[0::101]
yobs = np.zeros((nobs, 2))
for k in range(0, nobs):
    epsk = np.random.multivariate_normal(np.zeros(2), vobs[:, :])
    yobs[k, :] = ytrue[k, :] + epsk
args = (yobs, nobs, x0, gekf[:, :], tsim, vobs[:, :], tobs)
muopt_ns = sp_opt.minimize(log_likelihood, np.array([mu[0]]), args=(args, ))
for k in range(0, muvec_ns.size):
    if np.mod(k + 1, 50) == 0:
        print('Working on obs {0:d}'.format(k + 1))
    loglikval[k, 0] = log_likelihood(muvec_ns[k], args)

xr = sim_utils.sde_sim(x0, ts,
                       lambda _x, _t: vdp_utils.f(_x, 0, _t, mu[1]),
                       lambda _x, _t: gekf[0:2, 0:2], 2)
ytrue = xr[0::101]
yobs = np.zeros((nobs, 2))
for k in range(0, nobs):
    epsk = np.random.multivariate_normal(np.zeros(2), vobs[:, :])
    yobs[k, :] = ytrue[k, :] + epsk
args = (yobs, nobs, x0, gekf[:, :], tsim, vobs[:, :], tobs)
muopt_s = sp_opt.minimize(log_likelihood, np.array([mu[1]]), args=(args, ))
for k in range(0, muvec_s.size):
    if np.mod(k + 1, 50) == 0:
        print('Working on obs {0:d}'.format(k + 1))
    loglikval[k, 1] = log_likelihood(muvec_s[k], args)


cmap_blue = cmx.get_cmap('Blues')
cmap_green = cmx.get_cmap('Greens')
cmap_red = cmx.get_cmap('Reds')
lw = 4.0
fs = 15
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.rc('axes', titlesize=fs + 15)
matplotlib.rc('axes', labelsize=fs + 15)
matplotlib.rc('xtick', labelsize=fs + 8)
matplotlib.rc('ytick', labelsize=fs + 8)
matplotlib.rc('legend', fontsize=fs + 8)
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Cambria']})
matplotlib.rc('text', usetex=True)

gs = gridspec.GridSpec(2, 1)

fig = plt.figure(figsize=(18, 7))

ax1 = fig.add_subplot(gs[0])

ax1.plot(muvec_ns, loglikval[:, 0], label='log-likelihood ($\lambda=1$)', linewidth=lw)
ax1.plot(muopt_ns.x[0], muopt_ns.fun, 'o', label='ML estimate ($\hat{\lambda}=1.017$)', markersize=10, color='black')
ax1.set_ylim(top=-480, bottom=-520)
ax1.set_yticks(np.array([-520, -510, -500, -490, -480]))
ax1.legend(loc='upper center')
ax1.grid()

ax2 = fig.add_subplot(gs[1])
ax2.plot(muvec_s, loglikval[:, 1], label='log-likelihood ($\lambda=10$)', linewidth=lw)
ax2.plot(muopt_s.x[0], muopt_s.fun, 'o', label='ML estimate ($\hat{\lambda}=9.953$)', markersize=10, color='black')
ax2.legend(loc='upper center')
ax2.set_ylim(top=2000, bottom=-1500)
ax2.set_xlabel('$\lambda$')
ax2.set_yticks(np.array([-1500, -1000, -500, 0, 500, 1000, 1500, 2000]))
ax2.grid()

fig.suptitle('Offline Parameter Estimation', fontsize=fs + 20)

fig.savefig('/Users/nlbr/Dropbox/DTU Niclas Laursen Brok/papers/NMPC 2018/latex_nlbr_v2/fig/results/vdp_offline_estimation.eps',
            format='eps', dpi=1000, bbox_inches='tight')
