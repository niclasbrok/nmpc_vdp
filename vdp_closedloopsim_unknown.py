import scipy.integrate as sp_int
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


# Define global parameters

alpha = 0.001
mu = 1.0
tb = 10.0
xb_val = 1.0
xic = np.zeros(2) + 1.0
t0 = 0
t1 = 20
nx = 2
nu = 1

# Define sampling and simulation horizons

nobs = 80
nsim = 100
ds = 0.25
ts_end = nobs * ds
tobs = np.linspace(0, ts_end, nobs + 1)
vobs = np.eye(2) / 5
yobs = np.zeros((nobs + 1, nx))
ytrue = np.zeros((nobs + 1, nx))
ytrues = np.zeros((0, nx))
ttrues = np.zeros(0)
ypred = np.zeros((nobs + 1, nx))
zest = np.zeros((nobs + 1, nx + 1))
vest = np.zeros((3, 3, nobs + 1))
vpred = np.zeros((3, 3, nobs + 1))
nb = 2

# Define (initial) system parameters

gmod = np.zeros((2, 2))
gmod[0, 0] = 0.00
gmod[1, 1] = 0.25
gekf = np.zeros((3, 3))
gekf[0, 0] = 0.00
gekf[1, 1] = 0.25
gekf[2, 2] = 2.00
muest = mu + 0.5

# Define optimization parameters

n = 51
m = 21
tg = np.linspace(t0, t1, n)
tl = np.linspace(0, 1, m)
dt = np.zeros((n - 1) * (m - 1))
t = np.zeros((n - 1) * (m - 1) + 1)
ns = (n - 1) * (m - 1) + 1
nq = n - 1
nv = nx * ns + nu * (n - 1)
nc = nx * ns

nz = nx + 3 * nx * (n - 1) * (m - 1) + 1 * nu * nx * (m - 1) * nq

t[0] = t0
for k in range(0, n - 1):
    for j in range(1, m):
        _t = tg[k] + (tg[k + 1] - tg[k]) * tl[j]
        t[k * (m - 1) + j] = _t
        dt[k * (m - 1) + j - 1] = (tg[k + 1] - tg[k]) * (tl[j] - tl[j - 1])

gl = np.zeros(nc)
gu = np.zeros(nc)
xl = np.ones(nv) * (-1000.0)
xu = np.ones(nv) * 1000.0

qopt = np.zeros((nq, nobs))
sopt = np.zeros((ns, nx, nobs))

nlp = pyip.create(nv, xl, xu, nc, gl, gu, nz, 0,
                  ipopt_utils.eval_f,
                  ipopt_utils.eval_grad_f,
                  ipopt_utils.eval_c,
                  ipopt_utils.eval_jac_c)

nlp.int_option('print_level', 2)

# Run  closed-loop simulation

np.random.seed(1)
yobs[0, :] = xic
ytrue[0, :] = xic
ypred[0, :] = xic
zest[0, :] = np.append(xic, muest)
vest[:, :, 0] = np.zeros((3, 3))
vpred[:, :, 0] = np.zeros((3, 3))

for k in range(1, nobs + 1):
    print('Working on obs {0:4} out of {1:4}'.format(k, nobs))
    # Extract estimates from EKF
    zold = zest[k - 1, :]
    # Extract set-point
    _xb = np.zeros(((n - 1) * (m - 1) + 1, nx))
    _t0 = (k - 1) * ds
    for _k in range(0, t.size):
        _t = t[_k] + _t0
        if _t <= tb:
            _xb[_k, 0] = 0.0
            _xb[_k, 1] = 0.0
        elif tb < _t:
            _xb[_k, 0] = xb_val
            _xb[_k, 1] = 0.0
    # Make optimization
    _xic = zold[0:2]
    _mu = zold[2]
    user_data = (_mu, _xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, _xb, dt, alpha,)
    s0 = np.zeros((ns, nx))
    u0 = np.zeros((nq, nu))
    x0 = np.append(np.reshape(s0, (ns * nx,), 'F'), np.reshape(u0, (nq * nu,), 'F'))
    res = nlp.solve(x0, user_data)
    x = res[0]
    s, q = ipopt_utils.unwrap(x, user_data)
    qopt[:, k - 1] = np.squeeze(q)
    sopt[:, :, k - 1] = s

    def uopt(_t):
        _tid = ((_t - _t0) <= tg[1:]) & (tg[0:-1] <= (_t - _t0))
        _qt = q[_tid]
        return np.squeeze(_qt[0])

    # Simulate until next
    tsim = np.linspace((k - 1) * ds, k * ds, 1000)
    xr = sim_utils.sde_sim(ytrue[k - 1, :], tsim,
                           lambda _x, _t: vdp_utils.f(_x, uopt(_t), _t, mu),
                           lambda _x, _t: gmod, nb)
    xnew = xr[-1, :]
    ytrue[k, :] = xnew
    epsk = np.random.multivariate_normal(np.zeros(2), vobs)
    yobs[k, :] = xnew + epsk
    ytrues = np.vstack((ytrues, xr))
    ttrues = np.append(ttrues, tsim)
    # Apply EKF estimation (get ready for next optimization)
    y0 = np.append(zold, np.squeeze(np.reshape(vest[:, :, k - 1], (9, 1), order='F')))
    y = sp_int.odeint(lambda _y, _t: ekf_utils.fekf(_y, uopt, gekf, _t), y0, tsim)
    zpred = y[-1, 0:3]
    ypred[k, :] = zpred[0:2]
    _vpred = np.reshape(y[-1, 3:], (3, 3), order='F')
    vpred[:, :, k] = _vpred
    rzdz = np.zeros((2, 3))
    rzdz[0, 0] = 1.0
    rzdz[1, 1] = 1.0
    _tmp1 = _vpred.dot(np.transpose(rzdz))
    _tmp2 = rzdz.dot(_tmp1) + vobs
    kgain = _tmp1.dot(np.linalg.inv(_tmp2))
    zest[k, :] = zpred + kgain.dot(yobs[k, :] - zpred[0:2])
    zest[k, 2] = np.maximum(0.0, zest[k, 2])
    vest[:, :, k] = _vpred - kgain.dot(_tmp2).dot(np.transpose(kgain))


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

# Compute xb1
xb1s = np.zeros(ttrues.size)
for k in range(0, xb1s.size):
    _t = ttrues[k]
    if _t <= tb:
        xb1s[k] = 0.0
    elif tb < _t:
        xb1s[k] = xb_val

# Compute implemented control signals
us = np.zeros(ttrues.size)
for k in range(0, us.size):
    _t = ttrues[k]
    _obsid = (tobs[0:-1] <= _t) & (_t <= tobs[1:])
    _qopt = qopt[:, _obsid][:, 0]
    _t0 = tobs[0:-1][_obsid][0]

    def _uopt(__t):
        _tid = ((__t - _t0) <= tg[1:]) & (tg[0:-1] <= (__t - _t0))
        _qt = _qopt[_tid]
        return np.squeeze(_qt[0])

    us[k] = _uopt(_t)

gs = gridspec.GridSpec(2, 1)
gs.update(hspace=0.00)

# Plot state- and control signals

fig = plt.figure(figsize=(18, 7))

ax1 = fig.add_subplot(gs[0])

ax1.plot(ttrues, xb1s, label='$\overline{x}_1$', color='black', linewidth=lw)
ax1.plot(ttrues, ytrues[:, 0], label='$x_1$', color=cmap_blue(0.80), linewidth=lw - 2)
ax1.plot(tobs, yobs[:, 0], 'o', label='$x_{1,obs}$', color=cmap_green(0.80))
ax1.plot(tobs, zest[:, 0], 'x', label='$x_{1,est}$', color=cmap_red(0.80))
ax1.set_title('Set-Point Tracking ($\lambda$ unknown)')
ax1.set_ylabel('$x_1$')
ax1.tick_params(labelbottom='off')
ax1.set_xlim(left=t0 - 0.2, right=ts_end + 0.2)
ax1.grid()
ax1.legend(loc='lower right', ncol=4)

ax2 = fig.add_subplot(gs[1])
ax22 = ax2.twinx()

ax22.step(ttrues, us, label='$u$', color='black', linewidth=lw - 2)
ax22.set_ylabel('$u$')
ax22.set_ylim(top=10, bottom=-10)
ax2.plot(ttrues, ytrues[:, 1], label='$x_2$', color=cmap_blue(0.80), linewidth=lw - 2)
ax2.plot(tobs, yobs[:, 1], 'o', label='$x_{2,obs}$', color=cmap_green(0.80))
ax2.plot(tobs, zest[:, 1], 'x', label='$x_{2,est}$', color=cmap_red(0.80))
ax2.set_ylabel('$x_2$')
ax2.set_xlabel('Time')
ax2.set_xlim(left=t0 - 0.2, right=ts_end + 0.2)
ax2.grid()
ax2.legend(loc='lower right', ncol=4)
ax22.legend(loc='upper right')

fig.savefig('/Users/nlbr/Dropbox/DTU Niclas Laursen Brok/papers/NMPC 2018/latex/fig/results/vdp_closedloop_unknown_fig1.eps',
            format='eps', dpi=1000)


# Plot state-space in time-interval near set-point change

tlowid = 0
tuppid = 80
tlow = tobs[tlowid]
tupp = tobs[tuppid]
_sopt = sopt[:, :, tlowid]
_tsid = (tlow <= ttrues) & (ttrues <= tupp)

fig = plt.figure(figsize=(18, 7))

ax1 = fig.add_subplot(111)

ax1.plot(_sopt[0:(t.size - 50), 0], _sopt[0:(t.size - 50), 1], color='black', linewidth=lw - 1, label='Expected Trajectory')
ax1.plot(ytrues[_tsid, 0], ytrues[_tsid, 1], color=cmap_red(0.80), linewidth=lw - 1, label='Realiaed Trajectory')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('State-Space ($\lambda$ unknown)'.format(tlow, tupp))
ax1.legend(loc='lower right')
ax1.grid()

fig.savefig('/Users/nlbr/Dropbox/DTU Niclas Laursen Brok/papers/NMPC 2018/latex/fig/results/vdp_closedloop_unknown_fig2.eps',
            format='eps', dpi=1000)

# Plot lambda estimate

fig = plt.figure(figsize=(18, 7))

ax1 = fig.add_subplot(111)
_muest = np.append(zest[0, 2], zest[0:-1, 2])

ax1.step(tobs, mu * np.ones(tobs.size), label='$\lambda$', color='black', linewidth=lw - 2)
ax1.step(tobs, _muest, label='$\lambda_{1,est}$', color=cmap_blue(0.80), linewidth=lw - 2)
ax1.set_title('Set-Point Tracking ($\lambda$ unknown)')
ax1.set_ylabel('$x_1$')
ax1.set_xlim(left=t0 - 0.2, right=ts_end + 0.2)
ax1.set_ylim(bottom=0.0)
ax1.set_xlabel('Time')
ax1.set_ylabel('$\lambda$')
ax1.grid()
ax1.legend(loc='lower left', ncol=2)

fig.savefig('/Users/nlbr/Dropbox/DTU Niclas Laursen Brok/papers/NMPC 2018/latex/fig/results/vdp_closedloop_unknown_fig3.eps',
            format='eps', dpi=1000)
