import scipy.integrate as sp_int
import numpy as np
import pyipopt as pyip
import control_utils
import vdp_utils
import sim_utils
import ipopt_utils
import matplotlib
try:
    matplotlib.use('PyQt4')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


# Define variable consistent in both methods

alpha = 0.001
mu = 1.0
tb = 8.0
xb_val = 0.2
xic = np.zeros(2) + 0.1
t0 = 0
t1 = 20
nx = 2
nu = 1

# Solve optimal control via Pontryagin's minimums principle


def fopt_wrap(t, y):
    fopt_wrap = np.zeros(y.shape)
    for k in range(0, t.size):
        tk = t[k]
        xbtk = np.zeros(2)
        if tk <= tb:
            xbtk[0] = 0.0
        elif tb < tk:
            xbtk[0] = xb_val
        ytk = y[:, k]
        xtk = ytk[0:2]
        dxtk = xtk - xbtk
        ptk = ytk[2:4]
        utk = -ptk[1] / (2 * alpha)
        fopttk = np.zeros(y.shape[0])
        fopttk[0] = xtk[1]
        fopttk[1] = -xtk[0] + mu * (1.0 - xtk[0] ** 2) * xtk[1] + utk
        fopttk[2] = -(2 * (1 - alpha) * dxtk[0] - ptk[1] * (1.0 + 2 * xtk[0] * xtk[1]))
        fopttk[3] = -(ptk[0] + ptk[1] * (1.0 - xtk[0] ** 2))
        fopt_wrap[:, k] = fopttk
    return fopt_wrap


def bcopt_wrap(y0, y1):
    bcopt_wrap = np.zeros(y1.size)
    x0 = y0[0:2]
    p1 = y1[2:4]
    bcopt_wrap[0:2] = x0 - xic
    bcopt_wrap[2:4] = p1
    return bcopt_wrap


nt = 2000
tbvp = np.linspace(t0, t1, nt)
x0 = np.squeeze(xic)
p0 = np.squeeze(xic)
y0 = np.array([x0[0] * np.ones(nt), x0[1] * np.ones(nt), p0[0] * np.zeros(nt), p0[1] * np.zeros(nt)])
res = sp_int.solve_bvp(fopt_wrap, bcopt_wrap, tbvp, y0)

xb = np.zeros(nt)
for k in range(0, nt):
    tk = tbvp[k]
    if tk <= tb:
        xb[k] = 0.0
    elif tb < tk:
        xb[k] = xb_val
topt = res.x
yopt = res.y
xopt = yopt[0:2, :]
popt = yopt[2:4, :]
uopt = -popt[1, :] / (2 * alpha)


# Solve via local collocation
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

# Compute set-point values
xb = np.zeros(((n - 1) * (m - 1) + 1, nx))
t[0] = t0
for k in range(0, n - 1):
    for j in range(1, m):
        _t = tg[k] + (tg[k + 1] - tg[k]) * tl[j]
        t[k * (m - 1) + j] = _t
        _xb = np.zeros(nx)
        if _t <= tb:
            xb[k * (m - 1) + j, :] = _xb
        elif tb < _t:
            _xb[0] = xb_val
            xb[k * (m - 1) + j, :] = _xb
        dt[k * (m - 1) + j - 1] = (tg[k + 1] - tg[k]) * (tl[j] - tl[j - 1])

# Get ready for optimization
user_data = (mu, xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, xb, dt, alpha, )

s0 = -np.ones((ns, nx))
s0[:, 1] = 2 * s0[:, 1]
u0 = np.zeros((nq, nu))
x0 = np.append(np.reshape(s0, (ns * nx, ), 'F'), np.reshape(u0, (nq * nu, ), 'F'))

# Solve problem using ipopt

gl = np.zeros(nc)
gu = np.zeros(nc)
xl = np.ones(nv) * (-1000.0)
xu = np.ones(nv) * 1000.0

ipopt_utils.eval_f(x0, user_data)
ipopt_utils.eval_grad_f(x0, user_data)
ipopt_utils.eval_c(x0, user_data)
ipopt_utils.eval_jac_c(x0, False, user_data)

nlp = pyip.create(nv, xl, xu, nc, gl, gu, nz, 0,
                  ipopt_utils.eval_f,
                  ipopt_utils.eval_grad_f,
                  ipopt_utils.eval_c,
                  ipopt_utils.eval_jac_c)
res = nlp.solve(x0, user_data)
x = res[0]
s, q = ipopt_utils.unwrap(x, user_data)
qt = np.append(q[0], q)

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
ax1.plot(t, xb[:, 0], label='$\overline{x}_1$', linewidth=lw, color=cmap_blue(0.80))
ax1.plot(topt, xopt[0, :], label='$x_{1,true}$', linewidth=lw, color=cmap_green(0.80))
ax1.plot(t, s[:, 0], label='$x_{1,ipopt}$', linewidth=lw, color=cmap_red(0.80), linestyle='-')
ax1.tick_params(labelbottom='off')
ax1.set_ylabel('$x_1$')
ax1.set_xlim(left=t0 - 0.1, right=t1 + 0.1)
ax1.set_title('Verification of the Local Collocation Method')
ax1.grid()
ax1.legend(loc='lower right', ncol=3)

ax2 = fig.add_subplot(212)
ax2.plot(topt, uopt, label='$u_{true}$', linewidth=lw, color=cmap_blue(0.80))
ax2.step(tg, qt, label='$u_{ipopt}$', linewidth=lw, color=cmap_red(0.80), linestyle='-')
ax2.set_ylabel('$u$')
ax2.set_xlabel('Time')
ax2.set_xlim(left=t0 - 0.1, right=t1 + 0.1)
ax2.set_ylim(bottom=-4.1, top=2.1)
ax2.grid()
ax2.legend(loc='upper right', ncol=2)

fig.savefig('/Users/nlbr/Dropbox/DTU Niclas Laursen Brok/papers/NMPC 2018/latex/fig/results/vdp_verification.eps',
            format='eps', dpi=1000)
