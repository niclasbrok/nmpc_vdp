import numpy as np


def f(x, u, t, mu):
    f = np.zeros(2)
    f[0] = x[1]
    f[1] = -x[0] + mu * (1.0 - x[0] * x[0]) * x[1] + u
    return f


def fdx(x, u, t, mu):
    fdx = np.zeros((2, 2))
    fdx[0, 0] = 0.0
    fdx[0, 1] = 1.0
    fdx[1, 0] = -(2.0 * mu * x[0] * x[1] + 1)
    fdx[1, 1] = mu * (1.0 - x[0] * x[0])
    return fdx


def fdu(x, u, t, mu):
    fdu = np.zeros(2)
    fdu[0] = 0.0
    fdu[1] = 1.0
    return fdu

