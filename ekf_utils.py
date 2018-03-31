import vdp_utils
import numpy as np


def fz(z, u, t):
    x = z[0:2]
    mu = z[2]
    fx = vdp_utils.f(x, u, t, mu)
    fz = np.append(fx, 0.0)
    return fz


def fzdz(z, u, t):
    x = z[0:2]
    mu = z[2]
    fzdz = np.zeros((3, 3))
    fzdz[0, 0] = 0.0
    fzdz[0, 1] = 1.0
    fzdz[0, 2] = 0.0
    fzdz[1, 0] = -(2.0 * mu * x[0] * x[1] + 1.0)
    fzdz[1, 1] = mu * (1.0 - x[0] * x[0])
    fzdz[1, 2] = (1.0 - x[0] * x[0]) * x[1]
    fzdz[2, 0] = 0.0
    fzdz[2, 1] = 0.0
    fzdz[2, 2] = 0.0
    return fzdz


def fgamma(z, gamma, u, g, t):
    _fzdz = fzdz(z, u, t)
    fgamma = np.dot(_fzdz, gamma) + np.dot(gamma, np.transpose(_fzdz)) + np.dot(g, np.transpose(g))
    return fgamma


def fekf(zgamma, u, g, t):
    ut = u(t)
    z = zgamma[0:3]
    gamma = np.reshape(zgamma[3:], (3, 3), order='F')
    _fz = fz(z, ut, t)
    _fgamma = fgamma(z, gamma, ut, g, t)
    fekf = np.append(_fz, np.squeeze(np.reshape(_fgamma, (9, 1), order='F')))
    return fekf

