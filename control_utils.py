import numpy as np


def l(x, u, alpha):
    g = 0.5 * ((1 - alpha) * x[0] ** 2 + alpha * u ** 2)
    return g


def ldx(x, u, alpha):
    ldx = np.zeros((1, 2))
    ldx[0, 0] = (1 - alpha) * x[0]
    ldx[0, 1] = 0.0
    return ldx


def ldu(x, u, alpha):
    return alpha * u

