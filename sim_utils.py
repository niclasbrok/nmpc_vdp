import numpy as np


def brownian_motion(t, nb):
    b = np.zeros((t.size, nb))
    for k in range(1, t.size):
        dt = t[k] - t[k - 1]
        b[k, :] = np.random.normal(b[k - 1, :], np.sqrt(dt))
    return b


def sde_sim(x0, t, f, g, nb):
    x = np.zeros((t.size, x0.size))
    x[0, :] = x0
    b = brownian_motion(t, nb)
    for k in range(1, t.size):
        t_old = t[k - 1]
        t_new = t[k]
        x_old = x[k - 1, :]
        f_old = f(x_old, t_old)
        g_old = g(x_old, t_old)
        dt = t_new - t_old
        db = np.squeeze(b[k, :] - b[k - 1, :])
        x[k, :] = x[k - 1, :] + f_old * dt + np.dot(g_old, db)
    return x