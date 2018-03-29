import numpy as np
import control_utils
import vdp_utils


def unwrap(x, ud):
    mu, xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, xb, dt, a = \
        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], ud[15]
    s = np.zeros(((n - 1) * (m - 1) + 1, nx))
    q = np.zeros((n - 1, nu))
    for k in range(0, nx):
        s[:, k] = x[(k * ns):((k + 1) * ns)]
    for k in range(0, nu):
        q[:, k] = x[(nx * ns + k * nq):(nx * ns + (k + 1) * nq)]
    return s, q


def eval_f(x, ud):
    mu, xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, xb, dt, a = \
        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], \
        ud[15]
    s, q = unwrap(x, ud)
    f = 0.0
    for k in range(0, n - 1):
        qk = q[k, ]
        for j in range(0, m - 1):
            dskj = s[k * (m - 1) + j, ] - xb[k * (m - 1) + j, ]
            f += dt[k * (m - 1) + j] * control_utils.l(dskj, qk, a)
    return f


def eval_grad_f(x, ud):
    mu, xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, xb, dt, a = \
        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], \
        ud[15]
    s, q = unwrap(x, ud)
    gs = np.zeros(((n - 1) * (m - 1) + 1, nx))
    gq = np.zeros((n - 1, nu))
    for k in range(0, n - 1):
        qk = q[k, ]
        for j in range(0, m - 1):
            dskj = s[k * (m - 1) + j, ] - xb[k * (m - 1) + j, ]
            gs[k * (m - 1) + j, ] = dt[k * (m - 1) + j] * control_utils.ldx(dskj, qk, a)
            gq[k, ] += dt[k * (m - 1) + j] * control_utils.ldu(dskj, qk, a)
    g = np.append(np.reshape(gs, (ns * nx), 'F'), np.reshape(gq, (nq * nu), 'F'))
    return g


def eval_c(x, ud):
    mu, xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, xb, dt, a = \
        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], \
        ud[15]
    s, q = unwrap(x, ud)
    c = np.zeros(nc)
    c[0:nx] = s[0, ] - xic
    # Fill dynamical equations
    for k in range(0, n - 1):
        qk = q[k, ]
        for j in range(1, m):
            tid = k * (m - 1) + j
            skj_prev = s[tid - 1, ]
            skj_next = s[tid, ]
            dtkj = dt[k * (m - 1) + j - 1]
            f_prev = vdp_utils.f(skj_prev, qk, 0.0, mu)
            c[(k * nx * (m - 1) + nx * j):(k * nx * (m - 1) + nx * (j + 1))] = \
                (skj_next - skj_prev - dtkj * f_prev)
    return c


def eval_jac_c(x, flag, ud):
    mu, xic, n, tg, m, tl, nx, nu, nv, nc, ns, nq, nz, xb, dt, a = \
        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], \
        ud[15]
    if flag:
        nzrow = np.zeros(nz, dtype=int)
        nzcol = np.zeros(nz, dtype=int)
        nzid = 0
        # Initial value constraints
        for k in range(0, nx):
            cid = k
            sid = k * ns
            nzrow[nzid] = cid
            nzcol[nzid] = sid
            nzid += 1
        # Dynamical constraints
        for k in range(0, n - 1):
            for j in range(1, m):
                for ls in range(0, nx):
                    # State derivatives
                    cid = k * nx * (m - 1) + nx * j + ls
                    sid = k * (m - 1) + j + ls * ns
                    nzrow[nzid] = cid
                    nzcol[nzid] = sid
                    nzid += 1
                    nzrow[nzid] = cid
                    nzcol[nzid] = sid - 1
                    nzid += 1
                    nzrow[nzid] = cid
                    if ls == 0:
                        nzcol[nzid] = sid + ns - 1
                    elif ls == 1:
                        nzcol[nzid] = sid - ns - 1
                    nzid += 1
                    # Control derivatives
                    qid = k + 0 * nq + nx * ns
                    nzrow[nzid] = cid
                    nzcol[nzid] = qid
                    nzid += 1
        return nzrow, nzcol
    else:
        s, q = unwrap(x, ud)
        nzval = np.zeros(nz)
        nzid = 0
        # Initial value constraints
        for k in range(0, nx):
            nzval[nzid] = 1
            nzid += 1
        # Dynamical constraints
        for k in range(0, n - 1):
            qk = q[k, ]
            for j in range(1, m):
                skj = s[k * (m - 1) + j - 1, ]
                fdx = vdp_utils.fdx(skj, qk, 0, mu)
                fdu = vdp_utils.fdu(skj, qk, 0, mu)
                dtkj = dt[k * (m - 1) + j - 1]
                for ls in range(0, nx):
                    nzval[nzid] = 1.0
                    nzid += 1
                    nzval[nzid] = -(1.0 + dtkj * fdx[ls, ls])
                    nzid += 1
                    if ls == 0:
                        nzval[nzid] = -dtkj * fdx[0, 1]
                    elif ls == 1:
                        nzval[nzid] = -dtkj * fdx[1, 0]
                    nzid += 1
                    nzval[nzid] = -dtkj * fdu[ls]
                    nzid += 1
        return nzval

