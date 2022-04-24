import numpy as np
import scipy.sparse as sp
import osqp

from constants import *


def calc_A_dyn_t(f1tx, f1ty, f2tx, f2ty):
    A_dyn_t = sp.lil_matrix((dim_dyn_cqp, dim_x_cqp * 2))
    A_dyn_t[0, 0] = 1.0
    A_dyn_t[0, 8] = -1.0
    A_dyn_t[0, 10] = dt / m
    A_dyn_t[1, 1] = 1.0
    A_dyn_t[1, 9] = -1.0
    A_dyn_t[1, 11] = dt / m
    A_dyn_t[2, 8] = -dt * (f1ty + f2ty)
    A_dyn_t[2, 9] = dt * (f1tx + f2tx)
    A_dyn_t[2, 12] = dt * f1ty
    A_dyn_t[2, 13] = -dt * f1tx
    A_dyn_t[2, 14] = dt * f2ty
    A_dyn_t[2, 15] = -dt * f2tx

    return A_dyn_t


def calc_l_dyn_t(kt, kt_prev):
    l_dyn_t = np.array([0, 0, kt - kt_prev])
    return l_dyn_t


def calc_A_loc_t():
    A_loc_t = sp.lil_matrix((dim_loc_cqp, dim_x_cqp))
    A_loc_t[:, 4:] = sp.identity(4)
    return A_loc_t


def calc_l_loc_t(p1tx_prev, p2tx_prev):
    l_loc_t = np.array([p1tx_prev, 0, p2tx_prev, 0])
    return l_loc_t


def calc_A_kin_t():
    A_kin_t = sp.lil_matrix((dim_kin_cqp, dim_x_cqp))
    A_kin_t[:, 0] = [-1, -1, 1, 1, -1, -1, 1, 1]
    A_kin_t[:, 1] = [-1, 1, -1, 1, -1, 1, -1, 1]
    A_kin_t[:, 4] = [1, 1, -1, -1, 0, 0, 0, 0]
    A_kin_t[:, 5] = [1, -1, 1, -1, 0, 0, 0, 0]
    A_kin_t[:, 6] = [0, 0, 0, 0, 1, 1, -1, -1]
    A_kin_t[:, 7] = [0, 0, 0, 0, 1, -1, 1, -1]

    return A_kin_t


def calc_P():
    P = sp.lil_matrix(((N + 1) * dim_x_cqp, (N + 1) * dim_x_cqp))
    diag_P = np.array(
        [
            L_r,
            L_r,
            L_l,
            L_l,
            L_p,
            L_p,
            L_p,
            L_p,
        ]
    )
    P.setdiag(np.tile(diag_P, N + 1))
    return P


def calc_q_t(rt_fcp, lt_fcp, p1t_fcp, p2t_fcp):
    q_t = np.array(
        [
            -L_r * rt_fcp[0],
            -L_r * rt_fcp[1],
            -L_l * lt_fcp[0],
            -L_l * lt_fcp[1],
            -L_p * p1t_fcp[0],
            -L_p * p1t_fcp[1],
            -L_p * p2t_fcp[0],
            -L_p * p2t_fcp[1],
        ]
    )
    return q_t


def solve_contact_qp(X_prev):
    # extract relevant trajectories from input
    r_fqp = X_prev[0:2, :]
    l_fqp = X_prev[2:4, :]
    k_fqp = X_prev[5, :]
    p1_fqp = X_prev[6:8, :]
    p2_fqp = X_prev[8:10, :]
    f1_fqp = X_prev[10:12, :]
    f2_fqp = X_prev[12:, :]

    # constraints
    A = sp.lil_matrix((15 * N + 12, dim_x_cqp * (N + 1)))
    l = np.empty(15 * N + 12)
    u = np.empty(15 * N + 12)

    # dynamics constraints
    for idx in np.arange(N):
        f1t = f1_fqp[:, idx + 1]
        f2t = f2_fqp[:, idx + 1]
        A_dyn_t = calc_A_dyn_t(f1t[0], f1t[1], f2t[0], f2t[1])
        kt = k_fqp[idx + 1]
        kt_prev = k_fqp[idx]
        l_dyn_t = calc_l_dyn_t(kt, kt_prev)
        u_dyn_t = l_dyn_t

        row_indices = (idx * dim_dyn_cqp, (idx + 1) * dim_dyn_cqp)
        col_indices = (idx * dim_x_cqp, idx * dim_x_cqp + dim_x_cqp * 2)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_dyn_t
        l[row_indices[0] : row_indices[1]] = l_dyn_t
        u[row_indices[0] : row_indices[1]] = u_dyn_t

    # foot location constraints
    for t in np.arange(N + 1):
        p1tx_prev = p1_fqp[0, t]
        p2tx_prev = p2_fqp[0, t]

        A_loc_t = calc_A_loc_t()
        l_loc_t = calc_l_loc_t(p1tx_prev, p2tx_prev)
        u_loc_t = l_loc_t

        row_indices = (
            N * dim_dyn_cqp + t * dim_loc_cqp,
            N * dim_dyn_cqp + (t + 1) * dim_loc_cqp,
        )
        col_indices = (t * dim_x_cqp, (t + 1) * dim_x_cqp)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_loc_t
        l[row_indices[0] : row_indices[1]] = l_loc_t
        u[row_indices[0] : row_indices[1]] = u_loc_t

    # kinematic constraints
    for t in np.arange(N + 1):
        A_kin_t = calc_A_kin_t()
        l_kin_t = np.full(dim_kin_cqp, -np.inf)
        u_kin_t = np.full(dim_kin_cqp, Lmax)

        row_indices = (
            N * dim_dyn_cqp + (N + 1) * dim_loc_cqp + t * dim_kin_cqp,
            N * dim_dyn_cqp + (N + 1) * dim_loc_cqp + (t + 1) * dim_kin_cqp,
        )
        col_indices = (t * dim_x_cqp, (t + 1) * dim_x_cqp)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_kin_t
        l[row_indices[0] : row_indices[1]] = l_kin_t
        u[row_indices[0] : row_indices[1]] = u_kin_t

    # objective
    P = calc_P()
    q = np.empty((N + 1) * dim_x_cqp)
    for t in np.arange(N + 1):
        rt_fcp = r_fqp[:, t]
        lt_fcp = l_fqp[:, t]
        p1t_fcp = p1_fqp[:, t]
        p2t_fcp = p2_fqp[:, t]
        q_t = calc_q_t(rt_fcp, lt_fcp, p1t_fcp, p2t_fcp)

        row_indices = (t * dim_x_cqp, (t + 1) * dim_x_cqp)

        q[row_indices[0] : row_indices[1]] = q_t

    qp = osqp.OSQP()
    settings = {}

    qp.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=l, u=u, **settings)
    results = qp.solve()

    X_sol_cqp = results.x.reshape((dim_x_cqp, N + 1), order="F")

    X_sol = np.empty((dim_x, N + 1))
    X_sol[:4, :] = X_sol_cqp[:4, :]  # r, l
    X_sol[4:6, :] = X_prev[4:6, :]  # theta, k
    X_sol[6:10, :] = X_sol_cqp[4:8, :]  # p
    X_sol[10:, :] = X_prev[10:, :]  # f

    return X_sol


if __name__ == "__main__":
    from draw import animate

    X = np.empty((dim_x, N + 1))
    for t in np.arange(N + 1):
        r = np.array([0.1 * np.cos(t / 10), 0.1 + 0.05 * np.sin(t / 10)])
        l = np.array([0.0, 0.0])
        th = np.pi / 8 * np.sin(t / 14)
        k = 0.0
        p1 = np.array([-0.15, 0])
        p2 = np.array([0.15, 0])
        f1 = np.array([0, m * g / 2])
        f2 = np.array([0, m * g / 2])

        X[0:2, t] = r
        X[2:4, t] = l
        X[4, t] = th
        X[5, t] = k
        X[6:8, t] = p1
        X[8:10, t] = p2
        X[10:12, t] = f1
        X[12:14, t] = f2

    X_sol = solve_contact_qp(X)

    animate(X)
    animate(X_sol)
