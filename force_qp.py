import numpy as np
import scipy.sparse as sp
import osqp

from constants import *
from draw import animate


def calc_A_dyn_t(l1tx, l1ty, l2tx, l2ty):
    A_dyn_t = sp.lil_matrix((dim_dyn_fqp, dim_x_fqp * 2))
    A_dyn_t[:, :6] = sp.identity(6)
    A_dyn_t[:, 10:16] = -sp.identity(6)
    A_dyn_t[0, 12] = dt / m
    A_dyn_t[1, 13] = dt / m
    A_dyn_t[2, 16] = dt
    A_dyn_t[2, 18] = dt
    A_dyn_t[3, 17] = dt
    A_dyn_t[3, 19] = dt
    A_dyn_t[4, 15] = dt / I
    A_dyn_t[5, 16] = -dt * l1ty
    A_dyn_t[5, 17] = dt * l1tx
    A_dyn_t[5, 18] = -dt * l2ty
    A_dyn_t[5, 19] = dt * l2tx

    return A_dyn_t


def calc_A_fric_t():
    A_fric_t = sp.lil_matrix((dim_fric_fqp, dim_x_fqp))
    A_fric_t[0, 6] = -1
    A_fric_t[0, 7] = 1
    A_fric_t[1, 6] = 1
    A_fric_t[1, 7] = 1
    A_fric_t[2, 8] = -1
    A_fric_t[2, 9] = 1
    A_fric_t[3, 8] = 1
    A_fric_t[3, 9] = 1
    A_fric_t[4, 7] = 1
    A_fric_t[5, 9] = 1

    return A_fric_t


def calc_A_kin_t():
    A_kin_t = sp.lil_matrix((dim_kin_fqp, dim_x_fqp))
    A_kin_t[:, 0] = [-1, -1, 1, 1, -1, -1, 1, 1]
    A_kin_t[:, 1] = [-1, 1, -1, 1, -1, 1, -1, 1]

    return A_kin_t


def calc_u_kin_t(p1tx, p1ty, p2tx, p2ty):
    u_kin_t = np.array(
        [
            Lmax - p1tx - p1ty,
            Lmax - p1tx + p1ty,
            Lmax + p1tx - p1ty,
            Lmax + p1tx + p1ty,
            Lmax - p2tx - p2ty,
            Lmax - p2tx + p2ty,
            Lmax + p2tx - p2ty,
            Lmax + p2tx + p2ty,
        ]
    )
    return u_kin_t


def calc_A_air1_t():
    A_air_t = sp.lil_matrix((1, dim_x_fqp))
    A_air_t[0, 7] = 1
    return A_air_t


def calc_A_air2_t():
    A_air_t = sp.lil_matrix((1, dim_x_fqp))
    A_air_t[0, 9] = 1
    return A_air_t


def calc_P():
    P = sp.lil_matrix(((N + 1) * dim_x_fqp, (N + 1) * dim_x_fqp))
    diag_P = np.array(
        [
            phi_r + L_r,
            phi_r + L_r,
            phi_l + L_l,
            phi_l + L_l,
            phi_th + L_th,
            phi_k + L_k,
            psi,
            psi,
            psi,
            psi,
        ]
    )
    P.setdiag(np.tile(diag_P, N + 1))
    return P


def calc_q_t(h_prev_t, h_des_t):
    q_t = np.array(
        [
            -phi_r * h_des_t[0] - L_r * h_prev_t[0],
            -phi_r * h_des_t[1] - L_r * h_prev_t[1],
            -phi_l * h_des_t[2] - L_l * h_prev_t[2],
            -phi_l * h_des_t[3] - L_l * h_prev_t[3],
            -phi_th * h_des_t[4] - L_th * h_prev_t[4],
            -phi_k * h_des_t[5] - L_k * h_prev_t[5],
            0,
            0,
            0,
            0,
        ]
    )
    return q_t


def solve_force_qp(X_prev, h_des):
    # extract relevant trajectories from previous solution
    r_cqp = X_prev[0:2, :]
    p1_cqp = X_prev[6:8, :]
    p2_cqp = X_prev[8:10, :]
    l1_cqp = p1_cqp - r_cqp
    l2_cqp = p2_cqp - r_cqp
    h_prev = X_prev[0:6, :]

    # determine foot contact state
    c1 = p1_cqp[1,:] < eps_contact
    c2 = p2_cqp[1,:] < eps_contact
    # number of foot in air constraints to add
    C = np.sum(np.logical_not(c1)) + np.sum(np.logical_not(c2))

    # constraints
    A = sp.lil_matrix((20 * N + 14 + C, dim_x_fqp * (N + 1)))
    l = np.empty(20 * N + 14 + C)
    u = np.empty(20 * N + 14 + C)

    # dynamics constraints
    for idx in np.arange(N):
        l1t = l1_cqp[:, idx + 1]
        l2t = l2_cqp[:, idx + 1]
        A_dyn_t = calc_A_dyn_t(l1t[0], l1t[1], l2t[0], l2t[1])
        l_dyn_t = np.array([0, 0, 0, m * g * dt, 0, 0])
        u_dyn_t = l_dyn_t

        row_indices = (idx * dim_dyn_fqp, (idx + 1) * dim_dyn_fqp)
        col_indices = (idx * dim_x_fqp, idx * dim_x_fqp + dim_x_fqp * 2)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_dyn_t
        l[row_indices[0] : row_indices[1]] = l_dyn_t
        u[row_indices[0] : row_indices[1]] = u_dyn_t

    # friction constraints
    for t in np.arange(N + 1):
        A_fric_t = calc_A_fric_t()
        l_fric_t = np.zeros(dim_fric_fqp)
        u_fric_t = np.full(dim_fric_fqp, np.inf)

        row_indices = (
            N * dim_dyn_fqp + t * dim_fric_fqp,
            N * dim_dyn_fqp + (t + 1) * dim_fric_fqp,
        )
        col_indices = (t * dim_x_fqp, (t + 1) * dim_x_fqp)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_fric_t
        l[row_indices[0] : row_indices[1]] = l_fric_t
        u[row_indices[0] : row_indices[1]] = u_fric_t

    # kinematic constraints
    for t in np.arange(N + 1):
        p1t = p1_cqp[:, t]
        p2t = p2_cqp[:, t]

        A_kin_t = calc_A_kin_t()
        l_kin_t = np.full(dim_kin_fqp, -np.inf)
        u_kin_t = calc_u_kin_t(p1t[0], p1t[1], p2t[0], p2t[1])

        row_indices = (
            N * dim_dyn_fqp + (N + 1) * dim_fric_fqp + t * dim_kin_fqp,
            N * dim_dyn_fqp + (N + 1) * dim_fric_fqp + (t + 1) * dim_kin_fqp,
        )
        col_indices = (t * dim_x_fqp, (t + 1) * dim_x_fqp)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_kin_t
        l[row_indices[0] : row_indices[1]] = l_kin_t
        u[row_indices[0] : row_indices[1]] = u_kin_t

    # foot in air constraints
    air_idx = 0
    for t in np.arange(N + 1):
        if c1[t] == False:
            A_air1_t = calc_A_air1_t()
            l_air1_t = 0
            u_air1_t = 0

            row_idx = (
                N * dim_dyn_fqp
                + (N + 1) * dim_fric_fqp
                + (N + 1) * dim_kin_fqp
                + air_idx
            )
            col_indices = (t * dim_x_fqp, (t + 1) * dim_x_fqp)

            A[row_idx, col_indices[0] : col_indices[1]] = A_air1_t
            l[row_idx] = l_air1_t
            u[row_idx] = u_air1_t

            air_idx += 1

        if c2[t] == False:
            A_air2_t = calc_A_air2_t()
            l_air2_t = 0
            u_air2_t = 0

            row_idx = (
                N * dim_dyn_fqp
                + (N + 1) * dim_fric_fqp
                + (N + 1) * dim_kin_fqp
                + air_idx
            )
            col_indices = (t * dim_x_fqp, (t + 1) * dim_x_fqp)

            A[row_idx, col_indices[0] : col_indices[1]] = A_air2_t
            l[row_idx] = l_air2_t
            u[row_idx] = u_air2_t

            air_idx += 1

    # objective
    P = calc_P()
    q = np.empty((N + 1) * dim_x_fqp)
    for t in np.arange(N + 1):
        h_prev_t = h_prev[:, t]
        h_des_t = h_des[:, t]
        q_t = calc_q_t(h_prev_t, h_des_t)

        row_indices = (t * dim_x_fqp, (t + 1) * dim_x_fqp)

        q[row_indices[0] : row_indices[1]] = q_t

    qp = osqp.OSQP()

    qp.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=l, u=u, **osqp_settings)
    results = qp.solve()

    X_sol_fqp = results.x.reshape((dim_x_fqp, N + 1), order="F")

    X_sol = np.empty((dim_x, N + 1))
    X_sol[:6, :] = X_sol_fqp[:6, :]
    X_sol[6:10, :] = X_prev[6:10, :]
    X_sol[10:, :] = X_sol_fqp[6:, :]

    info = results.info

    return X_sol, info


if __name__ == "__main__":
    X = np.empty((dim_x, N + 1))
    h_des = np.empty((6, N + 1))
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

        h_des[0:2, t] = r
        h_des[2:4, t] = l
        h_des[4, t] = th
        h_des[5, t] = k

    X_sol, _ = solve_force_qp(X, h_des)

    animate(X)
    animate(X_sol)
