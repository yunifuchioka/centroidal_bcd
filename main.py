import numpy as np
import scipy.sparse as sp
import osqp

N = 5
dt = 0.01
m = 1.5
g = 9.81
I = 0.02
Lmax = 0.38

phi_r = 100.0
phi_l = 10.0
phi_th = 100.0
phi_k = 1.0
L_r = 100.0
L_l = 10.0
L_th = 100.0
L_k = 1.0
psi = 0.1

dim_x = 10
dim_dyn = 6
dim_fric = 6
dim_kin = 8


def calc_A_dyn_t(l1tx, l1ty, l2tx, l2ty):
    A_dyn_t = sp.lil_matrix((dim_dyn, dim_x * 2))
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
    A_fric_t = sp.lil_matrix((dim_fric, dim_x))
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
    A_kin_t = sp.lil_matrix((dim_kin, dim_x))
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


def calc_P():
    P = sp.lil_matrix(((N + 1) * dim_x, (N + 1) * dim_x))
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


def calc_q_t(h_des_t, h_prev_t):
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


if __name__ == "__main__":
    A = sp.lil_matrix((20 * N + 14, dim_x * (N + 1)))
    l = np.empty(20 * N + 14)
    u = np.empty(20 * N + 14)

    # dynamics constraints
    for idx in np.arange(N):
        A_dyn_t = calc_A_dyn_t(0.1, 0.1, 0.1, 0.1)  # todo: better values for l
        l_dyn_t = np.array([0, 0, 0, m * g * dt, 0, 0])
        u_dyn_t = l_dyn_t

        row_indices = (idx * dim_dyn, (idx + 1) * dim_dyn)
        col_indices = (idx * dim_x, idx * dim_x + dim_x * 2)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_dyn_t
        l[row_indices[0] : row_indices[1]] = l_dyn_t
        u[row_indices[0] : row_indices[1]] = u_dyn_t

    # friction constraints
    for t in np.arange(N + 1):
        A_fric_t = calc_A_fric_t()
        l_fric_t = np.zeros(dim_fric)
        u_fric_t = np.full(dim_fric, np.inf)

        row_indices = (N * dim_dyn + t * dim_fric, N * dim_dyn + (t + 1) * dim_fric)
        col_indices = (t * dim_x, (t + 1) * dim_x)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_fric_t
        l[row_indices[0] : row_indices[1]] = l_fric_t
        u[row_indices[0] : row_indices[1]] = u_fric_t

    # kinematic constraints
    for t in np.arange(N + 1):
        A_kin_t = calc_A_kin_t()
        l_kin_t = np.full(dim_kin, -np.inf)
        u_kin_t = calc_u_kin_t(0.1, 0.1, 0.1, 0.1)  # todo: better values for p

        row_indices = (
            N * dim_dyn + (N + 1) * dim_fric + t * dim_kin,
            N * dim_dyn + (N + 1) * dim_fric + (t + 1) * dim_kin,
        )
        col_indices = (t * dim_x, (t + 1) * dim_x)

        A[row_indices[0] : row_indices[1], col_indices[0] : col_indices[1]] = A_kin_t
        l[row_indices[0] : row_indices[1]] = l_kin_t
        u[row_indices[0] : row_indices[1]] = u_kin_t

    # objective
    P = calc_P()
    q = np.empty((N + 1) * dim_x)
    for t in np.arange(N + 1):
        h_des_t = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        h_prev_t = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        q_t = calc_q_t(h_des_t, h_prev_t)

        row_indices = (t * dim_x, (t + 1) * dim_x)

        q[row_indices[0] : row_indices[1]] = q_t

    m = osqp.OSQP()
    settings = {}

    m.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=l, u=u, **settings)
    results = m.solve()
