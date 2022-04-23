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


def calc_l_loc_t(p1tx_des, p2tx_des):
    l_loc_t = np.array([p1tx_des, 0, p2tx_des, 0])
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


if __name__ == "__main__":
    A_dyn_t = calc_A_dyn_t(1, 2, 3, 4)
    l_dyn_t = calc_l_dyn_t(1, 2)
    u_dyn_t = l_dyn_t

    A_loc_t = calc_A_loc_t()
    l_loc_t = calc_l_loc_t(1, 2)
    u_loc_t = l_loc_t

    A_kin_t = calc_A_kin_t()
    l_kin_t = np.full(dim_kin_fqp, -np.inf)
    u_kin_t = np.full(dim_kin_fqp, Lmax)

    P = calc_P()
    rt_fcp = np.array([1, 2])
    lt_fcp = np.array([1, 2])
    p1t_fcp = np.array([1, 2])
    p2t_fcp = np.array([1, 2])
    q_t = calc_q_t(rt_fcp, lt_fcp, p1t_fcp, p2t_fcp)
    import ipdb

    ipdb.set_trace()
