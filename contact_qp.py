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


if __name__ == "__main__":
    A_dyn_t = calc_A_dyn_t(1, 2, 3, 4)
    import ipdb

    ipdb.set_trace()
