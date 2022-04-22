import numpy as np
import scipy.sparse as sp
import osqp

N = 5
dt = 0.01
m = 1.5
g = 9.81
I = 0.02
Lmax = 0.38

dim_x = 10


def calc_A_dyn_t(l1tx, l1ty, l2tx, l2ty):
    A_dyn_t = sp.lil_matrix((6, dim_x * 2))
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
    A_fric_t = sp.lil_matrix((6, dim_x))
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


if __name__ == "__main__":
    A_dyn_t = calc_A_dyn_t(1, 2, 3, 4)
    l_dyn_t = np.array([0, 0, 0, m * g * dt, 0, 0])
    u_dyn_t = l_dyn_t

    A_fric_t = calc_A_fric_t()
    l_fric_t = np.zeros(6)
    u_fric_t = np.full(6, np.inf)

    print(A_fric_t.toarray())

    import ipdb

    ipdb.set_trace()
