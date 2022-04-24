import numpy as np

from constants import *
from draw import animate
from force_qp import solve_force_qp
from contact_qp import solve_contact_qp

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

    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)
    X = solve_force_qp(X, h_des)
    X = solve_contact_qp(X)

    animate(X)
