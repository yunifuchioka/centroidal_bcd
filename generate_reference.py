import numpy as np
from scipy.interpolate import interp1d

from constants import *
from draw import animate


def generate_reference(motion_type="default"):
    if motion_type == "random":
        num_points = N // 50
        t_des = np.linspace(0, N * dt, num_points)
        rx_des = np.random.rand(num_points) * 0.14 - 0.07
        ry_des = np.random.rand(num_points) * 0.14 + 0.07
        r_des = np.vstack((rx_des, ry_des))
        th_des = np.random.rand(num_points) * np.pi / 4.0 - np.pi / 8.0

        interp_kind = "linear"
        r_interp_func = interp1d(t_des, r_des, kind=interp_kind)
        th_interp_func = interp1d(t_des, th_des, kind=interp_kind)

    X = np.empty((dim_x, N + 1))
    h_des = np.empty((6, N + 1))
    for t in np.arange(N + 1):
        if motion_type == "default":
            r = np.array([0.1 * np.cos(t / 10), 0.1 + 0.05 * np.sin(t / 10)])
            l = np.array([0.0, 0.0])
            th = np.pi / 8 * np.sin(t / 14)
            k = 0.0
            p1 = np.array([-0.15, 0])
            p2 = np.array([0.15, 0])
            f1 = np.array([0, m * g / 2])
            f2 = np.array([0, m * g / 2])
        if motion_type == "random":
            r = r_interp_func(t * dt)
            l = np.array([0.0, 0.0])
            th = th_interp_func(t * dt)
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

    return X, h_des


if __name__ == "__main__":
    X, _ = generate_reference(motion_type="random_linear")

    animate(X)
