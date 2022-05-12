import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicHermiteSpline

from constants import *
from draw import animate


# linearly interpolate x and y, evaluate at t
def linear_interp_t(x, y, t):
    f = interp1d(x, y, "linear")
    return f(t)


# interpolate x and y, evaluate at t using cubic splines with zero deriatives
# this creates an interpolation similar to linear interpolation, but with
# smoothed corners
def cubic_interp_t(x, y, t):
    f = CubicHermiteSpline(x, y, np.zeros_like(x))
    return f(t)


# sinusoidal function evaluated at t defined using oscillation period, minimum
# and maximum values
def sinusoid(period, min_val, max_val, t, phase_offset=0):
    return (max_val - min_val) / 2.0 * (
        1 - np.cos(2 * np.pi / period * t + phase_offset)
    ) + min_val


def generate_reference(motion_type="default"):
    if motion_type == "random":
        interp_points = N // 50
        t_interp = np.linspace(0, N * dt, interp_points)
        rx_interp = np.random.rand(interp_points) * 0.46 - 0.23
        ry_interp = np.random.rand(interp_points) * 0.14 + 0.07
        th_interp = np.random.rand(interp_points) * np.pi / 4.0 - np.pi / 8.0

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
            r = np.array(
                [
                    linear_interp_t(t_interp, rx_interp, t * dt),
                    linear_interp_t(t_interp, ry_interp, t * dt),
                ]
            )
            l = np.array([0.0, 0.0])
            th = linear_interp_t(t_interp, th_interp, t * dt)
            k = 0.0
            p1 = np.array([r[0] - 0.15, max(sinusoid(0.3, -0.07, 0.07, t * dt), 0.0)])
            p2 = np.array([r[0] + 0.15, max(sinusoid(0.3, -0.07, 0.07, t * dt), 0.0)])
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
    X, _ = generate_reference(motion_type="random")

    animate(X)
