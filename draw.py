import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from constants import *

plt.style.use("seaborn")


def rot_mat_2d(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def draw(r, th, p1, p2, f1, f2):
    # plot body
    body_tr = r + rot_mat_2d(th) @ [body_l / 2, body_h / 2]
    body_tl = r + rot_mat_2d(th) @ [-body_l / 2, body_h / 2]
    body_bl = r + rot_mat_2d(th) @ [-body_l / 2, -body_h / 2]
    body_br = r + rot_mat_2d(th) @ [body_l / 2, -body_h / 2]
    body_coords = np.vstack((body_tr, body_tl, body_bl, body_br, body_tr)).T
    plt.plot(body_coords[0, :], body_coords[1, :], "-o", markersize=7)

    # plot feet
    plt.plot(p1[0], p1[1], marker="o", color="g", markersize=7)
    plt.plot(p2[0], p2[1], marker="o", color="g", markersize=7)

    # plot forces
    f_len = 0.02
    f1_vec = p1 + f_len * f1
    f2_vec = p2 + f_len * f2
    f1_coords = np.vstack((p1, f1_vec)).T
    f2_coords = np.vstack((p2, f2_vec)).T
    plt.plot(f1_coords[0, :], f1_coords[1, :], color="r")
    plt.plot(f2_coords[0, :], f2_coords[1, :], color="r")


def init_fig():
    anim_fig = plt.figure(figsize=(6, 6))
    ax = anim_fig.add_subplot()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.25, 0.75])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return anim_fig, ax


if __name__ == "__main__":
    r = np.array([0, 0.15])
    th = 0.0
    p1 = np.array([-0.15, 0])
    p2 = np.array([0.15, 0])
    f1 = np.array([0, m * g / 2])
    f2 = np.array([0, m * g / 2])

    anim_fig, ax = init_fig()

    draw(r, th, p1, p2, f1, f2)
    plt.show()
