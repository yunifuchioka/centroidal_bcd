import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from constants import *
from utils import extract_state

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
    plt.plot(body_coords[0, :], body_coords[1, :], "-o", color="b", markersize=7)

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

    # plot root coordinate system
    axis_len = 0.1
    axis_head_x = r + rot_mat_2d(th) @ [axis_len, 0]
    axis_coords_x = np.vstack((r, axis_head_x)).T
    axis_head_y = r + rot_mat_2d(th) @ [0, axis_len]
    axis_coords_y = np.vstack((r, axis_head_y)).T
    plt.plot(axis_coords_x[0, :], axis_coords_x[1, :], color="r")
    plt.plot(axis_coords_y[0, :], axis_coords_y[1, :], color="g")


def init_fig():
    anim_fig = plt.figure(figsize=(6, 6))
    ax = anim_fig.add_subplot()
    fontsize = 15
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.25, 0.75])
    plt.xlabel("x", size=fontsize)
    plt.ylabel("y", size=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    return anim_fig, ax


def animate(X, fname=None, display=True, repeat=False):
    anim_fig, ax = init_fig()

    def draw_frame(frame_idx):
        r, l, th, k, p1, p2, f1, f2 = extract_state(X, frame_idx)
        while ax.lines:
            ax.lines.pop()
        draw(r, th, p1, p2, f1, f2)
        plt.title("t={:.3}".format(frame_idx * dt))

    anim = animation.FuncAnimation(
        anim_fig,
        draw_frame,
        frames=N + 1,
        interval=dt * 1000.0,
        repeat=repeat,
        blit=False,
    )

    # if fname is not None:
    #     Writer = animation.writers["ffmpeg"]
    #     writer = Writer(fps=int(1 / dt), metadata=dict(artist="Me"), bitrate=1000)
    #     anim.save("videos/" + fname + ".mp4", writer=writer)

    if display:
        plt.show()


if __name__ == "__main__":
    X = np.empty((dim_x, N + 1))
    for t in np.arange(N + 1):
        r = np.array([0, 0.1 + 0.05 * np.sin(t / 10)])
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

    # anim_fig, ax = init_fig()

    # r, l, th, k, p1, p2, f1, f2 = extract_state(X, 0)
    # draw(r, th, p1, p2, f1, f2)

    animate(X)

    plt.show()
