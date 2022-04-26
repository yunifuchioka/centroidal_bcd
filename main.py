import numpy as np
import matplotlib.pyplot as plt

from constants import *
from draw import animate
from force_qp import solve_force_qp
from contact_qp import solve_contact_qp
from generate_reference import generate_reference

plt.style.use("seaborn")


def calc_consensus(X, X_prev):
    l = X[5, :]
    l_prev = X_prev[5, :]
    consensus = np.linalg.norm(l - l_prev) ** 2 / N
    return consensus


if __name__ == "__main__":
    X, h_des = generate_reference(motion_type="random")

    consensus_arr = []
    obj_arr = []
    time_arr = []

    X_prev = X
    for iter in range(maxiter):
        X, info_fqp = solve_force_qp(X, h_des)
        X, info_cqp = solve_contact_qp(X)

        obj_arr.append(info_fqp.obj_val)
        time_arr.append(info_fqp.run_time)
        time_arr.append(info_cqp.run_time)

        consensus = calc_consensus(X, X_prev)
        print("iteration {}, consensus = {}".format(iter, consensus))
        consensus_arr.append(consensus)
        if consensus < eps_f:
            break

        X_prev = X
    X, info = solve_force_qp(X, h_des)
    time_arr.append(info_fqp.run_time)

    consensus_arr = np.array(consensus_arr)
    obj_arr = np.array(obj_arr)
    time_arr = np.array(time_arr)

    print("\ntotal time used in OSQP: {} seconds".format(np.sum(time_arr)))

    fontsize = 15
    plt.subplot(2, 1, 1)
    plt.plot(obj_arr, "-o")
    plt.ylabel("Force QP Objective", size=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    plt.subplot(2, 1, 2)
    plt.plot(consensus_arr, "-o")
    plt.yscale("log")
    plt.xlabel("BCD Iterations", size=fontsize)
    plt.ylabel("Consensus Parameter", size=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    plt.show()

    animate(X, repeat=True)
