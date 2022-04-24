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
    X, h_des = generate_reference()

    consensus_arr = []

    X_prev = X
    for iter in range(maxiter):
        X = solve_force_qp(X, h_des)
        X = solve_contact_qp(X)

        consensus = calc_consensus(X, X_prev)
        print(consensus)
        consensus_arr.append(consensus)
        if consensus < eps_f:
            break

        X_prev = X
    consensus_arr = np.array(consensus_arr)

    plt.plot(consensus_arr, "-o")
    plt.yscale("log")
    plt.xlabel("BCD Iterations", size=20)
    plt.ylabel("Consensus Parameter", size=20)
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=20)
    plt.show()

    animate(X)
