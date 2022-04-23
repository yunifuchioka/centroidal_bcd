import numpy as np

from constants import *


def extract_state(X, t):
    r = X[0:2, t]
    l = X[2:4, t]
    th = X[4, t]
    k = X[5, t]
    p1 = X[6:8, t]
    p2 = X[8:10, t]
    f1 = X[10:12, t]
    f2 = X[12:14, t]

    return r, l, th, k, p1, p2, f1, f2
