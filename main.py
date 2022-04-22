import numpy as np
from scipy.sparse import csc_matrix
import osqp

P = csc_matrix((2, 2))
P[0, 0] = 1
P[1, 1] = 1
q = np.array([1, 1])

A = csc_matrix((2, 2))
A[0, 0] = 1
A[1, 1] = 1
l = np.array([-1, -1])
u = np.array([1, 1])

# print(P.toarray())

m = osqp.OSQP()

settings = {}

m.setup(P=P, q=q, A=A, l=l, u=u, **settings)
results = m.solve()
