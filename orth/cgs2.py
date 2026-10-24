"""Implementation of the Classical Gram-Schmidt with reorthogonalization (CGS2)"""

import numpy as np

def cgs2(A, M=None):
    A = np.asarray(A, order="F")
    n, m = A.shape
    dt = A.dtype

    Q = np.empty(A.shape, dtype=dt, order="F")
    R = np.zeros((m, m), dtype=dt)
    MQ = np.empty(A.shape, dtype=dt, order="F") if M is not None else None

    for j in range(m):
        v = A[:, j].astype(dt, copy=True)

        if j:
            Qj = Q[:, :j]

            if M is None:
                r1 = Qj.T @ v
            else:
                MQj = MQ[:, :j]
                r1 = (v.T @ MQj).T

            v -= Qj @ r1

            if M is None:
                r2 = Qj.T @ v
            else:
                r2 = (v.T @ MQj).T

            v -= Qj @ r2
            R[:j, j] = r1 + r2

        if M is None:
            norm_squared = v.T @ v
        else:
            Mv = M @ v
            norm_squared = v.T @ Mv

        R[j, j] = np.sqrt(norm_squared)
        Q[:, j] = v / R[j, j]

        if M is not None:
            MQ[:, j] = Mv / R[j, j]

    return Q, R