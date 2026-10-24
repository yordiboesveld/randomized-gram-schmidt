"""Implementation of the Classical Gram-Schmidt (CGS)"""

import numpy as np

def cgs(A, M=None):
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
                r = Qj.T @ v
            else:
                MQj = MQ[:, :j]
                r = (v.T @ MQj).T

            R[:j, j] = r
            v -= Qj @ r

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