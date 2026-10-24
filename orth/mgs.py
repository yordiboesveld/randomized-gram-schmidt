"""Implementation of Modified Gram-Schmidt (MGS)"""

import numpy as np
from scipy.linalg.blas import daxpy, ddot

def mgs(A, M=None, compact=False):
    if compact:
        A = np.asarray(A, order="F")
        n, m = A.shape
        dt = A.dtype
        Q = np.empty((n, m), dtype=dt, order="F")
        MQ = np.empty((n, m), dtype=dt, order="F") if M is not None else None
        R = np.zeros((m, m), dtype=dt, order="F")
        v = np.empty(n, dtype=dt)
        T = np.zeros((m, m), dtype=dt, order="F")
        for j in range(m):
            a = A[:, j].copy()
            if j > 0:
                if M is None:
                    r = T[:j, :j].T @ (Q[:, :j].T @ a)
                else:
                    r = T[:j, :j].T @ (MQ[:, :j].T @ a)
                a -= Q[:, :j] @ r
                R[:j, j] = r
            if M is None:
                mq = a
            else:
                mq = M @ a
            tmp = a @ mq
            rjj = np.sign(tmp) * np.sqrt(abs(tmp))
            if rjj == 0:
                raise RuntimeError(f"Breakdown at column {j}")
            R[j, j] = rjj
            q = a / rjj
            Q[:, j] = q
            if MQ is not None:
                mq /= rjj
                MQ[:, j] = mq
            T[j, j] = 1.0
            if j > 0:
                if M is None:
                    T[:j, j] = T[:j, :j] @ -(Q[:, :j].T @ q)
                else:
                    T[:j, j] = T[:j, :j] @ -(MQ[:, :j].T @ q)
    else:
        A = np.asarray(A, order="F")
        n, m = A.shape
        dt = A.dtype

        Q = np.empty((n, m), dtype=dt, order="F")
        R = np.zeros((m, m), dtype=dt, order="F")
        MQ = np.empty((n, m), dtype=dt, order="F") if M is not None else None
        v = np.empty(n, dtype=dt)

        for j in range(m):
            np.copyto(v, A[:, j])

            for i in range(j):
                if M is None:
                    rij = ddot(Q[:, i], v)
                else:
                    rij = ddot(v, MQ[:, i])

                R[i, j] = rij
                v = daxpy(Q[:, i], v, a=-rij)

            if M is None:
                Mv = v
            else:
                Mv = M @ v

            R[j, j] = np.sqrt(v.T @ Mv)
            Q[:, j] = v / R[j, j]

            if MQ is not None:
                MQ[:, j] = Mv / R[j, j]
    return Q, R