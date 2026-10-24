"""Implementation of Modified Gram-Schmidt with reorthogonalization (MGS2)."""

import numpy as np
from scipy.linalg.blas import daxpy, ddot

def mgs2(A, M=None, compact=True):
    if compact:
        A = np.asarray(A, order="F")
        n, m = A.shape
        dt = A.dtype
        Q = np.empty((n, m), dtype=dt, order="F")
        MQ = np.empty((n, m), dtype=dt, order="F") if M is not None else None
        R = np.zeros((m, m), dtype=dt, order="F")
        T = np.zeros((m, m), dtype=dt, order="F")
        for j in range(m):
            a = A[:, j].copy()
            if j > 0:
                Qj = Q[:, :j]
                if M is None:
                    r1 = T[:j, :j].T @ (Qj.T @ a)
                else:
                    MQj = MQ[:, :j]
                    r1 = T[:j, :j].T @ (MQj.T @ a)
                a -= Qj @ r1
                if M is None:
                    r2 = T[:j, :j].T @ (Qj.T @ a)
                else:
                    r2 = T[:j, :j].T @ (MQj.T @ a)
                a -= Qj @ r2
                R[:j, j] = r1 + r2
            mq = a if M is None else M @ a
            tmp = a @ mq
            if tmp <= 0:
                raise RuntimeError(f"Breakdown at column {j}")
            rjj = np.sqrt(tmp)
            R[j, j] = rjj
            q = a / rjj
            Q[:, j] = q
            if MQ is not None:
                MQ[:, j] = mq / rjj
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

            for i in range(j):
                if M is None:
                    correction = ddot(Q[:, i], v)
                else:
                    correction = ddot(v, MQ[:, i])

                R[i, j] += correction
                v = daxpy(Q[:, i], v, a=-correction)

            if M is None:
                Mv = v
            else:
                Mv = M @ v

            R[j, j] = np.sqrt(v.T @ Mv)
            Q[:, j] = v / R[j, j]

            if MQ is not None:
                MQ[:, j] = Mv / R[j, j]
    return Q, R
