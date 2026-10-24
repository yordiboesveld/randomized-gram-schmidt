"""Implementation of RGS with an L2-MGS correction (RGS-L2M)."""

import numpy as np
from scipy.linalg.blas import daxpy, ddot

from helpers.leastsquares_sketch import (
    _incremental_qr_add,
    _solve_from_qr,
)
from helpers._randomized import _prepare_randomized


def rgsl2m(A, M=None, k=None, seed=42, compact=False, sketch_type="srht"):
    if k is None:
        raise ValueError("k must be provided")
    A = np.asarray(A, dtype=np.float64, order="F")
    n, m = A.shape
    Q = np.empty(A.shape, dtype=np.float64, order="F")
    R = np.zeros((m, m), dtype=np.float64, order="F")
    MQ = np.empty(A.shape, dtype=np.float64, order="F") if M is not None else None
    S = np.empty((k, m), dtype=np.float64, order="F")
    U = np.zeros((k, m), dtype=np.float64, order="F")
    T = np.zeros((m, m), dtype=np.float64, order="F")
    _, theta_A, generate_time = _prepare_randomized(A, M, k, seed, sketch_type)
    if compact:
        Tmgs = np.zeros((m, m), dtype=np.float64, order="F")
        for j in range(m):
            p = theta_A[:, j]
            r1 = _solve_from_qr(U, T, p, j)
            v = A[:, j].copy()
            if j > 0:
                R[:j, j] = r1
                v -= Q[:, :j] @ r1
                s = p - S[:, :j] @ r1
                if M is None:
                    corr = Tmgs[:j, :j].T @ (Q[:, :j].T @ v)
                else:
                    corr = Tmgs[:j, :j].T @ (MQ[:, :j].T @ v)
                R[:j, j] += corr
                v -= Q[:, :j] @ corr
                s -= S[:, :j] @ corr
            else:
                s = p.copy()
            mq = v if M is None else M @ v
            rjj = np.sqrt(v @ mq)
            R[j, j] = rjj
            q = v / rjj
            Q[:, j] = q
            if MQ is not None:
                MQ[:, j] = mq / rjj
            S[:, j] = s / rjj
            _incremental_qr_add(S, U, T, j, reorthogonalize=False)
            Tmgs[j, j] = 1.0
            if j > 0:
                if M is None:
                    Tmgs[:j, j] = Tmgs[:j, :j] @ -(Q[:, :j].T @ q)
                else:
                    Tmgs[:j, j] = Tmgs[:j, :j] @ -(MQ[:, :j].T @ q)
    else:
        v = np.empty(n, dtype=np.float64)

        for j in range(m):
            p = theta_A[:, j]
            r1 = _solve_from_qr(U, T, p, j)
            np.copyto(v, A[:, j])

            if j:
                R[:j, j] = r1
                v -= Q[:, :j] @ r1
                s = p - S[:, :j] @ r1

                for i in range(j):
                    if M is None:
                        correction = ddot(Q[:, i], v)
                    else:
                        correction = ddot(v, MQ[:, i])

                    R[i, j] += correction
                    v = daxpy(Q[:, i], v, a=-correction)
                    s -= correction * S[:, i]
            else:
                s = p.copy()

            Mv = v if M is None else M @ v
            R[j, j] = np.sqrt(v.T @ Mv)
            Q[:, j] = v / R[j, j]

            if MQ is not None:
                MQ[:, j] = Mv / R[j, j]

            S[:, j] = s / R[j, j]
            _incremental_qr_add(S, U, T, j, reorthogonalize=False)
    return Q, R, S, generate_time
