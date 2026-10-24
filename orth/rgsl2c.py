"""Implementation of RGS with an L2-CGS correction (RGS-L2C)."""

import numpy as np
from helpers.leastsquares_sketch import (
    _incremental_qr_add,
    _solve_from_qr,
)
from helpers._randomized import _prepare_randomized


def rgsl2c(A, M=None, k=None, seed=42, sketch_type="srht"):
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

    _, theta_A, generate_time = _prepare_randomized(
        A, M, k, seed, sketch_type
    )

    for j in range(m):
        p = theta_A[:, j]
        r1 = _solve_from_qr(U, T, p, j)
        np.copyto(v := np.empty(n, dtype=np.float64), A[:, j])

        if j:
            Qj = Q[:, :j]
            v -= Qj @ r1
            s = p - S[:, :j] @ r1

            if M is None:
                correction = Qj.T @ v
            else:
                MQj = MQ[:, :j]
                correction = (v.T @ MQj).T

            v -= Qj @ correction
            s -= S[:, :j] @ correction
            R[:j, j] = r1 + correction
        else:
            s = p.copy()

        if M is None:
            Mv = v
        else:
            Mv = M @ v

        R[j, j] = np.sqrt(v.T @ Mv)
        Q[:, j] = v / R[j, j]

        if MQ is not None:
            MQ[:, j] = Mv / R[j, j]

        S[:, j] = s / R[j, j]
        _incremental_qr_add(S, U, T, j, reorthogonalize=False)

    return Q, R, S, generate_time
