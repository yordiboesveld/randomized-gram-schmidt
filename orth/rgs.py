"""Implementation of Randomized Gram-Schmidt (RGS)."""

import numpy as np
from helpers.leastsquares_sketch import (
    _incremental_qr_add,
    _solve_from_qr,
)
from helpers._randomized import _prepare_randomized


def rgs(A, M=None, k=None, seed=42, sketch_type="srht"):
    if k is None:
        raise ValueError("k must be provided")
    A = np.asarray(A, dtype=np.float64, order="F")
    _, m = A.shape
    Q = np.empty(A.shape, dtype=np.float64, order="F")
    R = np.zeros((m, m), dtype=np.float64, order="F")
    v = np.empty(A.shape[0], dtype=np.float64)
    S = np.empty((k, m), dtype=np.float64, order="F")
    U = np.zeros((k, m), dtype=np.float64, order="F")
    T = np.zeros((m, m), dtype=np.float64, order="F")
    _, theta_A, generate_time = _prepare_randomized(A, M, k, seed, sketch_type)
    for j in range(m):
        p = theta_A[:, j]
        r = _solve_from_qr(U, T, p, j)
        if j > 0:
            R[:j, j] = r
        np.copyto(v, A[:, j])
        if j > 0:
            v -= Q[:, :j] @ r
            s = p - S[:, :j] @ r
        else:
            s = p.copy()
        R[j, j] = np.linalg.norm(s)
        Q[:, j] = v / R[j, j]
        S[:, j] = s / R[j, j]
        _incremental_qr_add(S, U, T, j, reorthogonalize=False)
    return Q, R, S, generate_time
