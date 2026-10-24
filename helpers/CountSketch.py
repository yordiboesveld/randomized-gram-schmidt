"CountSketch implementation"

from numba import njit, prange
import numpy as np


class CountSketch:
    def __init__(self, n, d, rng):
        self.n, self.d = n, d
        self.hash_idx = rng.integers(0, d, size=n).astype(np.int64)
        self.hash_sgn = (2 * rng.integers(0, 2, size=n) - 1).astype(np.float64)

    def apply_columns(self, X):
        return countsketch_columns(
            X, self.hash_idx, self.hash_sgn, self.d
        )

    def apply_columns_C(self, X, C_perm):
        return fused_sparse_C(
            C_perm.indptr, C_perm.indices, C_perm.data,
            self.hash_idx, self.hash_sgn, X, self.d,
        )

    def __call__(self, x):
        return self.apply_columns(np.asarray(x)[:, None])[:, 0]

@njit(parallel=True, cache=True)
def countsketch_columns(X, rows, signs, d):
    n, m = X.shape
    out = np.zeros((m, d), dtype=X.dtype)
    for j in prange(m):
        for i in range(n):
            out[j, rows[i]] += signs[i] * X[i, j]
    return out.T

@njit(parallel=True, cache=True)
def fused_sparse_C(indptr, indices, data, rows, signs, X, d):
    n, m = X.shape
    out = np.zeros((m, d), dtype=X.dtype)
    for j in prange(m):
        for i in range(n):
            acc = X[i, j] * 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += data[p] * X[indices[p], j]
            out[j, rows[i]] += signs[i] * acc
    return out.T