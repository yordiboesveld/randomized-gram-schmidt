"""Setting up the generalized randomized embedding and incomplete Cholesky factorization."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

import numpy as np
from numba import njit
from scipy import sparse



@njit(cache=True)
def _ic0_ldlt_kernel(indptr, indices, values, shift_abs, shift_rel, pivot_floor):
    """Compute unit-lower Incomplete Cholesky(0) LDL^T values in the CSR pattern.
    """
    n = indptr.size - 1
    out = np.zeros(values.size, dtype=np.float64)
    diag_pos = np.empty(n, dtype=np.int64)
    d = np.empty(n, dtype=np.float64)

    for i in range(n):
        diag_pos[i] = indptr[i + 1] - 1

    for i in range(n):
        row_start = indptr[i]
        row_diag = diag_pos[i]

        # Strict-lower entries L_unit[i,j].
        for pos_ij in range(row_start, row_diag):
            j = indices[pos_ij]
            s = values[pos_ij]

            # Intersection of row i columns < j with row j columns < j.
            pi = row_start
            pj = indptr[j]
            end_j = diag_pos[j]
            while pi < pos_ij and pj < end_j:
                ki = indices[pi]
                kj = indices[pj]
                if ki == kj:
                    s -= out[pi] * d[ki] * out[pj]
                    pi += 1
                    pj += 1
                elif ki < kj:
                    pi += 1
                else:
                    pj += 1

            pivot_j = d[j]
            if pivot_j <= 0.0 or not np.isfinite(pivot_j):
                raise ValueError("Nonpositive or invalid pivot in IC(0)")
            out[pos_ij] = s / pivot_j

        # Diagonal pivot.
        aii = values[row_diag]
        pivot = aii
        for pos_ik in range(row_start, row_diag):
            k = indices[pos_ik]
            lik = out[pos_ik]
            pivot -= lik * lik * d[k]

        floor_i = max(pivot_floor, shift_abs + shift_rel * abs(aii))
        if not np.isfinite(pivot):
            raise ValueError("Invalid pivot in IC(0)")
        if pivot < floor_i:
            pivot = floor_i

        d[i] = pivot
        out[row_diag] = 1.0  # diagonal of L_unit

    return out, d

def _prepare_lower_ic0_pattern(
    M: sp.spmatrix,
) -> sp.csr_matrix:
    M = sp.csr_matrix(
        M,
        dtype=np.float64,
        copy=False,
    )

    if not M.has_canonical_format:
        M.sum_duplicates()
        M.eliminate_zeros()
        M.sort_indices()

    diagonal = M.diagonal()

    if diagonal.size != M.shape[0]:
        raise ValueError("Missing diagonal")

    if np.any(diagonal <= 0.0):
        raise ValueError(
            "M must have a positive diagonal"
        )

    lower = sp.tril(
        M,
        format="csr",
    )

    lower.sort_indices()
    return lower


def incomplete_cholesky_ldlt(
    M: sp.spmatrix,
    *,
    ordering: str = "rcm",
    shift_abs: float = 0.0,
    shift_rel: float = 1e-12,
    pivot_floor: float = 1e-30,
):
    """Return sparse L, diagonal vector d, and permutation p."""
    A = sp.csr_matrix(M, dtype=np.float64)
    n = A.shape[0]

    if ordering == "rcm":
        graph = A.copy()
        graph.data[:] = 1.0
        p = reverse_cuthill_mckee(graph, symmetric_mode=True).astype(np.int64)
    elif ordering == "natural":
        p = np.arange(n, dtype=np.int64)
    else:
        raise ValueError("ordering must be 'rcm' or 'natural'")

    Ap = A[p, :][:, p]
    lower = _prepare_lower_ic0_pattern(Ap)

    unit_values, d = _ic0_ldlt_kernel(
        lower.indptr,
        lower.indices,
        lower.data,
        float(shift_abs),
        float(shift_rel),
        float(pivot_floor),
    )

    # The kernel gives L D L^T, but we want L D^-1 L^T.
    # We set L_new = LD, because then L_new D^-1 L_new^T = L D L^T
    requested_values = unit_values * d[lower.indices]
    L = sp.csr_matrix(
        (requested_values, lower.indices, lower.indptr),
        shape=lower.shape, copy=False,
    )

    return L, d, p


def _create_ThetaC(
    M: sparse.spmatrix,
):
    """
    Prepare the generalized randomized embedding.
    """
    M = sparse.csr_matrix(M, dtype=np.float64)

    # This remains inside _prepare_randomized so the numerical factorization,
    # reordering, shifting, and C @ A application are included in RGS timing.
    L, d, permutation = incomplete_cholesky_ldlt(
        M,
        ordering="rcm",
        shift_abs=0.0,
        shift_rel=1e-12,
        pivot_floor=1e-30,
    )

    if np.any(~np.isfinite(d)) or np.any(d <= 0.0):
        raise np.linalg.LinAlgError(
            "Stabilized IC(0)-LDL produced a nonpositive or invalid pivot."
        )
    

    # Apply: C @ A = diag(1 / sqrt(d)) @ L.T @ P @ A
    inverse_sqrt_d = 1.0 / np.sqrt(d)
    LT = L.T.tocsr()

    return inverse_sqrt_d, LT, permutation

def _apply_ThetaC(
    v: np.ndarray,
    Omega,
    inverse_sqrt_d: np.ndarray,
    LT: sparse.spmatrix,
    permutation: np.ndarray,
    C_perm: sparse.spmatrix | None = None,
) -> np.ndarray:
    if v.ndim == 1:
        if C_perm is not None:
            v_block = np.asarray(v).reshape(-1, 1, order="F")
            result = Omega.apply_columns_C(v_block, C_perm)
            return result[:, 0]

        permuted_v = v[permutation]
        LT_v = LT @ permuted_v
        LT_v *= inverse_sqrt_d
        return Omega(LT_v)

    if C_perm is not None:
        return Omega.apply_columns_C(
            v,
            C_perm,
        )

    permuted_v = v[permutation, :]
    LT_v = LT @ permuted_v
    LT_v *= inverse_sqrt_d[:, None]

    return Omega.apply_columns(
        np.asarray(LT_v, order="F"),
    )

def _apply_theta_plain(X, state):
    theta = state["Theta"]

    if X.ndim == 1:
        return theta(X)
    
    return theta.apply_columns(X)

def warmup_ic0_ldlt(n) -> None:
    """
    Compile the Numba IC(0)-LDL kernel outside the timed RGS runs.
    """
    warmup_size = n

    warmup_M = sparse.diags(
        diagonals=(
            -0.25 * np.ones(
                warmup_size - 1,
                dtype=np.float64,
            ),
            2.0 * np.ones(
                warmup_size,
                dtype=np.float64,
            ),
            -0.25 * np.ones(
                warmup_size - 1,
                dtype=np.float64,
            ),
        ),
        offsets=(-1, 0, 1),
        shape=(warmup_size, warmup_size),
        format="csr",
    )

    incomplete_cholesky_ldlt(
        warmup_M,
        ordering="rcm",
        shift_abs=0.0,
        shift_rel=1e-12,
        pivot_floor=1e-30,
    )

def make_permuted_sketch_operator(inverse_sqrt_d, LT, permutation):
    scaled_LT = LT.multiply(inverse_sqrt_d[:, None]).tocsr()
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(permutation.size)
    return scaled_LT[:, inverse_permutation].tocsr()