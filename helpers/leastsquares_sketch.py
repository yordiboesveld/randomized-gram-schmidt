import numpy as np
from scipy.linalg import solve_triangular

def _incremental_qr_add(S, U, T, j, reorthogonalize=True):
    u = S[:, j].copy()

    if j:
        Uj = U[:, :j]
        coeff = (u.conj() @ Uj).conj()
        u -= Uj @ coeff
        T[:j, j] = coeff

        if reorthogonalize:
            correction = (u.conj() @ Uj).conj()
            u -= Uj @ correction
            T[:j, j] += correction

    T[j, j] = np.linalg.norm(u)
    U[:, j] = u / T[j, j]


def _solve_from_qr(
    U: np.ndarray, T: np.ndarray, p: np.ndarray, j: int
) -> np.ndarray:
    """Solve min ||S_j r-p|| using the maintained QR S_j=U_j T_j."""
    if j == 0:
        return np.empty(0, dtype=U.dtype)
    y = (p.conj() @ U[:, :j]).conj()
    return solve_triangular(
        T[:j, :j], y, lower=False, check_finite=False, overwrite_b=False
    )