"""Diagnostics utilities for evaluating Gram-Schmidt orthogonalization results."""

import numpy as np
from scipy import sparse
from typing import Sequence, Dict
from numpy import dot as ddot

def make_checkpoints(m: int, step: int) -> np.ndarray:
    points = list(range(step, m + 1, step))
    if not points or points[-1] != m:
        points.append(m)
    return np.asarray(points, dtype=int)


def relative_prefix_residual_decomp(
    A: np.ndarray, Q: np.ndarray, R: np.ndarray, width: int
) -> float:
    residual_sq = 0.0
    input_sq = 0.0
    for j in range(width):
        difference = A[:, j] - Q[:, : j + 1] @ R[: j + 1, j]
        residual_sq += ddot(difference, difference)
        input_sq += ddot(A[:, j], A[:, j])
    return float(np.sqrt(residual_sq / input_sq))

def relative_prefix_residual_ortho(
    Q: np.ndarray, MQ: np.ndarray, width: int
) -> float:
    residual_sq = 0.0

    Qi = Q[:, :width]
    MQi = MQ[:, :width]

    for j in range(width):
        difference = (
            np.eye(width)[:, j]
            - Qi.T @ MQi[:, j]
        )
        residual_sq += ddot(difference, difference)

    return float(np.sqrt(residual_sq))

def prefix_metrics(
    A: np.ndarray, M: sparse.spmatrix | None, Q: np.ndarray, R: np.ndarray, checkpoints: Sequence[int]
) -> Dict[str, np.ndarray]:
    cond_q, residual, orthogonality2, orthogonalityF = [], [], [], []
    MQ = Q if M is None else np.asarray(M @ Q, dtype=np.float64, order="F")
    for width in checkpoints:
        Qi = Q[:, :width]
        gram = Qi.T @ MQ[:, :width]
        eigenvalues = np.linalg.eigvalsh(gram)
        cond_q.append(float(np.sqrt(eigenvalues[-1] / eigenvalues[0])))
        residual.append(relative_prefix_residual_decomp(A, Q, R, width))
        orthogonality2.append(
            float(np.linalg.norm(np.eye(width) - gram, ord=2))
        )
        orthogonalityF.append(relative_prefix_residual_ortho(Q, MQ, width))
    return {
        "cond_q": np.asarray(cond_q),
        "residual": np.asarray(residual),
        "orthogonality2": np.asarray(orthogonality2),
        "orthogonalityF": np.asarray(orthogonalityF)
    }

def sketch_prefix_metrics(
    S: np.ndarray,
    checkpoints: Sequence[int],
) -> Dict[str, np.ndarray]:
    orthogonality2 = []
    orthogonalityF = []

    for width in checkpoints:
        Si = S[:, :width]
        gram = Si.T @ Si
        identity = np.eye(width)

        orthogonality2.append(
            float(np.linalg.norm(identity - gram, ord=2))
        )
        orthogonalityF.append(
            float(np.linalg.norm(identity - gram, ord="fro"))
        )

    return {
        "orthogonality2": np.asarray(orthogonality2),
        "orthogonalityF": np.asarray(orthogonalityF),
    }