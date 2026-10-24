"""Shared sketch preparation for randomized orthogonalization methods."""

import numpy as np
import time

from .generate_sketch import generate_sketch
from .minnerproduct_createsketch import (
    _apply_ThetaC,
    _apply_theta_plain,
    make_permuted_sketch_operator,
)


def _prepare_randomized(A, M, k, seed, sketch_type):
    A = np.asarray(A, dtype=np.float64, order="F")
    rng = np.random.default_rng(seed)

    if M is None:
        theta, generate_time = generate_sketch(
            A.shape[0],
            k,
            sketch_type=sketch_type,
            rng=rng,
        )
        theta_A = _apply_theta_plain(
            A,
            {"Theta": theta},
        )
        return theta, theta_A, generate_time
    theta, inverse_sqrt_d, LT, permutation, generate_time = generate_sketch(
        A.shape[0],
        k,
        M=M,
        sketch_type=sketch_type,
        rng=rng,
    )
    C_perm = make_permuted_sketch_operator(
        inverse_sqrt_d,
        LT,
        permutation,
    )
    theta_A = _apply_ThetaC(
        A,
        theta,
        inverse_sqrt_d,
        LT,
        permutation,
        C_perm=C_perm,
    )
    return theta, theta_A, generate_time
