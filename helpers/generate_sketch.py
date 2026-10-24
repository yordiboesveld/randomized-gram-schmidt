"""Random sketching matrices. Port of ``generate_sketch.m``.

Returns either a dense ndarray (gaussian), a SciPy sparse matrix (CountSketch),
or a dict describing an SRHT operator.
"""

import numpy as np
import time

from .minnerproduct_createsketch import _create_ThetaC
from .SRHT import SRHT, Trimmed_SRHT
from .CountSketch import CountSketch
from .GaussianSketch import GaussianSketch


def generate_sketch(n, d, M=None, sketch_type="srht", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if sketch_type is None:
        sketch_type = "gaussian"

    kind = sketch_type.lower()

    t0 = time.perf_counter()
    if kind == "gaussian":
        Theta = GaussianSketch(n, d, rng)

    elif kind == "srht":
        srht_seed = int(rng.integers(0, 2**32 - 1))
        Theta = SRHT(n, d, seed=srht_seed)

    elif kind == "sparse":
        Theta = CountSketch(n, d, rng)
    else:
        raise ValueError(f"Unknown sketch type: {sketch_type}")
    generate_time = time.perf_counter() - t0
    
    if M is not None:
        inverse_sqrt_d, LT, permutation = _create_ThetaC(M)
        return Theta, inverse_sqrt_d, LT, permutation, generate_time
    
    return Theta, generate_time
