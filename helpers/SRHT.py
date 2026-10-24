"""
SRHT.py

An implementation of the trimmed Fast Walsh-Hadamard Transform (FWHT) and the standard FWHT for the SRHT sketch matrix. 

Trimmed FWHT computes only selected outputs efficiently from the sampling matrix P, avoiding unnecessary computations
for unselected entries. This results in nlog(k) complexity, where k is the number of selected outputs. The standard FWHT
computes all outputs with nlog(n) complexity.

However, trimmed FWHT is generally less GPU-friendly than the standard FWHT because the amount of active work decreases 
irregularly at each transform level. This produces an uneven workload and makes it more difficult to maintain high GPU occupancy.
For large k, most branches remain active for many levels, so relatively little computation is eliminated. At the same time,
the trimmed implementation still incurs additional overhead from branch checks, indirect indexing, execution-plan lookups,
and less regular loops. Consequently, the standard FWHT can outperform the trimmed version despite performing more arithmetic.
"""

import numpy as np
from numba import njit, prange


# -----------------------------------------------------------------------------
# Trimmed FWHT SRHT Class
# -----------------------------------------------------------------------------


@njit(cache=True)
def build_trimmed_fwht_plan(
    transform_size: int,
    indices: np.ndarray,
):
    """
    Build a reusable, non-recursive execution plan for selected FWHT outputs.
    """
    indices = np.asarray(indices, dtype=np.int64)

    if transform_size <= 0 or (transform_size & (transform_size - 1)) != 0:
        raise ValueError("transform_size must be a positive power of two")

    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")

    if np.any(indices < 0) or np.any(indices >= transform_size):
        raise ValueError("An index is outside the transform range")

    # Each node contains:
    #   positions: locations in the final output
    #   local_indices: requested indices relative to the current node
    nodes = [
        (
            np.arange(indices.size, dtype=np.int64),
            indices.copy(),
        )
    ]

    level_sum_children = []
    level_diff_children = []
    node_counts = []
    node_sizes = []

    node_size = transform_size

    while node_size > 1:
        half = node_size // 2
        number_of_nodes = len(nodes)

        sum_children = np.full(
            number_of_nodes,
            -1,
            dtype=np.int64,
        )
        diff_children = np.full(
            number_of_nodes,
            -1,
            dtype=np.int64,
        )

        next_nodes = []

        for node_id, (positions, local_indices) in enumerate(nodes):
            sum_mask = local_indices < half
            diff_mask = ~sum_mask

            if np.any(sum_mask):
                sum_children[node_id] = len(next_nodes)

                next_nodes.append(
                    (
                        positions[sum_mask],
                        local_indices[sum_mask],
                    )
                )

            if np.any(diff_mask):
                diff_children[node_id] = len(next_nodes)

                next_nodes.append(
                    (
                        positions[diff_mask],
                        local_indices[diff_mask] - half,
                    )
                )

        level_sum_children.append(sum_children)
        level_diff_children.append(diff_children)
        node_counts.append(number_of_nodes)
        node_sizes.append(node_size)

        nodes = next_nodes
        node_size = half

    output_leaves = np.empty(indices.size, dtype=np.int64)

    for leaf_id, (positions, _) in enumerate(nodes):
        for position in positions:
            output_leaves[position] = leaf_id

    number_of_levels = len(node_sizes)
    maximum_node_count = 0

    for count in node_counts:
        maximum_node_count = max(maximum_node_count, count)

    sum_children = np.full(
        (number_of_levels, maximum_node_count),
        -1,
        dtype=np.int64,
    )
    diff_children = np.full(
        (number_of_levels, maximum_node_count),
        -1,
        dtype=np.int64,
    )

    for level in range(number_of_levels):
        count = node_counts[level]

        sum_children[level, :count] = level_sum_children[level]
        diff_children[level, :count] = level_diff_children[level]

    return (
        sum_children,
        diff_children,
        np.asarray(node_counts, dtype=np.int64),
        np.asarray(node_sizes, dtype=np.int64),
        output_leaves,
    )


@njit(cache=True)
def trimmed_fwht_iterative(
    work: np.ndarray,
    sum_children: np.ndarray,
    diff_children: np.ndarray,
    node_counts: np.ndarray,
    node_sizes: np.ndarray,
    output_leaves: np.ndarray,
) -> np.ndarray:
    """
    Compute selected entries of the unnormalized FWHT without recursion.
    The execution plan is reused and no index tree is constructed here.
    """
    padded_size = work.size
    destination = np.empty(padded_size, dtype=np.float64)

    source = work
    current_node_size = padded_size

    for level in range(node_sizes.size):
        number_of_nodes = node_counts[level]
        half = current_node_size // 2

        for node in range(number_of_nodes):
            source_offset = node * current_node_size

            sum_node = sum_children[level, node]
            diff_node = diff_children[level, node]

            if sum_node >= 0 and diff_node >= 0:
                sum_offset = sum_node * half
                diff_offset = diff_node * half

                for i in range(half):
                    a = source[source_offset + i]
                    b = source[source_offset + half + i]

                    destination[sum_offset + i] = a + b
                    destination[diff_offset + i] = a - b

            elif sum_node >= 0:
                sum_offset = sum_node * half

                for i in range(half):
                    destination[sum_offset + i] = (
                        source[source_offset + i]
                        + source[source_offset + half + i]
                    )

            elif diff_node >= 0:
                diff_offset = diff_node * half

                for i in range(half):
                    destination[diff_offset + i] = (
                        source[source_offset + i]
                        - source[source_offset + half + i]
                    )

        temporary = source
        source = destination
        destination = temporary

        current_node_size = half

    result = np.empty(output_leaves.size, dtype=np.float64)

    for output_position in range(output_leaves.size):
        result[output_position] = source[
            output_leaves[output_position]
        ]

    return result


#Adding a trimmed FWHT for complex numbers, since njit specializes in only one dtype.
@njit(cache=True)
def trimmed_fwht_iterative_complex(
    work,
    sum_children,
    diff_children,
    node_counts,
    node_sizes,
    output_leaves,
):
    padded_size = work.size
    destination = np.empty(padded_size, dtype=np.complex128)

    source = work
    current_node_size = padded_size

    for level in range(node_sizes.size):
        number_of_nodes = node_counts[level]
        half = current_node_size // 2

        for node in range(number_of_nodes):
            source_offset = node * current_node_size
            sum_node = sum_children[level, node]
            diff_node = diff_children[level, node]

            if sum_node >= 0 and diff_node >= 0:
                sum_offset = sum_node * half
                diff_offset = diff_node * half

                for i in range(half):
                    a = source[source_offset + i]
                    b = source[source_offset + half + i]
                    destination[sum_offset + i] = a + b
                    destination[diff_offset + i] = a - b

            elif sum_node >= 0:
                sum_offset = sum_node * half

                for i in range(half):
                    destination[sum_offset + i] = (
                        source[source_offset + i]
                        + source[source_offset + half + i]
                    )

            elif diff_node >= 0:
                diff_offset = diff_node * half

                for i in range(half):
                    destination[diff_offset + i] = (
                        source[source_offset + i]
                        - source[source_offset + half + i]
                    )

        source, destination = destination, source
        current_node_size = half

    result = np.empty(output_leaves.size, dtype=np.complex128)

    for position in range(output_leaves.size):
        result[position] = source[output_leaves[position]]

    return result


@njit(cache=True)
def apply_srht_vector_trimmed(
    x: np.ndarray,
    signs: np.ndarray,
    padded_size: int,
    scale: float,
    sum_children: np.ndarray,
    diff_children: np.ndarray,
    node_counts: np.ndarray,
    node_sizes: np.ndarray,
    output_leaves: np.ndarray,
) -> np.ndarray:
    work = np.zeros(padded_size, dtype=np.float64)

    for i in range(x.size):
        work[i] = signs[i] * x[i]

    transformed = trimmed_fwht_iterative(
        work,
        sum_children,
        diff_children,
        node_counts,
        node_sizes,
        output_leaves,
    )

    for row in range(transformed.size):
        transformed[row] *= scale

    return transformed


class Trimmed_SRHT:
    """SRHT: Theta(x) = k^(-1/2) R H D x.

    H is unnormalized. Inputs are zero-padded to the next power of two.
    Random signs and sampled indices are fixed for the operator lifetime.
    """

    def __init__(self, n: int, k: int, seed: int = 42) -> None:
        self.n = int(n)
        self.k = int(k)

        if self.n <= 0 or self.k <= 0:
            raise ValueError("n and k must be positive")

        self.N = 1 << (self.n - 1).bit_length()

        if self.k > self.N:
            raise ValueError(
                f"k={self.k} exceeds padded dimension N={self.N}"
            )

        rng = np.random.default_rng(seed)

        self.signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=self.n,
        )

        self.indices = rng.choice(
            self.N,
            size=self.k, replace=False,
        ).astype(np.int64)

        self.scale = 1.0 / np.sqrt(self.k)
        self.work = np.empty(self.N, dtype=np.float64)

        (
            self.sum_children,
            self.diff_children,
            self.node_counts,
            self.node_sizes,
            self.output_leaves,
        ) = build_trimmed_fwht_plan(
            self.N,
            self.indices,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if x.ndim != 1 or x.size != self.n:
            raise ValueError(f"Expected a vector of shape ({self.n},)")

        if np.iscomplexobj(x):
            return apply_srht_vector_trimmed_complex(
                np.asarray(x, dtype=np.complex128),
                self.signs,
                self.N,
                self.scale,
                self.sum_children,
                self.diff_children,
                self.node_counts,
                self.node_sizes,
                self.output_leaves,
            )

        return apply_srht_vector_trimmed(
            np.asarray(x, dtype=np.float64),
            self.signs,
            self.N,
            self.scale,
            self.sum_children,
            self.diff_children,
            self.node_counts,
            self.node_sizes,
            self.output_leaves,
        )
    
    def apply_columns_C(self, X, C_perm):
        if np.iscomplexobj(X):
            kernel, dtype = fused_trimmed_srht_C_complex, np.complex128
        else:
            kernel, dtype = fused_trimmed_srht_C, np.float64
        return kernel(
            C_perm.indptr, C_perm.indices, C_perm.data,
            self.signs, self.N, self.scale,
            self.sum_children, self.diff_children,
            self.node_counts, self.node_sizes, self.output_leaves,
            np.asarray(X, dtype=dtype),
        )

    def apply_columns(self, A):
        A = np.asarray(A)

        if A.ndim != 2 or A.shape[0] != self.n:
            raise ValueError(f"Expected a matrix with {self.n} rows")

        if np.iscomplexobj(A):
            result = apply_srht_columns_parallel_trimmed_complex(
                np.asarray(A, dtype=np.complex128),
                self.signs,
                self.N,
                self.scale,
                self.sum_children,
                self.diff_children,
                self.node_counts,
                self.node_sizes,
                self.output_leaves,
            )
        else:
            result = apply_srht_columns_parallel_trimmed(
                np.asarray(A, dtype=np.float64),
                self.signs,
                self.N,
                self.scale,
                self.sum_children,
                self.diff_children,
                self.node_counts,
                self.node_sizes,
                self.output_leaves,
            )

        return np.asfortranarray(result)

#Adding a trimmed FWHT for complex numbers, since njit specializes in only one dtype.
@njit(cache=True)
def apply_srht_vector_trimmed_complex(
    x,
    signs,
    padded_size,
    scale,
    sum_children,
    diff_children,
    node_counts,
    node_sizes,
    output_leaves,
):
    work = np.zeros(padded_size, dtype=np.complex128)

    for i in range(x.size):
        work[i] = signs[i] * x[i]

    transformed = trimmed_fwht_iterative_complex(
        work,
        sum_children,
        diff_children,
        node_counts,
        node_sizes,
        output_leaves,
    )

    for row in range(transformed.size):
        transformed[row] *= scale

    return transformed


@njit(parallel=True, cache=True)
def apply_srht_columns_parallel_trimmed(
    A: np.ndarray,
    signs: np.ndarray,
    padded_size: int,
    scale: float,
    sum_children: np.ndarray,
    diff_children: np.ndarray,
    node_counts: np.ndarray,
    node_sizes: np.ndarray,
    output_leaves: np.ndarray,
) -> np.ndarray:
    """
    Apply the non-recursive trimmed SRHT to every column of A.

    Parameters
    ----------
    A:
        Input matrix with shape (n, m).

    signs:
        Random signs with shape (n,).

    padded_size:
        Power-of-two padded transform size.

    scale:
        SRHT scaling factor, normally 1 / sqrt(k).

    sum_children:
        Precomputed destination indices for sum branches.

    diff_children:
        Precomputed destination indices for difference branches.

    node_counts:
        Number of active nodes at every transform level.

    node_sizes:
        Size of each active node at every transform level.

    output_leaves:
        Leaf location corresponding to every requested output index.

    Returns
    -------
    result:
        Sketched matrix with shape (k, m).
    """
    n, m = A.shape
    k = output_leaves.size

    result = np.empty((k, m), dtype=np.float64)

    for j in prange(m):
        # Private padded vector for column j.
        work = np.zeros(padded_size, dtype=np.float64)

        # Apply random signs and copy the column into the padded vector.
        for i in range(n):
            work[i] = signs[i] * A[i, j]

        # Compute only the selected Hadamard coefficients.
        transformed = trimmed_fwht_iterative(
            work,
            sum_children,
            diff_children,
            node_counts,
            node_sizes,
            output_leaves,
        )

        # Scale and store the selected coefficients.
        for row in range(k):
            result[row, j] = scale * transformed[row]

    return result


@njit(parallel=True, cache=True)
def apply_srht_columns_parallel_trimmed_complex(
    A,
    signs,
    padded_size,
    scale,
    sum_children,
    diff_children,
    node_counts,
    node_sizes,
    output_leaves,
):
    n, m = A.shape
    k = output_leaves.size

    result = np.empty((k, m), dtype=np.complex128)

    for j in prange(m):
        work = np.zeros(padded_size, dtype=np.complex128)

        for i in range(n):
            work[i] = signs[i] * A[i, j]

        transformed = trimmed_fwht_iterative_complex(
            work,
            sum_children,
            diff_children,
            node_counts,
            node_sizes,
            output_leaves,
        )

        for row in range(k):
            result[row, j] = scale * transformed[row]

    return result

@njit(parallel=True, cache=True)
def fused_trimmed_srht_C(indptr, indices, data, signs, padded_size, scale,
                         sum_children, diff_children, node_counts, node_sizes,
                         output_leaves, X):
    n, m = X.shape
    k = output_leaves.size
    out = np.empty((m, k), dtype=np.float64)
    for j in prange(m):
        work = np.zeros(padded_size, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += data[p] * X[indices[p], j]
            work[i] = signs[i] * acc
        transformed = trimmed_fwht_iterative(
            work, sum_children, diff_children, node_counts, node_sizes, output_leaves
        )
        for r in range(k):
            out[j, r] = scale * transformed[r]
    return out.T


@njit(parallel=True, cache=True)
def fused_trimmed_srht_C_complex(indptr, indices, data, signs, padded_size, scale,
                                 sum_children, diff_children, node_counts, node_sizes,
                                 output_leaves, X):
    n, m = X.shape
    k = output_leaves.size
    out = np.empty((m, k), dtype=np.complex128)
    for j in prange(m):
        work = np.zeros(padded_size, dtype=np.complex128)
        for i in range(n):
            acc = 0.0 + 0.0j
            for p in range(indptr[i], indptr[i + 1]):
                acc += data[p] * X[indices[p], j]
            work[i] = signs[i] * acc
        transformed = trimmed_fwht_iterative_complex(
            work, sum_children, diff_children, node_counts, node_sizes, output_leaves
        )
        for r in range(k):
            out[j, r] = scale * transformed[r]
    return out.T




# -----------------------------------------------------------------------------
# Regular fwht SRHT Class
# -----------------------------------------------------------------------------


#Compiles the fwht to increase performance
@njit(cache=True)
def fwht_inplace(x: np.ndarray) -> None:
    """Apply an unnormalized Fast Walsh-Hadamard transform in place."""
    n = x.size
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            for offset in range(h):
                left = start + offset
                right = left + h
                a = x[left]
                b = x[right]
                x[left] = a + b
                x[right] = a - b
        h *= 2

# Compiles the SRHT vector application for complex vectors to increase performance
@njit(cache=True)
def apply_srht_vector_complex(
    x,
    signs,
    indices,
    padded_size,
    scale,
):
    work = np.zeros(padded_size, dtype=np.complex128)

    for i in range(x.size):
        work[i] = signs[i] * x[i]

    fwht_inplace(work)

    result = np.empty(indices.size, dtype=np.complex128)

    for row in range(indices.size):
        result[row] = scale * work[indices[row]]

    return result

class SRHT:
    """SRHT: Theta(x) = k^(-1/2) R H D x.

    H is unnormalized. Inputs are zero-padded to the next power of two.
    Random signs and sampled indices are fixed for the operator lifetime.
    """

    def __init__(self, n: int, k: int, seed: int = 42) -> None:
        self.n = int(n)
        self.k = int(k)
        if self.n <= 0 or self.k <= 0:
            raise ValueError("n and k must be positive")
        self.N = 1 << (self.n - 1).bit_length()
        if self.k > self.N:
            raise ValueError(f"k={self.k} exceeds padded dimension N={self.N}")

        rng = np.random.default_rng(seed)
        self.signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64), size=self.n
        )
        self.indices = rng.choice(self.N, size=self.k, replace=False)
        self.scale = 1.0 / np.sqrt(self.k)
        self.work = np.empty(self.N, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 1 or x.size != self.n:
            raise ValueError(f"Expected a vector of shape ({self.n},)")

        if np.iscomplexobj(x):
            return apply_srht_vector_complex(
                np.asarray(x, dtype=np.complex128),
                self.signs,
                self.indices,
                self.N,
                self.scale,
            )

        self.work.fill(0.0)
        np.multiply(self.signs, x, out=self.work[: self.n])
        fwht_inplace(self.work)
        result = self.work[self.indices].copy()
        result *= self.scale
        return result

    def apply_columns(self, A: np.ndarray) -> np.ndarray:
        """
        Applies SRHT to the matrix A at once.
        """
        if A.ndim != 2 or A.shape[0] != self.n:
            raise ValueError(f"Expected a matrix with {self.n} rows")

        if np.iscomplexobj(A):
            result = apply_srht_columns_parallel_complex(
                np.asarray(A, dtype=np.complex128),
                self.signs,
                self.indices,
                self.N,
                self.scale,
            )
        else:
            result = apply_srht_columns_parallel(
                np.asarray(A, dtype=np.float64),
                self.signs,
                self.indices,
                self.N,
                self.scale,
            )

        return np.asfortranarray(result)

    def apply_columns_C(self, X, C_perm):
        return fused_srht_C(
            C_perm.indptr, C_perm.indices, C_perm.data,
            self.signs, self.indices, self.N, self.scale, X,
        )

@njit(parallel=True, cache=True)
def apply_srht_columns_parallel(
    A: np.ndarray,
    signs: np.ndarray,
    indices: np.ndarray,
    padded_size: int,
    scale: float,
) -> np.ndarray:
    """
    Applies SHRT to the matrix A parallel using prange.
    """
    n, m = A.shape
    k = indices.size

    result = np.empty((k, m), dtype=np.float64)

    for j in prange(m):
        # Private work array for this prange iteration.
        work = np.zeros(padded_size, dtype=np.float64)

        for i in range(n):
            work[i] = signs[i] * A[i, j]

        fwht_inplace(work)

        for row in range(k):
            result[row, j] = scale * work[indices[row]]

    return result


@njit(parallel=True, cache=True)
def apply_srht_columns_parallel_complex(
    A,
    signs,
    indices,
    padded_size,
    scale,
):
    n, m = A.shape
    k = indices.size

    result = np.empty((k, m), dtype=np.complex128)

    for j in prange(m):
        work = np.zeros(padded_size, dtype=np.complex128)

        for i in range(n):
            work[i] = signs[i] * A[i, j]

        fwht_inplace(work)

        for row in range(k):
            result[row, j] = scale * work[indices[row]]

    return result

@njit(parallel=True, cache=True)
def fused_srht_C(indptr, indices, data, signs, sel, padded_size, scale, X):
    n, m = X.shape
    k = sel.size
    out = np.empty((m, k), dtype=X.dtype)
    for j in prange(m):
        work = np.zeros(padded_size, dtype=X.dtype)
        for i in range(n):
            acc = X[i, j] * 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += data[p] * X[indices[p], j]
            work[i] = signs[i] * acc
        fwht_inplace(work)
        for r in range(k):
            out[j, r] = scale * work[sel[r]]
    return out.T