"""Run Gram-Schmidt experiments on supplied matrices or the built-in example."""

import argparse
import gc
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.io import mmread
from threadpoolctl import threadpool_limits

from helpers.diagnostics import (
    make_checkpoints,
    prefix_metrics,
    sketch_prefix_metrics,
)
from helpers.generate_sketch import generate_sketch
from helpers.minnerproduct_createsketch import (
    _apply_ThetaC,
    _apply_theta_plain,
    make_permuted_sketch_operator,
    warmup_ic0_ldlt,
)
from orth import cgs, cgs2, mgs, mgs2, rgs, rgsl2c, rgsl2m
from helpers.plotting import create_plots, create_sketch_plot


METHODS: Dict[str, Tuple[str, Callable]] = {
    "cgs": ("CGS", cgs),
    "cgs2": ("CGS2", cgs2),
    "mgs": ("MGS", mgs),
    "mgs2": ("MGS2", mgs2),
    "rgs": ("RGS", rgs),
    "rgsl2c": ("RGS-L2C", rgsl2c),
    "rgsl2m": ("RGS-L2M", rgsl2m),
}
SKETCH_TYPES = ("srht", "gaussian", "sparse")


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--A-file",
        dest="A_file",
        type=Path,
        help=("Input Q matrix in .npy, .mmf, .mtx, or .mm format; "
              "omit to generate the paper example"),
    )
    parser.add_argument(
        "--M-file",
        type=Path,
        help=(
            "Optional metric matrix in .npz, .mmf, .mtx, or .mm format"
        ),
    )
    parser.add_argument(
        "--rgs-all-sketch-types",
        action="store_true",
        help=(
            "Run only RGS for srht, gaussian, and sparse sketch types"
        ),
    )
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument(
        "--k",
        dest="ks",
        type=int,
        nargs="+",
        default=[1000],
        help="One or more sketch dimensions",
    )
    parser.add_argument(
        "--sketch-type",
        choices=SKETCH_TYPES,
        default="srht",
        help="Sketch matrix: srht, gaussian, or sparse (CountSketch)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gs_results")
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=tuple(METHODS),
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--compactMGS", action="store_true", help="Use compact-WY MGS")
    return parser.parse_args(args)

def _load_A_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        loaded = np.load(path)
    elif suffix in {".mmf", ".mtx", ".mm"}:
        loaded = mmread(path, spmatrix=True)
        if sparse.issparse(loaded):
            loaded = loaded.toarray()
    else:
        raise ValueError(
            f"Unsupported A file format '{path.suffix}'. "
            "Use .npy, .mmf, .mtx, or .mm."
        )

    A = np.asarray(
        loaded,
        dtype=np.float64,
        order="F",
    )

    if A.ndim != 2:
        raise ValueError("The input A matrix must be two-dimensional")

    return A

def _load_M_file(path: Path) -> sparse.csr_matrix:
    suffix = path.suffix.lower()

    if suffix == ".npz":
        loaded = sparse.load_npz(path)
    elif suffix in {".mmf", ".mtx", ".mm"}:
        loaded = mmread(path, spmatrix=True)
    else:
        raise ValueError(
            f"Unsupported M file format '{path.suffix}'. "
            "Use .npz, .mmf, .mtx, or .mm."
        )

    if sparse.issparse(loaded):
        M = loaded.tocsc()
    else:
        M = sparse.csc_matrix(
            np.asarray(loaded, dtype=np.float64)
        )

    return M

def _load_inputs(args):
    if args.A_file is None:
        if args.n <= 0 or args.m <= 0:
            raise ValueError("n and m must be positive")
        A = create_test_matrix(args.n, args.m)
        print(f"Generated example matrix A with shape {A.shape}")
    else:
        A = _load_A_file(args.A_file)
        print(f"Loaded input matrix from {args.A_file}")

    M = None
    if args.M_file is not None:
        M = _load_M_file(args.M_file)
        if M.shape != (A.shape[0], A.shape[0]):
            raise ValueError(
                f"M must have shape {(A.shape[0], A.shape[0])}, got {M.shape}"
            )

    return A, M

def create_test_matrix(n: int, m: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mu = np.linspace(0.0, 1.0, m, dtype=np.float64)
    A = np.empty((n, m), dtype=np.float64, order="F")
    for j, value in enumerate(mu):
        A[:, j] = np.sin(10.0 * (value + x)) / (
            np.cos(100.0 * (value - x)) + 1.1
        )
    return A

def _warmup_randomized_methods(n, M, ks, sketch_type, seed):
    A_warmup = np.zeros(
        (n, 2),
        dtype=np.float64,
        order="F",
    )

    if M is None:
        for k in ks:
            theta, _ = generate_sketch(
                n,
                k,
                sketch_type=sketch_type,
                rng=np.random.default_rng(seed),
            )

            theta_state = {"Theta": theta}

            _apply_theta_plain(
                A_warmup,
                theta_state,
            )

        return

    warmup_ic0_ldlt(n)

    for k in ks:
        theta, inverse_sqrt_d, LT, permutation, _ = generate_sketch(
            n,
            k,
            M=M,
            sketch_type=sketch_type,
            rng=np.random.default_rng(seed),
        )

        C_perm = make_permuted_sketch_operator(
            inverse_sqrt_d,
            LT,
            permutation,
        )

        _apply_ThetaC(
            A_warmup,
            theta,
            inverse_sqrt_d,
            LT,
            permutation,
            C_perm=C_perm,
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.checkpoint_step <= 0:
        raise ValueError("checkpoint-step must be positive")
    if any(k <= 0 for k in args.ks):
        raise ValueError("All sketch dimensions must be positive")

    A, M = _load_inputs(args)
    n, m = A.shape
    if args.rgs_all_sketch_types:
        selected_methods = ("rgs",)
        selected_sketch_types = SKETCH_TYPES
    else:
        selected_methods = args.methods
        selected_sketch_types = (args.sketch_type,)
    checkpoints = make_checkpoints(m, args.checkpoint_step)
    print(f"A shape: {A.shape}; memory: {A.nbytes / 1024**3:.3f} GiB")
    if M is None:
        metric_name = "I"
        print("Using the Euclidean inner product (M = I)")
    else:
        metric_name = "M"
        print(f"M shape: {M.shape}; nnz: {M.nnz}")

    if args.rgs_all_sketch_types:
        print(f"Running RGS for all sketch types: {SKETCH_TYPES}")
    else:
        print(f"Using sketch matrix type: {selected_sketch_types[0]}")

    if any(k < m for k in args.ks) and any(
        method.startswith("rgs") for method in selected_methods
    ):
        small_ks = [k for k in args.ks if k < m]
        print(
            "Warning: the following sketch dimensions are smaller than "
            f"m={m}: {small_ks}. Late sketch-space systems may be "
            "underdetermined."
        )

    if any(method.startswith("rgs") for method in selected_methods):
        for sketch_type in selected_sketch_types:
            _warmup_randomized_methods(
                n,
                M,
                args.ks,
                sketch_type,
                args.seed,
            )

    cond_a = None
    if not args.skip_plots:
        print("Computing condition numbers of A_i...")
        cond_a = np.asarray(
            [np.linalg.cond(A[:, :i]) for i in checkpoints]
        )

    results: Dict[str, Dict[str, np.ndarray]] = {}
    timings: Dict[str, float] = {}
    sketch_results: Dict[str, Dict[str, np.ndarray]] = {}
    compact = args.compactMGS

    runs = []

    for sketch_type in selected_sketch_types:
        for key in selected_methods:
            base_label, function = METHODS[key]

            if key.startswith("rgs"):
                runs.extend(
                    (
                        key,
                        base_label,
                        function,
                        k,
                        compact,
                        sketch_type,
                    )
                    for k in args.ks
                )
            else:
                runs.append(
                    (
                        key,
                        base_label,
                        function,
                        None,
                        compact,
                        sketch_type,
                    )
                )

    for key, base_label, function, k, compact, sketch_type in runs:
        if k is None:
            label = base_label
        else:
            label = f"{base_label} k={k}"

        if args.rgs_all_sketch_types:
            label = f"{label} sketch={sketch_type}"
        print(f"\nRunning optimized {label}...")
        with threadpool_limits(limits=args.threads, user_api="blas"):
            start = time.perf_counter()
            if k is None:
                if base_label.startswith("MGS"):
                    Q, R = function(A, M, compact)
                else:
                    Q, R = function(A, M)
                generate_time = 0.0
            elif base_label.endswith("L2M"):
                Q, R, S, generate_time = function(
                    A,
                    M,
                    k,
                    args.seed,
                    compact,
                    sketch_type,
                )
                print(f"{label} sketch generation time: {generate_time:.3f} seconds")
            else:
                Q, R, S, generate_time = function(
                    A,
                    M,
                    k,
                    args.seed,
                    sketch_type,
                )
                print(f"{label} sketch generation time: {generate_time:.3f} seconds")
            elapsed = time.perf_counter() - start - generate_time

        timings[label] = elapsed
        print(f"{label}: {elapsed:.3f} seconds")

        MQ = Q if M is None else M @ Q
        gram = Q.T @ MQ
        final_orth_2 = np.linalg.norm(np.eye(m) - gram, ord=2)
        final_orth_fro = np.linalg.norm(np.eye(m) - gram, ord="fro")
        print(
            f"{label} final orthogonality error "
            f"||I-Q.T {metric_name} Q||_2: {final_orth_2:.6e}"
        )
        print(
            f"{label} final orthogonality error "
            f"||I-Q.T {metric_name} Q||_F: {final_orth_fro:.6e}"
        )
        if base_label == "RGS":
            final_orth_2_sketch = np.linalg.norm(
                np.eye(m) - S.T @ S, ord=2
            )
            final_orth_fro_sketch = np.linalg.norm(
                np.eye(m) - S.T @ S, ord="fro"
            )
            print(
                f"{label} final sketch orthogonality error "
                f"||I-S.T S||_2: {final_orth_2_sketch:.6e}"
            )
            print(
                f"{label} final sketch orthogonality error "
                f"||I-S.T S||_F: {final_orth_fro_sketch:.6e}"
            )

        if not args.skip_plots:
            print(f"Computing prefix diagnostics for {label}...")
            results[label] = prefix_metrics(
                A, M, Q, R, checkpoints
            )
            if base_label == "RGS":
                sketch_results[label] = sketch_prefix_metrics(
                    S, checkpoints
                )

        del Q, R
        if k is not None:
            del S
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "timings.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        for label, elapsed in timings.items():
            handle.write(f"{label}: {elapsed:.6f} seconds\n")

    if not args.skip_plots:
        metric_columns = {
            "input_condition": np.asarray(cond_a),
            **{
                f"{label.replace(' ', '_').replace('=', '')}_{metric}":
                np.asarray(values)
                for label, metrics in results.items()
                for metric, values in metrics.items()
            },
            **{
                f"{label.replace(' ', '_').replace('=', '')}_sketch_{metric}":
                np.asarray(values)
                for label, metrics in sketch_results.items()
                for metric, values in metrics.items()
            },
        }
        table = np.column_stack(
            [np.asarray(checkpoints), *metric_columns.values()]
        )
        np.savetxt(
            args.output_dir / "metrics.txt",
            table,
            header=" ".join(["checkpoint", *metric_columns.keys()]),
            fmt=["%d"] + ["%.16e"] * len(metric_columns),
        )
        create_plots(
            checkpoints,
            cond_a,
            results,
            args.output_dir,
            args.show,
            metric_name,
        )
        create_sketch_plot(
            checkpoints, sketch_results, args.output_dir, args.show
        )

    print("\nTiming summary")
    for label, elapsed in timings.items():
        print(f"  {label}: {elapsed:.3f} seconds")
    print(f"Saved output to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
