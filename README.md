# randomized-gram-schmidt

*This repository contains part of the work I carried out during my internship at Ansys France (part of Synopsys). The implementation developed here was later integrated into Ansys's eigensolvers.*

A Python testbed for comparing deterministic and randomized Gram-Schmidt
orthogonalization methods on the same input matrix. The randomized Gram-Schmidt algorithms are inspired by the paper "Randomized Orthogonalization Process With Reorthogonalization"
from Y. Jang and L. Grigori (2025). It supports both the
Euclidean inner product and the generalized inner product defined by a sparse
symmetric positive-definite metric matrix `M`.

For an input matrix `A` of shape `(n, m)`, the methods compute a factorization

$$
A \approx QR
$$

and measure reconstruction error, basis conditioning, and loss of
orthogonality. The randomized methods also maintain a sketch-space basis $S = \Theta Q$.
The driver compares runtime and numerical quality using checkpoint tables and
three diagnostic figures.

Available methods:

- **CGS** and **CGS2**: classical Gram-Schmidt, with one or two passes.
- **MGS** and **MGS2**: modified Gram-Schmidt, with one or two passes.
- **RGS**: randomized Gram-Schmidt using sketch-space least-squares projections.
- **RGS-L2C**: RGS with an exact-inner-product CGS correction.
- **RGS-L2M**: RGS with an exact-inner-product MGS correction.

The sketch choices are **SRHT**, **Gaussian**, and **sparse CountSketch**.

---

## Quick start

Run the driver directly with `python main.py` from this directory. The examples
below use Windows PowerShell; its multiline continuation character is a backtick.

```powershell
cd D:\repos\Cross-Solver-Internships\Yordi\gram_schmidt_tester

# One-time setup, if you do not already have an environment.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Small generated example: all seven methods, Euclidean inner product.
python main.py --n 1000 --m 30 --k 100 --output-dir gs_results\example

# Inspect all command-line options.
python main.py --help
```

For later sessions, activate the environment before running the driver.
Dependencies are listed in [requirements.txt](./requirements.txt): NumPy,
SciPy, Matplotlib, Numba, and threadpoolctl.

Running with no arguments uses the larger configured defaults:

```powershell
python main.py
```

This generates an input with `n=100000` and `m=100`, runs every method with
`k=1000` for the randomized methods, and writes results to `gs_results`.
Start with the smaller example above when checking a new environment.

### Supplied matrices

Use an existing NumPy input and sparse metric:

```powershell
python main.py `
    --A-file Path\A.npy `
    --M-file Path\M.npz `
    --k 1000 `
    --compactMGS `
    --output-dir gs_results\supplied
```

The file name does not determine the matrix dimensions. The driver obtains
`n` and `m` from the loaded input and checks that `M` has shape `(n, n)`.

Matrix Market files can be used for either input:

```powershell
python main.py `
    --A-file Path\A.mmf `
    --M-file Path\M.mmf `
    --k 1000 `
    --output-dir gs_results\matrix_market
```

Replace the example file names with files present on your machine. Omit
`--M-file` to use the Euclidean inner product.

---

## What the tester does

[main.py](./main.py) performs the following steps:

1. Loads the supplied input and optional metric, or generates the built-in example.
2. Selects the methods, sketch types, and sketch dimensions.
3. Warms the selected randomized sketch-application paths outside the run timers.
4. Computes input-prefix condition numbers, unless `--skip-plots` is set.
5. Runs each selected configuration and reports its elapsed time and final
   orthogonality errors.
6. Computes prefix diagnostics and sketch diagnostics where applicable.
7. Saves timing data, a metrics table, and figures.

### Matrix notation

The command-line name `--A-file` refers to the **input** matrix. Internally,
the driver calls that matrix `A`; the computed basis returned by an
orthogonalization method is called `Q`.

```text
A: input, shape (n, m)
Q: computed basis, shape (n, m)
R: factorization coefficients, shape (m, m)
S: randomized sketch-space basis, shape (k, m)
M: optional metric, shape (n, n)
```

For deterministic methods and the corrected randomized methods, the intended
orthogonality target is

```text
Q.T @ Q approximately equals I       when M is omitted
Q.T @ M @ Q approximately equals I   when M is supplied
```

Plain RGS instead normalizes in sketch space, targeting `S.T @ S` close to
identity. Its physical-space orthogonality error need not be near machine
precision, even if the sketch-space error is small.

### Built-in example

When `--A-file` is omitted, `create_test_matrix` in [main.py](./main.py)
constructs

```text
x  = linspace(0, 1, n)
mu = linspace(0, 1, m)
A[:, j] = sin(10 * (mu[j] + x)) / (cos(100 * (mu[j] - x)) + 1.1)
```

The result is a dense Fortran-order `float64` array. The example is deterministic:
`--seed` controls the randomized sketches, not this input generator.

---

## Routine reference

The orthogonalization methods live in [orth/](./orth/), and shared numerical,
sketching, diagnostic, and plotting routines live in [helpers/](./helpers/).

### Driver

| Module | Routine | Purpose |
|--------|---------|---------|
| [main.py](./main.py) | `main`, `parse_args` | Configure and run experiments; report and save results. |
| [main.py](./main.py) | `_load_A_file`, `_load_M_file`, `_load_inputs` | Read supported formats and validate dimensional compatibility. |
| [main.py](./main.py) | `create_test_matrix` | Generate the built-in dense test matrix. |
| [main.py](./main.py) | `_warmup_randomized_methods` | Warm sketch application for each selected `k` and sketch type. |

### Orthogonalization kernels

| Module | Routine / CLI name | Purpose |
|--------|--------------------|---------|
| [orth/cgs.py](./orth/cgs.py) | `cgs` | Classical Gram-Schmidt projection. |
| [orth/cgs2.py](./orth/cgs2.py) | `cgs2` | Classical Gram-Schmidt with a second projection pass. |
| [orth/mgs.py](./orth/mgs.py) | `mgs` | Modified Gram-Schmidt; sequential or compact-WY formulation. |
| [orth/mgs2.py](./orth/mgs2.py) | `mgs2` | Modified Gram-Schmidt with reorthogonalization; sequential or compact-WY formulation. |
| [orth/rgs.py](./orth/rgs.py) | `rgs` | Randomized Gram-Schmidt projection. |
| [orth/rgsl2c.py](./orth/rgsl2c.py) | `rgsl2c` | Randomized projection followed by an CGS correction and exact-inner-product normalization. |
| [orth/rgsl2m.py](./orth/rgsl2m.py) | `rgsl2m` | Randomized projection followed by an MGS correction; sequential or compact-WY formulation. |

### Utilities

| Module | Routine / class | Purpose |
|--------|-----------------|---------|
| [helpers/_randomized.py](./helpers/_randomized.py) | `_prepare_randomized` | Generate the sketch, apply it to the input, and return the sketch-generation time. |
| [helpers/generate_sketch.py](./helpers/generate_sketch.py) | `generate_sketch` | Construct an SRHT, Gaussian, or CountSketch operator, optionally with metric factorization data. |
| [helpers/SRHT.py](./helpers/SRHT.py) | `SRHT`, `Trimmed_SRHT` | Standard and trimmed Hadamard sketch implementations; the current implementation selects `SRHT`. |
| [helpers/GaussianSketch.py](./helpers/GaussianSketch.py) | `GaussianSketch` | Dense Gaussian sketch, scaled by `1 / sqrt(k)`. |
| [helpers/CountSketch.py](./helpers/CountSketch.py) | `CountSketch` | Signed hash sketch and fused sparse-metric application. |
| [helpers/minnerproduct_createsketch.py](./helpers/minnerproduct_createsketch.py) | `incomplete_cholesky_ldlt`, `_create_ThetaC` | Compute stabilized incomplete cholesky factorization for the metric and generalized embedding data. |
| [helpers/minnerproduct_createsketch.py](./helpers/minnerproduct_createsketch.py) | `_apply_ThetaC`, `_apply_theta_plain` | Apply generalized or plain sketches to vectors and matrices. |
| [helpers/minnerproduct_createsketch.py](./helpers/minnerproduct_createsketch.py) | `make_permuted_sketch_operator`, `warmup_ic0_ldlt` | Assemble the reusable metric operator and warm the compiled incomplete-factorization kernel. |
| [helpers/leastsquares_sketch.py](./helpers/leastsquares_sketch.py) | `_incremental_qr_add`, `_solve_from_qr` | Maintain sketch-space QR and solve projection least-squares systems with a triangular solve. |
| [helpers/diagnostics.py](./helpers/diagnostics.py) | `make_checkpoints`, `prefix_metrics`, `sketch_prefix_metrics` | Compute checkpoint locations, reconstruction error, conditioning, and orthogonality errors. |
| [helpers/plotting.py](./helpers/plotting.py) | `create_plots`, `create_sketch_plot` | Save condition-number, error, and sketch-orthogonality figures. |

---

## Orthogonalization setup

### Euclidean and generalized problems

All seven methods accept `M=None` for the Euclidean inner product. With a
metric, use a real symmetric positive-definite matrix of shape `(n, n)`.
The driver checks the shape, but does not perform a complete symmetry or
positive-definiteness test.

Use real, finite inputs with independent columns and normally `n >= m`.
The implementations do not provide uniform numerical-rank detection or
automatic deflation. Zero or dependent columns can cause breakdown, division
by zero, or non-finite diagnostics.

### Sketch types

| CLI value | Operator | Properties |
|-----------|----------|------------|
| `srht` | Subsampled randomized Hadamard transform | Random signs, zero-padding to the next power of two, fast Hadamard transform, and sampling without replacement. |
| `gaussian` | Dense Gaussian matrix | Entries are independent standard normal values divided by `sqrt(k)`; explicitly stores a `(k, n)` dense matrix. |
| `sparse` | CountSketch | Maps each input coordinate to one signed output bucket. |

The CLI default is `srht`.

Choose `k >= m` for a potentially full-column-rank sketch of the basis.
This is necessary but not sufficient for a well-conditioned embedding.
The driver warns, but does not stop, when `k < m`.

For SRHT, also require

```text
k <= N, where N is the smallest power of two greater than or equal to n.
```

For example, `n=798` permits `k <= 1024`. The default `k=1000` is not valid
for a smaller input whose padded dimension is below 1000.

### Generalized sketch transformation

With `M` supplied, the randomized methods apply a sketch to a metric-transformed
input rather than directly to `A`:

```text
theta_A = Theta @ C @ A
C = diag(1 / sqrt(d)) @ L.T @ P
```

Here `L`, `d`, and the permutation defining `P` come from the stabilized
IC(0)-LDL construction. Reverse Cuthill-McKee ordering is used to decrease the bandwidth to increase stability. `C_perm` folds the permutation
into the sparse operator used by the optimized sketch implementations.

This is an **incomplete** metric factorization: `C.T @ C` approximates `M`
and is not generally identical to it. Both embedding distortion and the
metric approximation can affect plain RGS orthogonality in the original
`M` inner product. RGS-L2C and RGS-L2M additionally use the original inner
product for their correction and normalization.

### Comparing all sketch types

```powershell
python main.py `
    --rgs-all-sketch-types `
    --A-file ..\data\A1000_30_wing.npy `
    --M-file ..\data\M1000_wing.npz `
    --k 100 500 1000 `
    --output-dir gs_results\rgs_all_sketches
```

This runs **only plain RGS**, once for each sketch-type/`k` combination.
It overrides `--methods` and `--sketch-type`; it does not run RGS-L2C or
RGS-L2M. Labels include the sketch type to keep results separate.

---

## Command-line API

All options are defined in [main.py](./main.py). Boolean options use
`store_true`: supply the flag to enable it, without writing `True`.

| Option | Default | Meaning |
|--------|---------|---------|
| `-h`, `--help` | - | Show usage and exit. |
| `--A-file PATH` | Omitted | Input matrix. Supports `.npy`, `.mmf`, `.mtx`, and `.mm`. Omit to generate the example. |
| `--M-file PATH` | Omitted (`M=None`) | Optional metric. Supports `.npz`, `.mmf`, `.mtx`, and `.mm`. |
| `--n N` | `100000` | Row count for the generated example; ignored when an input file is supplied. |
| `--m M` | `100` | Column count for the generated example; ignored when an input file is supplied. |
| `--k K [K ...]` | `1000` | One or more positive sketch dimensions. Each randomized method runs once per value. Non-randomized methods run once. |
| `--sketch-type TYPE` | `srht` | `srht`, `gaussian`, or `sparse`. Used by the randomized methods in normal mode. |
| `--rgs-all-sketch-types` | Off | Run only RGS with all three sketch types and all requested `k` values. |
| `--methods METHOD [METHOD ...]` | All seven methods | Select any of `cgs`, `cgs2`, `mgs`, `mgs2`, `rgs`, `rgsl2c`, `rgsl2m`. Overridden by all-sketch mode. |
| `--seed SEED` | `42` | Seed used when constructing each randomized sketch. |
| `--threads THREADS` | `16` | BLAS thread limit around each measured method call. Does not set the Numba worker count or limit every diagnostic operation. |
| `--checkpoint-step STEP` | `20` | Positive spacing between prefix diagnostics. The final column count is always included. |
| `--output-dir PATH` | `gs_results` | Directory for timing data, metrics, and figures. Created if needed. |
| `--skip-plots` | Off | Skip input condition numbers, prefix diagnostics, the metrics table, and all figures. Final orthogonality diagnostics and timing output still run. |
| `--show` | Off | Display figures in addition to saving them; may block until plot windows are closed. Has no effect with `--skip-plots`. |
| `--compactMGS` | Off | Enable compact-WY MGS, MGS2, and RGS-L2M formulations. |

### Selecting methods and sketch dimensions

```powershell
python main.py `
    --n 1000 --m 30 `
    --methods mgs2 rgs rgsl2c rgsl2m `
    --sketch-type gaussian `
    --k 100 200 `
    --compactMGS `
    --checkpoint-step 10 `
    --output-dir gs_results\gaussian_comparison
```

This runs MGS2 once and each randomized method twice. Checkpoints are
`10`, `20`, and `30`.

---

## Orthogonalization API

For Python scripts placed next to [main.py](./main.py), the package initializer
[orth/__init__.py](./orth/__init__.py) exposes the seven functions:

```python
from orth import cgs, cgs2, mgs, mgs2, rgs, rgsl2c, rgsl2m
```

### Deterministic methods

```python
Q, R = cgs(A, M=None)
Q, R = cgs2(A, M=None)
Q, R = mgs(A, M=None, compact=False)
Q, R = mgs2(A, M=None, compact=True)
```

### Randomized methods

```python
Q, R, S, generate_time = rgs(
    A, M=None, k=1000, seed=42, sketch_type="srht"
)

Q, R, S, generate_time = rgsl2c(
    A, M=None, k=1000, seed=42, sketch_type="srht"
)

Q, R, S, generate_time = rgsl2m(
    A, M=None, k=1000, seed=42, compact=False, sketch_type="srht"
)
```

`k` must be supplied. `generate_time` is measured in seconds around the
internal `generate_sketch` call. The timing does not include creating the incomplete Cholesky factorization for the metric. The randomized methods return **four**
values, not just `(Q, R, S)`.

### Sketch helper interfaces

```python
from helpers.generate_sketch import generate_sketch
from helpers.minnerproduct_createsketch import (
    _apply_theta_plain,
    _apply_ThetaC,
    make_permuted_sketch_operator,
)
```

For the Euclidean path:

```python
theta = generate_sketch(n, k, sketch_type="srht", rng=rng)
theta_A = _apply_theta_plain(A, {"Theta": theta})
```

The state key is exactly `"Theta"`; `_apply_theta_plain` expects a dictionary,
not the raw operator.

For the generalized path:

```python
theta, inverse_sqrt_d, LT, permutation = generate_sketch(
    n, k, M=M, sketch_type="srht", rng=rng
)
C_perm = make_permuted_sketch_operator(inverse_sqrt_d, LT, permutation)
theta_A = _apply_ThetaC(
    A, theta, inverse_sqrt_d, LT, permutation, C_perm=C_perm
)
```

Here `rng` is a NumPy generator, for example `np.random.default_rng(42)`.
Supplying `C_perm` selects the optimized generalized application path.

---

## Timing and warmup

### Reported runtime

For each randomized run, the driver reports the sketch-operator construction
time separately and subtracts it from the elapsed method-call time:

$$
t_{\mathrm{reported}} = t_{\mathrm{elapsed}} - t_{\mathrm{sketch}}.
$$

The sketch timer measures construction of the selected SRHT, Gaussian, or
CountSketch operator inside `generate_sketch`. It does **not** include the
incomplete factorization of $M$.

The reported method time therefore includes the remaining work performed by
the randomized method, including:

- incomplete factorization and preprocessing of $M$, when supplied;
- construction of $C_{\mathrm{perm}}$, when applicable;
- application of the sketch to the input matrix;
- Gaussian preparation of $\Theta C_{\mathrm{perm}}$, when applicable;
- work-array allocation, sketch-space QR updates, projection solves, and
  physical-space updates.

Thus, $t_{\mathrm{reported}}$ is not solely the orthogonalization-loop time.
The outer timer also includes printing the sketch-generation time.

For non-randomized methods, the reported time is the elapsed method-call time;
there is no sketch-generation time to subtract.

The sketch-generation time is printed for randomized runs but is not saved in
`timings.txt` or included in the final timing summary. Those contain only
$t_{\mathrm{reported}}$.

### Warmup coverage

Some sketch operations and the incomplete metric factorization use Numba-compiled kernels. The driver warms the selected sketch-application path before timing, and also warms the IC(0)-LDL kernel when a metric matrix is supplied. This helps keep first-call compilation overhead out of the measured runtimes. If any randomized method is selected, the driver warms each selected sketch
type and each requested `k`:

- Without `M`: applies the plain sketch to a zero `(n, 2)` matrix.
- With `M`: warms IC(0)-LDL on a synthetic tridiagonal metric, then constructs
  and applies each generalized sketch using the actual supplied `M`.

The zero matrix is safe here because these calls only apply sketches; they
do not normalize basis columns. Warmup results are discarded. Each measured
run constructs its own sketch and metric-factorization data again.

---

## Metrics and plots

For checkpoints `i`, define `A_i = A[:, :i]`, `Q_i = Q[:, :i]`, and
`G_i = Q_i.T @ M @ Q_i`, with `M=I` in Euclidean mode.

| Metric key | Definition | Interpretation |
|------------|------------|----------------|
| `input_condition` | `cond(A_i)` | Euclidean 2-norm condition number of the input prefix. |
| `cond_q` | `sqrt(lambda_max(G_i) / lambda_min(G_i))` | Condition number of the basis in the selected inner product; Euclidean when `M=I`. |
| `residual` | `norm(A_i - Q_i @ R[:i, :i], "fro") / norm(A_i, "fro")` | Relative Euclidean reconstruction error, also when a metric is supplied. The implementation evaluates the triangular reconstruction column by column. |
| `orthogonality2` | `norm(I - G_i, 2)` | Spectral-norm loss of orthogonality. |
| `orthogonalityF` | `norm(I - G_i, "fro")` | Frobenius-norm loss of orthogonality. |

For **plain RGS**, additional sketch metrics are computed from
`S_i = S[:, :i]`: spectral and Frobenius norms of `I - S_i.T @ S_i`.
The driver does not collect these sketch-prefix metrics for RGS-L2C or
RGS-L2M.

Final physical-space orthogonality errors are printed even with
`--skip-plots`. Plain RGS also prints its final sketch-space errors.
The prefix metrics and input condition numbers are only computed when
`--skip-plots` is absent.

### Output files

| File | Contents |
|------|----------|
| `timings.txt` | One reported elapsed time per configuration, in seconds, excluding its measured `generate_sketch` time. Always written. |
| `metrics.txt` | Whitespace-separated checkpoint table with a header: checkpoint, input condition number, method metrics, and plain-RGS sketch metrics. Written unless `--skip-plots`. |
| `figure1_condition_numbers.png` | Input-prefix and computed-basis condition numbers. |
| `figure2_errors.png` | Reconstruction error and spectral/Frobenius physical-space orthogonality errors. |
| `figure3_sketch_orthogonality.png` | Plain-RGS sketch-space spectral/Frobenius orthogonality errors. |

All three figures are saved at 220 dpi when plotting is enabled. Figures are closed after saving unless `--show` is set.
The driver does not save the computed `Q`, `R`, or `S` arrays.

Metrics column names derive from run labels by replacing spaces with
underscores and removing `=`. For example:

```text
RGS k=1000 sketch=srht
    -> RGS_k1000_sketchsrht_orthogonality2
    -> RGS_k1000_sketchsrht_sketch_orthogonality2
```

Use a unique output directory for each experiment. Existing files with the
same names are overwritten, and `--skip-plots` does not remove old metrics
or figures left by an earlier run.
