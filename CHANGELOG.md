# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public API is still being reviewed and may change
between minor versions.

## [Unreleased]

### Added

- **NativeArray interop**: zero-copy view constructors (`new floatN(array)`,
  `new floatMxN(rows, cols, array)`) over an existing `NativeArray`'s memory, plus
  `CopyTo`/`CopyFrom(NativeArray)` overloads; all element dtypes.
- `IsCreated` on every vector/matrix type (false for `default` and after `Dispose()`).
- `zeroInPlace()` / `fillInPlace(s)` component ops for vectors AND matrices, float through
  integer dtypes.
- **Linear programming (`LP`)**: dense simplex, a bounded-variable revised simplex (default) and
  dual simplex over an LU-factored basis, Mehrotra primal-dual interior point, and a matrix-free
  interior-point variant over a sparse (BSR) constraint matrix.
- **Least-absolute-deviation / quantile regression (`LP.lad`)**: two reformulation-free exact
  engines — Barrodale-Roberts specialized simplex and Frisch-Newton interior point — behind a
  size-routed hybrid default, plus a matrix-free sparse route. Both engines also fit an arbitrary
  quantile (`tau` overloads), not just the median.
- **Warm-started LP re-solve**: `LPBasis` persists the terminal basis across re-solves of a
  perturbed problem; an optional per-dtype cache additionally persists the basis factorization and
  pricing weights so a re-solve skips the fixed per-call rebuild/refactorization cost.
- **Quadratic programming (`QP`)**: active-set solver for equality- and inequality-constrained
  convex QPs, with an LP-powered phase 1 for finding an initial feasible point.
- **Mixed-integer programming (`MIP`)**: branch-and-bound over the warm-started dual simplex, with
  pseudocost/reliability branching, domain propagation, and a rounding heuristic.
- **Control**: discrete-time LQR (`Control.lqr`, structured doubling algorithm) with a warm
  re-solve state and gain scheduling across a sequence of operating points.
- Rank-revealing row-pivoted LQ (`LQRP`) and complete-orthogonal-decomposition minimum-norm solves
  (`minNormSolveInPlace`) on `QRCP` and `LQRP` (pseudoinverse-equivalent least squares for
  rank-deficient systems).
- Multiple-right-hand-side (`AX=B`) overloads across the direct solver family (LU, Cholesky,
  pivoted Cholesky, QR, QRCP, LQ, SVD) — factor once, solve a whole block of right-hand sides.
- Transposed LU solves (`decompSolveTransA` / `solveInPlaceTransA`).
- BSR random sparse gallery generators for large-scale sparse benchmarking.

### Changed

- LU gains a blocked level-3 `decompInPlace` path; pivoted Cholesky (`CHOP`) gains a blocked
  level-3 factorization.
- QRCP: blocked (dlaqps-style) panel factorization, and a fused destructive `solveInPlace` that
  skips reconstructing `Q`.
- `LQ.minNormSolve` gains a fused fast path (factor once, apply `Qᵀ` from the reflectors) — about
  2× faster.
- Iterative/sparse solver throughput: SIMD width-4 accumulator reductions across dense GEMV, CG,
  SVD and eigen kernels; block-Jacobi and BSR block-triangular-sweep (SSOR) preconditioners; block
  SpMM for block-operator applies; fused Krylov vector kernels — roughly 2-4× depending on kernel.
- `Eigen.lobpcg`'s block operator applies are ~2.3-2.5× faster; LOBPCG itself was folded into
  `Eigen` (previously its own class).
- API: the `Solvers` class was retired, split into `Krylov` (iterative solvers) and `Blas`
  (triangular solves); `determinant`/`logDeterminant` moved onto `Analysis`.
- Sparse debug print (`Print.Spy` on a BSR matrix): absent blocks now render as spaces instead of
  dots, for readability on larger grids.

## [0.1.0] — 2026-07-03

First public preview. The library is feature-complete for its core scope and heavily
tested; the surface is still being finalized ahead of a `1.0`.

### Core

- Arena memory model — a stable-heap `Arena` handle over an `ArenaCore`, with
  `ClearTemp` / `Dispose` lifetime and pointer-stable in-arena records for growable state.
- Typed vectors & matrices for `float`, `double`, `int`, `short`, `long`, `bool`.
- Element-wise ops, dot / matrix-multiply (register-tiled GEMM), transpose, outer product,
  select, comparisons; zero-allocation (preallocated-output / reusable-workspace) variants
  for hot loops.

### Numerical

- Decompositions: LU, Cholesky, QR & LQ (level-3 blocked, compact-WY), pivoted variants
  (QRCP), and SVD (thin, truncated GKL/Lanczos, randomized).
- Solvers: direct, least-squares (over-determined), min-norm (under-determined, CGNE/Craig),
  and iterative — CG/PCG, MINRES, BiCGSTAB, CGLS/LSQR/LSMR with Tikhonov damping and
  Jacobi / column-equilibration preconditioners. Every solver returns a diagnostics struct.
- Eigensolvers: dominant eigenpair (power iteration), symmetric (Householder + QL,
  `Eigen.symmetric`), non-symmetric eigenvalues (Francis QR), and matrix-free
  power / inverse / Lanczos (dense or sparse).
- Numerical LA: norms, condition number, determinant, trace, rank.

### Sparse

- Block-sparse (BSR) matrices with rectangular blocks and a COO builder; symmetric
  upper-block storage; block-Jacobi preconditioner; block-size-unrolled spMV kernels.
- The iterative solvers and matrix-free eigensolvers run over dense *or* sparse operands
  through a shared linear-operator interface.

### Data, ML & signal

- ML: PCA (covariance / SVD / truncated / randomized routes) and k-means (k-means++).
- Statistics: mean, var/std, median, min/max, argmin/max, row/col reductions,
  covariance, correlation.
- FFT/DFT: power-of-two FFT/IFFT and real-input rfft/irfft (radix-4 workspace) plus a
  direct DFT for arbitrary N.
- Random: distribution samplers and structured / multivariate matrix generators.
- Histograms, resampling (nearest / linear / Catmull-Rom), transforms (normalize /
  standardize / softmax / …), 1D optimizers (root-finding, minimization, gradient descent),
  find/query ops, and a matrix gallery of classic test matrices.

### Tooling

- Type-generic codegen (templates → `float`/`double`, `int`/`short`/`long`, `bool`),
  runnable headlessly via `Tools/regen.ps1`.
- Burst-safe `Print.Log` / `Print.Spy` (including block-sparse spy) and managed
  `Print.ToText` / `ToCsv` / `SaveCsv` export for every matrix type.

[0.1.0]: https://github.com/viliwonka/LinearAlgebraBursted/releases/tag/v0.1.0
