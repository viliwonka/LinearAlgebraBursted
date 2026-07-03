# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public API is still being reviewed and may change
between minor versions.

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
  `eigenSymmetric`), non-symmetric eigenvalues (Francis QR), and matrix-free
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
