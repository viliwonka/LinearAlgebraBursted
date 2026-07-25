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
- **Linear programming (`LP`)**: a bounded-variable revised simplex (default) and dual simplex over
  an LU-factored basis, plus a Mehrotra primal-dual interior point. Dense constraint matrices only —
  large-scale sparse LP is out of scope.
- **Least-absolute-deviation / quantile regression (`LP.lad`)**: two reformulation-free exact
  engines — Barrodale-Roberts specialized simplex and Frisch-Newton interior point — behind a
  size-routed hybrid default. Both work directly on the original design matrix, with no
  split-variable LP reformulation, and both fit an arbitrary quantile (`tau` overloads), not just
  the median. Frisch-Newton additionally accepts a sparse (BSR) design, streaming the stored blocks
  into an n×n normal solve per step, so the constraint count never enters the factorization.
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
- **Kalman filtering (`Kalman`)**: linear predict/update with a steady-state gain option, extended
  (EKF) via user Jacobian functors, and unscented (UKF, Van der Merwe scaled sigma points).
- **Model-predictive control (`MPC.solve`)**: condensed linear MPC with box/soft constraints over
  the warm-started active-set QP, with a persistent per-horizon state for per-frame re-solves.
- **Nonlinear least squares (`Optimize.nlsSolve` / `Optimize.curveFit`)**: Levenberg-Marquardt
  with Nielsen damping and optional robust losses (Huber, Cauchy, Tukey).
- Non-throwing preconditioner builds: `BlockJacobi`/`ILU0`/`IC0` gain `out PreconditionerInfo`
  overloads (status + rescuing diagonal shift + attempts); failed builds release their buffers.
- `Equals`/`GetHashCode` (buffer-handle identity) on every vector/matrix type, removing the
  CS0660/CS0661 warning pair for consumers of the `==`/`!=` element-wise operators.
- `Blas.dot(a, b, ref c, transposeA, transposeB)`: `A·Bᵀ` (and `Aᵀ·Bᵀ`) without materializing
  the transpose — `dot(A, A, …, transposeB: true)` routes through a dedicated symmetric `A·Aᵀ`
  kernel. `Blas.dotSym` gains a `dotSymT` sibling for symmetric-by-construction `A·Bᵀ` products
  (upper triangle + mirror, ~2× and exactly symmetric output).

### Changed

- **Breaking — the root namespace is now `BULA`** (was `LinearAlgebra`). Update `using LinearAlgebra;`
  → `using BULA;`, and the sub-namespaces likewise (`BULA.Sparse`, `BULA.Control`, `BULA.ML`,
  `BULA.Internal`). Assembly names (`BurstLinearAlgebra.*`), type names, folder layout, and asmdef
  references are unchanged, so this is a find-and-replace on `using` directives and any
  fully-qualified type references.
- **Breaking — short tuning-parameter names**: `maxIterations` → `maxIter`, `tolerance` → `tol`,
  `relativeTolerance` → `relTol` on every public API.
- **Breaking**: `Eigen.valuesQR` → `Eigen.valuesQRInPlace` (it destroys `A`; the suffix now says so).
- **Breaking — behavior**: the buffer×buffer `mulInPlace(a, b)` now multiplies INTO its receiver
  (`a *= b`, `b` untouched), matching `addInPlace`/`subInPlace`/`divInPlace`; it previously mutated
  its second argument. The `*` operators are unaffected.
- **Breaking**: `LQ.minNormSolve` takes `in A, in b` (it never modified them); bool `Analysis.isDiagonal`
  now tests true diagonality (off-diagonal all-false, square required) instead of the identity
  pattern, and its undocumented `compare` parameter is gone.
- Integer correctness: scalar-shifted-by-vector bit shifts compute at the element type's own width
  (`long` shifts past 31 were truncated); `LinVec` interpolates in double (`long`/large-`int` ramps
  had float-resolution interior values).
- `Analysis.cond` / `Blas.matrixL2` return NaN and `Analysis.rank` throws when the underlying SVD
  fails to converge (previously they read an unwritten buffer); `rowMean`/`colMean` throw on an
  empty axis like every other statistic.
- `Krylov.biCGStab`'s parameterless overload defaults its iteration budget to `A.M_Rows`
  (was `2*A.M_Rows`), matching the rest of the square-solver family.
- `SVD.values`/`SVD.thin` honor their `tol` parameter (deflation was previously hardcoded to
  `eps·‖A‖`); the default threshold is looser than the old hardcoded value, so default-path sweep
  counts can differ.
- `MIP` node/iteration limits return the best incumbent found by the rounding heuristic from the
  last node (previously reported `+inf` as if nothing feasible had been seen).

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
- **Breaking**: the k-means workspace no longer carries a `Ct` (transposed-centroids) buffer — the
  assignment GEMM reads centroids through the transposed-B kernel directly.
- Kalman/EKF/UKF steps drop several per-call transpose temporaries (`Hᵀ`, `K`, `(I−KH)ᵀ`) in favor
  of transposed-operand GEMM forms, and the UKF sigma-point covariance recombinations are now
  GEMMs instead of scalar rank-1 loops; results are equal up to floating-point summation order.
- `Rand.spdInPlace` output is now exactly symmetric by construction (mirrored triangle instead of
  a post-hoc averaging pass); `Rand`-generated matrices and k-means/Kalman results may differ
  bitwise from previous versions at equal seeds.
- `Blas.trans` and the symmetric-product mirror passes are cache-blocked (results unchanged —
  pure copy reordering).
- GEMM cache-blocks (packed operand panels) once the product's working set exceeds ~24 MB —
  about 1.3–1.4× at 2048×2048 — with bit-identical results to the direct route at every size.
- Blocked factorization trailing updates (Cholesky, pivoted Cholesky, LU, QR/LQ block
  reflectors) fuse four panel columns into each pass over the output row — factorizations are
  ~10–17% faster at level-3 sizes, with bit-identical results.
- **Breaking — float numeric change**: float dot products (`Blas.dot` on vectors and everything
  built on it) now reduce through 8-lane AVX accumulator chains — roughly 2× faster while
  operands are cache-resident. Float results differ from previous versions at the usual
  floating-point-summation-order level; double results are unchanged (up to the sign of an
  all-zero reduction). Results remain deterministic and identical across instruction sets
  (the non-AVX fallback uses the same summation tree).
- The `transposeB` matrix products and float dot/GEMV/norm reductions run on full-register-width
  (8-lane AVX) accumulators with lane-identical fallbacks — float `A·Bᵀ` about 1.5×, float
  symmetric `A·Aᵀ` about 1.6×, float GEMV about 1.7× at mid-to-large sizes; double unchanged.

### Removed

- **Breaking — the `Arena` allocator is gone.** Every vector/matrix/workspace/preconditioner is
  now standalone: construct with an explicit `Allocator` (`new floatN(n, Allocator.Temp)`,
  `new floatQRCache(m, n, allocator)`, `new floatIC0(in A, allocator)`, …) and `Dispose()` what
  outlives its scope; `Allocator.Temp` allocations are freed automatically at end of frame/job.
  The arena's factory conveniences moved to static classes with the same names and arguments:
  `GenerateOP` (constructors/random/kernels), `floatGallery` (test matrices), `Query`, `FFT`,
  `ConvertOP`. `Copy()`/`TempCopy()` now return `Allocator.Temp` copies.
- **Breaking — allocating operators removed**: `+ - * / %` on vectors/matrices (and bitwise
  `~ & | ^ << >>` on integer types, logical `! & | ^` on bool types). Use the in-place kernels
  (`floatComp.addInPlace(dst, src)`, `intComp.bitwiseOrInPlace(...)`, `boolComp.notInPlace(...)`).
  Comparators (`== != < <= > >=`) remain.
- **Breaking**: the no-workspace FFT overloads `FFT.fft(ref re, ref im)`, `FFT.ifft(...)`,
  `FFT.rfft(in real, ref re, ref im)`, and `FFT.irfft(...)` (the `sin`/`cos`-recurrence path).
  They were slower than building a workspace and running one transform, and not deterministic across
  architectures. Use the workspace overloads: build a `floatFFTCache(n, allocator)` once and pass it
  (`in ws`). `dft`/`idft` still cover arbitrary (non-power-of-two) N.

## [0.1.0] — 2026-07-03

First public preview. Feature-complete for its core scope and heavily tested.

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
- Solvers: direct, least-squares (over-determined), min-norm (under-determined),
  and iterative — CG, MINRES, BiCGSTAB, GMRES, LSQR/LSMR with Tikhonov damping and
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

[0.1.0]: https://github.com/viliwonka/BULA/releases/tag/v0.1.0
