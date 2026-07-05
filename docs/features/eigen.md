# Eigensolvers

`Eigen` — dense symmetric/non-symmetric eigensolvers plus a matrix-free trio that works over dense
*or* [sparse BSR](sparse-bsr.md) operands through the shared `IfloatLinearOperator` interface.

## Dense

- **`eigenSymmetric(ref A, ref eigenvalues, ref V, ...)`** — the default: Householder
  tridiagonalization (with orthogonal accumulation) + implicit-shift QL, all eigenpairs, descending
  order, `A` destroyed. **`eigenvaluesSymmetric(ref A, ref eigenvalues, ...)`** — values-only, skips
  the eigenvector accumulation entirely, faster still.
- `eigenDecomposition` (cyclic two-sided Jacobi) is `[Obsolete]` — Jacobi's column-oriented rotations
  resist SIMD; kept only because it has superior relative accuracy on graded spectra, and as a
  cross-check oracle for the Householder path.
- **`eigenvaluesQR(ref A, ref eigenvaluesReal, ref eigenvaluesImag, ...)`** — non-symmetric, Francis
  double-shift QR on an upper-Hessenberg reduction (elmhes+hqr). Values only; complex-conjugate pairs
  are represented as real/imag arrays (no complex type), via 2×2 Schur blocks.

## Matrix-free (sparse-capable)

Generic `<TOp> where TOp : struct, IfloatLinearOperator`, with thin dense (`floatMxN`) and
[`floatBSR`](sparse-bsr.md) forwarders — same body, not forked:

- **`powerIteration<TOp>(in A, ref v, ref w, out lambda, ...)`** — dominant eigenpair, Rayleigh
  quotient.
- **`inversePowerIteration<TOp>(in A, ..., out lambda, ...)`** — smallest eigenpair of SPD `A`, via an
  inner `Solvers.cg` solve each outer iteration (no explicit inverse formed).
- **`lanczos<TOp>(in A, ref ws, ref eigenvalues, int steps, ...)`** — twice-reorthogonalized symmetric
  Lanczos tridiagonalization + `eigenvaluesSymmetric` on the result → Ritz **values**.
  **`lanczosVectors<TOp>(...)`** — same tridiagonalization, then forms Ritz **vectors** too (not
  zero-alloc — allocates 3 Temp vectors internally via `eigenSymmetric`).
- **`LOBPCG.lobpcg<TOp[,TPre]>(in A[, in M], ref ws, int k, float tol, int maxIter)`** — blocked
  Locally Optimal Block Preconditioned Conjugate Gradient: the `k` SMALLEST eigenpairs of a symmetric
  operator, via deflation-based locking (a converged pair is frozen and projected out of the active
  subspace) and a small dense Rayleigh-Ritz sub-problem solved with `eigenSymmetric` (a 3-block
  `[X,W,P]` reduction that falls back to 2-block `[X,W]` if the 3-block Cholesky is too
  ill-conditioned). Dense (`floatMxN`), [BSR](sparse-bsr.md), and BSR+`floatBlockJacobi`
  (preconditioned) forwarders all share this one body — the same pattern as the rest of this section.
  **Results are ascending** (index 0 = smallest) — the one `Eigen`-family method that isn't
  descending, since "the k smallest" is the entire point. Zero-alloc at the O(n) scale via
  `floatLOBPCGCache` (`arena.floatLOBPCGCache(n, k)`, reusable/warm-startable across calls); the
  O(k)-scale Rayleigh-Ritz sub-solve still allocates a few small, bounded Temp vectors internally —
  the same exception `lanczosVectors` already has. Returns `LOBPCGInfo`.

This is the intended tool for sparse smallest-eigenpair problems (structural-stability / buckling /
modal-analysis use cases, the Fiedler vector) that `lanczos`/`inversePowerIteration` aren't the best
fit for; the dense small-scale case is still better served by `eigenSymmetric`.

**Generalized pencil form** — `lobpcg` also solves `A·x = λ·B·x` with B SPD: every overload has a
`+B` twin (generic operator, dense, BSR, BSR+block-Jacobi) that B-orthonormalizes the basis and
returns B-orthonormal eigenvectors, ascending. `B = I` forwarders are bit-identical to the standard
path. The buckling recipe (documented with a worked sample in the class doc): for
`K_E·φ + λ·K_G·φ = 0` put the SPD elastic stiffness `K_E` in the **B slot** and the (typically
indefinite) geometric stiffness `K_G` in the A slot; the returned ascending `μ[0]` (most negative)
gives the smallest positive critical load as `λ_cr = −1/μ[0]`.

## Diagnostics structs

Eigensolvers follow the same by-value, implicit-`bool` diagnostics convention as
[`Solvers`](solvers.md), with their own structs (all reuse `IterativeSolveStatus` — no dedicated
eigensolver enum):

| Struct | Fields | Used by |
|---|---|---|
| `EigenSolveInfo` | `iterations`, `residual` (double, `‖Av-λv‖`), `status` | `powerIteration`, `inversePowerIteration` |
| `LanczosInfo` | `produced` (≤ `steps`, less only on early breakdown), `status` | `lanczos`, `lanczosVectors` |
| `LOBPCGInfo` | `iterations`, `converged` (0..k pairs locked), `maxResidual` (double, worst-case relative residual over all k pairs), `status` | `LOBPCG.lobpcg` |

## Benchmarks

Single-thread, this machine, float, `Burst IJob.Run` median of 9 — Householder tridiagonalization vs.
cyclic Jacobi, same result (commits `4902032`, `facbbff`, `4a188a5`):

| Method | N | Jacobi | Householder | Speedup |
|---|---|---|---|---|
| values only (`eigenvaluesSymmetric`) | 256 | 213.2ms | 2.85ms | 74.9× |
| values only | 128 | 25.7ms | 0.473ms | 54.3× |
| values only | 64 | 3.15ms | 0.0924ms | 34× |
| values + vectors (`eigenSymmetric`), as shipped | 256 | 213.5ms | 35.8ms | 6.0× |
| values + vectors, after a follow-up vectorization of the eigenvector accumulation | 256 | 213.5ms | 11.1ms | 19.2× (derived: 213.5/11.1) |
| values + vectors | 128 | 25.6ms | 1.76ms | 14.5× |

The reduction's O(n³) hot loop is `gemv` + a symmetric rank-2 update (two contiguous `axpy` calls per
row) and runs once; Jacobi does several full sweeps of strided column rotations — an algorithm choice,
not a micro-optimization.

Current absolute number at a larger representative size, N=1024 (`Benchmarks/EigenSvdBenchmark.cs`).
AMD Ryzen 9 9950X3D, single CCD pinned, 2026-07-05, commit `0714c97`, Unity Editor batchmode (checks
likely on):

| Method | dtype | med(ms) |
|---|---|---|
| `eigenvaluesSymmetric` (values only) | float | 162.97 |
| `eigenvaluesSymmetric` | double | 195.59 |
| `eigenSymmetric` (values + vectors) | float | 428.56 |
| `eigenSymmetric` | double | 545.01 |

`LOBPCG.lobpcg` (`Benchmarks/LOBPCGBenchmark.cs`), dense SPD `A = MᵀM + I`, N=512, k=4 smallest,
maxIter fixed at 50 (deterministic timing — same convention as the other iterative-solver
benchmarks; `tol` is set near machine-epsilon so the budget is never met early). Same
machine/date/commit/config as above:

| dtype | med(ms) | iterations | converged | maxResidual |
|---|---|---|---|---|
| float | 84.91 | 50 | 0/4 | 7.2×10⁻² |
| double | 85.34 | 50 | 0/4 | 2.2×10⁻² |

(`converged`/`maxResidual` show the fixed 50-iteration budget makes real but incomplete progress on
this well-conditioned test matrix — the point of this benchmark is the per-iteration cost, not a
convergence demonstration; a real caller would set a reachable `tol` instead.)

`powerIteration`/`inversePowerIteration`/`lanczos`/`lanczosVectors` — still not benchmarked.
