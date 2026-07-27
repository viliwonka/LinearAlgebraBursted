# Eigensolvers

`Eigen` - dense symmetric/non-symmetric eigensolvers plus a matrix-free trio that works over dense
*or* [sparse BSR](sparse-bsr.md) operands through the shared `IfloatLinearOperator` interface.

## Dense

- **`Eigen.symmetricInPlace(ref A, ref eigenvalues, ref V, ...)`** - the default: Householder
  tridiagonalization (with orthogonal accumulation) + implicit-shift QL, all eigenpairs, descending
  order, `A` destroyed. **`Eigen.valuesSymmetricInPlace(ref A, ref eigenvalues, ...)`** - values-only, skips
  the eigenvector accumulation entirely, faster still.
- `Eigen.decompInPlace` (cyclic two-sided Jacobi) is `[Obsolete]` - kept for cross-validation and
  superior accuracy on graded spectra, but column-oriented rotations resist SIMD.
- **`Eigen.valuesQRInPlace(ref A, ref eigenvaluesReal, ref eigenvaluesImag, ...)`** - non-symmetric,
  Francis double-shift QR on an upper-Hessenberg reduction (elmhes+hqr). Values only; `A` destroyed;
  complex-conjugate pairs are represented as real/imag arrays (no complex type), via 2×2 Schur blocks.

## Matrix-free (sparse-capable)

Generic `<TOp> where TOp : struct, IfloatLinearOperator`, with thin dense (`floatMxN`) and
[`floatBSR`](sparse-bsr.md) forwarders - same body, not forked:

- **`powerIteration<TOp>(in A, ref v, ref w, out lambda, ...)`** - dominant eigenpair, Rayleigh
  quotient.
- **`inversePowerIteration<TOp>(in A, ..., out lambda, ...)`** - smallest eigenpair of SPD `A`, via an
  inner `Krylov.cg` solve each outer iteration (no explicit inverse formed).
- **`lanczos<TOp>(in A, ref ws, ref eigenvalues, int steps, ...)`** - twice-reorthogonalized symmetric
  Lanczos tridiagonalization + `Eigen.valuesSymmetricInPlace` on the result → Ritz **values**.
  **`lanczosVectors<TOp>(...)`** - same tridiagonalization, then forms Ritz **vectors** too (not
  zero-alloc - allocates 3 Temp vectors internally via `Eigen.symmetricInPlace`).
- **`Eigen.lobpcg<TOp[,TPre]>(in A[, in M], ref ws, int k, float tol, int maxIter)`** - blocked
  Locally Optimal Block Preconditioned Conjugate Gradient: the `k` SMALLEST eigenpairs of a symmetric
  operator, via deflation-based locking (a converged pair is frozen and projected out of the active
  subspace) and a small dense Rayleigh-Ritz sub-problem solved with `Eigen.symmetricInPlace` (a 3-block
  `[X,W,P]` reduction that falls back to 2-block `[X,W]` if the 3-block Cholesky is too
  ill-conditioned). Dense (`floatMxN`), [BSR](sparse-bsr.md), and BSR+`floatBlockJacobi`
  (preconditioned) overloads share this implementation.
  **Results are ascending** (index 0 = smallest), unlike other `Eigen` methods. Zero-alloc at the O(n) scale via
  `floatLOBPCGCache` (`new floatLOBPCGCache(n, k, Allocator.Persistent)`, reusable/warm-startable
  across calls); the
  O(k)-scale Rayleigh-Ritz sub-solve still allocates a few small, bounded Temp vectors internally -
  the same exception `lanczosVectors` already has. Returns `LOBPCGInfo`.

Use for sparse smallest-eigenpair problems (structural-stability, buckling, modal-analysis, Fiedler
vector). For dense small-scale, prefer `Eigen.symmetricInPlace`.

**Generalized pencil form** - `lobpcg` also solves `A·x = λ·B·x` with B SPD: every overload has a
`+B` twin (generic operator, dense, BSR, BSR+block-Jacobi) that B-orthonormalizes the basis and
returns B-orthonormal eigenvectors, ascending. `B = I` forwarders are bit-identical to the standard
path. The buckling recipe: for
`K_E·φ + λ·K_G·φ = 0` put the SPD elastic stiffness `K_E` in the **B slot** and the (typically
indefinite) geometric stiffness `K_G` in the A slot; the returned ascending `μ[0]` (most negative)
gives the smallest positive critical load as `λ_cr = −1/μ[0]`.

## Diagnostics structs

Eigensolvers follow the same by-value, implicit-`bool` diagnostics convention as the
[direct solvers](solvers.md), with their own structs (all reuse `IterativeSolveStatus` - no dedicated
eigensolver enum):

| Struct | Fields | Used by |
|---|---|---|
| `EigenSolveInfo` | `iterations`, `residual` (double, `‖Av-λv‖`), `status` | `powerIteration`, `inversePowerIteration` |
| `EigenInfo` | `sweeps`, `converged`, `status` | `symmetricInPlace`, `valuesSymmetricInPlace`, `valuesQRInPlace`, `decompInPlace` |
| `LanczosInfo` | `produced` (≤ `steps`, less only on early breakdown), `status` | `lanczos`, `lanczosVectors` |
| `LOBPCGInfo` | `iterations`, `converged` (0..k pairs locked), `maxResidual` (double, worst-case relative residual over all k pairs), `status` | `Eigen.lobpcg` |

## Performance

The symmetric eigensolvers use Householder tridiagonalization: its O(n³) hot loop is a `gemv` plus a
symmetric rank-2 update, run once, rather than the repeated strided column-rotation sweeps of a Jacobi
solver.

Ryzen 9 9950X3D, single-thread Burst, median of 9. N=1024:

| Method | dtype | med(ms) |
|---|---|---|
| `Eigen.valuesSymmetricInPlace` (values only) | float | 161.86 |
| `Eigen.valuesSymmetricInPlace` | double | 199.36 |
| `Eigen.symmetricInPlace` (values + vectors) | float | 420.25 |
| `Eigen.symmetricInPlace` | double | 542.04 |

`Eigen.lobpcg`, dense SPD `A = MᵀM + I`, N=512, k=4 smallest,
maxIter fixed at 50 (deterministic timing; `tol` is set near machine-epsilon so the budget is never
met early):

| dtype | med(ms) | iterations | converged | maxResidual |
|---|---|---|---|---|
| float | 84.32 | 50 | 0/4 | 7.2×10⁻² |
| double | 84.43 | 50 | 0/4 | 2.2×10⁻² |

(`converged`/`maxResidual` show the fixed 50-iteration budget makes real but incomplete progress on
this well-conditioned test matrix - the point of this benchmark is the per-iteration cost, not a
convergence demonstration; a real caller would set a reachable `tol` instead.)

`powerIteration`/`inversePowerIteration`/`lanczos`/`lanczosVectors` - still not benchmarked.
