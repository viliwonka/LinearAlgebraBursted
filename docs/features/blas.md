# Blas, Norms & Analysis — core numeric primitives

Three merged (bare-name) classes covering the primitives everything else is built from.

## Blas — dot, transpose, outer product, GEMM

Every op has a zero-alloc `ref`-destination primitive and an allocating wrapper of the same name
(see [zero-alloc-ops](../zero-alloc-ops.md) for the aliasing-guard rules):

- `dot(floatN a, floatN b)` / `dot(a, b, start, end)` — vector dot product (+ ranged form).
- `dot(in floatMxN A, in floatN x, ref floatN result)` — mat·vec; `dot(in floatN y, in floatMxN A,
  ref floatN result)` — vec·mat; `dot(in floatMxN a, in floatMxN b, ref floatMxN c, bool
  transposeA = false)` — mat·mat (GEMM), dispatching to `matMatDot` or `matMatDotTransA`.
- `outerDot(in floatN a, in floatN b, ref floatMxN result)` — outer product.
- `trans(in floatMxN A, ref floatMxN T)` — transpose.
- `householderInPlace(ref floatMxN matrix, in floatN u)` — apply a Householder reflection directly.

**GEMM tiling:** `matMatDot`/`matMatDotTransA` hold an 8×16 block of the output in named scalar
accumulators across the whole k-reduction (each `A` value reused 16×, each `B` value 8×). Bit-identical
to the untiled kernel (no k-splitting, same accumulation order) — a pure ILP/reuse win, not a
rounding change. Small matrices and remainder rows fall back to the untiled kernel.

## Norms

`L1`/`L2`/`LInf(in floatN|floatMxN)`, ranged `L2Range(a, start, end)`; in-place `NormalizeL1`/
`NormalizeL2`/`NormalizeLMax`/`NormalizeLP(x, p)` (each returns the pre-normalization length) and the
enum-dispatched `Normalize(x, Norm n)`; per-axis `NormalizeRows`/`NormalizeColumns(ref A, Norm n)`
(NaN-safe: a zero-norm row/column is left unchanged); induced matrix norms `matrixL1`/`matrixLInf`
(max abs column/row sum) and `matrixL2` (spectral norm, via SVD).

## Analysis

Structural predicates: `isAnyNan`/`isAnyInf`, `isZero(x, eps)`, `isIdentity`/`isSymmetric`/
`isDiagonal`/`isUpperTriangular`/`isLowerTriangular`/`isOrthogonal(A[, epsilon])`, `any`/`all(in
boolN|boolMxN)`. Scalar matrix metrics — moved here from `Blas` so "summarizes a matrix" and
"computes a product" don't share a class: `trace(A)`, `cond(A)` (κ₂ via SVD, `+Infinity` if
singular), `rank(A[, relTol])` (singular values above `relTol·σmax`, auto-tolerance if omitted).
`determinant` lives on `LU` instead (it needs a factorization, not just a summary read).

## Benchmarks

GEMM register-tiling (8×16 micro-kernel vs. the prior untiled axpy-GEMM), single-thread, this
machine, bit-identical output (commit `5790605`):

| Size | float (GFLOP/s) | double (GFLOP/s) |
|---|---|---|
| 512² | 70 → 93 (+33%) | 34 → 50 (+47%) |
| 1024² | 69 → 86 (+25%) | 34 → 50 (+47%) |

~70 GFLOP/s (float, untiled) is the ceiling every blocked decomposition in
[decompositions.md](decompositions.md) is measured against — none of them exceed it since they all
route their trailing-matrix updates through this same kernel.
