# Direct solvers & the diagnostics-struct convention

Direct (non-iterative) solve entry points live on the factorization classes themselves, following the shared
`decomp`/`decompInPlace`/`decompSolve`/`solveInPlace` token grid. They build on triangular-solve primitives
in [`Blas`](la-primitives.md). Iterative and least-squares solvers live on `Krylov` (see [least-squares.md](least-squares.md));
this page covers the direct family and the diagnostics-struct convention shared by all solvers in the library.

## Direct solve family

- `Blas.triUpper(ref U, ref b_to_x)` / `Blas.triLower(ref L, ref b_to_x)` — in-place triangular
  solves (precondition: non-singular diagonal, unguarded); `triUpperLU`/`triLowerLU` overloads also
  apply a `Pivot`.
- `LU.decompSolve(ref LU, in Pivot P, ref b_to_x)` / `decompSolve(ref L, ref U, in Pivot P, ref
  b_to_x)` — solve from either LU form (compact in-place or split L/U), factors read-only.
  `LU.solveInPlace(ref A_to_LU, ref Pivot P, ref b_to_x)` fuses `decompInPlace`+`decompSolve` into one
  driver; `A_to_LU` exits as a usable factor (valid input to a further `decompSolve`).
- `CHO.decompSolve(ref L, ref b_to_x)` — solve from a factor, read-only. `CHO.solveInPlace(ref
  A_to_L, ref b_to_x)` fuses `decompInPlace`+`decompSolve`; `A_to_L` exits as a usable factor. (The
  old `choleskySolve(in A, ref L, ref b)` two-line composition-in-disguise was deleted — write the
  explicit `CHO.decomp` + `CHO.decompSolve` composition if `A` must survive.)
- `QR.decompSolve(ref Q, ref R, ref b, ref x)` — solve from a precomputed QR factorization, reusable
  across multiple `b` (`b` preserved). `QR.solveInPlace(ref A, ref b, ref x[, ref u])` — precondition:
  full column rank (unguarded divide on a rank-deficient diagonal); fused kernel that streams `Qᵀb`
  without ever forming `Q`, so `A`/`b` exit as undefined scratch, not usable factors. For
  rank-deficient/wide systems use `QRCP.solveInPlace` (truncated LS, see
  [least-squares.md](least-squares.md)) or `SVD.pinvSolve` (minimum-norm, see [svd.md](svd.md)).

## Multiple right-hand sides (`AX = B`)

Every direct solve above has a matrix-RHS overload: pass an `M×k` matrix in place of the length-`M`
vector and each **column** is a separate right-hand side, solved together. Same method names, same
destructive/preserving contracts — the argument type (`fProxyN` → `fProxyMxN`) picks the overload.

- `Blas.triUpper/triLower/triUpperLU/triLowerLU(ref factor, ref B_to_X)` — the TRSM primitives.
- `LU.decompSolve` / `LU.solveInPlace`, `CHO.decompSolve` / `CHO.solveInPlace`,
  `CHOP.decompSolve` / `CHOP.solveInPlace` (full-rank *and* rank-deficient min-norm) — factor once,
  solve a whole block; the pivot is applied to `B`'s rows.
- `QR.decompSolve(Q, R, B, X)` (reuses one factorization, `QᵀB` is a single GEMM, `B` preserved) and
  the fused `QR.solveInPlace(A, B, X)` (destroys `A`/`B`, never forms `Q`).
- `QRCP.decompSolve(Q, R, P, B, X[, relTol])` (from a precomputed factorization, `B` preserved) and
  `QRCP.solveInPlace(A, B, X[, relTol])` — rank-safe truncated least squares. Like the vector form,
  the multi-RHS `solveInPlace` is **fused and destructive**: it applies `Qᵀ` to `B`'s columns *during*
  factorization and never reconstructs `Q` (the ~⅓-runtime saving), so `A` and `B` are both destroyed.
- `LQ.minNormSolve(A, B, X)` (underdetermined min-norm) and `SVD.pinvSolve(A, B, X)` (any shape/rank).

Block solves reuse each factor entry across all `k` right-hand sides: each substitution step is a contiguous
axpy across the block, and `QᵀB` / `UᵀB` become GEMMs (level-3 BLAS). Results match the column-by-column
vector solve to summation-order rounding.

**Speedup: fused `solveInPlace` vs. looping per-RHS** (`Benchmarks/MultiRhsSolveBenchmark.cs`, N=512 square, float, Ryzen 9 9950X3D, Burst single-thread).
The fused `solveInPlace` factors once and solves all `k` RHS together (cost ≈ factor + `k` · solve). Looping the one-call API re-factors each time (cost = `k` · (factor + solve)):

| solver | 1-RHS `solveInPlace` (ms) | block, k=16 | block, k=64 | block, k=256 | speedup @ k=256 |
|---|---|---|---|---|---|
| LU | 2.9 | 3.96 | 3.55 | 4.95 | 151× |
| Cholesky | 2.05 | 2.94 | 2.61 | 3.92 | 134× |
| QR | 4.7 | 5.86 | 5.49 | 7.46 | 160× |
| QRCP | 7.0 | 8.47 | 8.05 | 10.22 | 175× |

One block call is almost as cheap as a single-RHS `solveInPlace` — the speedup comes from eliminating the
`k`-fold refactorization. QR's block also shows the benefit of not forming Q: `QᵀB` is streamed during
factorization, so `Q` is never reconstructed.

If you already hold a factorization, use `decompSolve` block overload instead (preserves `B`): it pays only
the triangular-solve cost, reusing the factor across all `k` columns, without refactoring.

## Diagnostics-struct convention

Every solver returns its info struct **by value**, with an implicit `bool` conversion (`info ==
true` reads as "solved") so `if (solve(...))` call shapes keep compiling:

| Struct | Fields | Used by |
|---|---|---|
| `DirectSolveInfo` | `status : DirectSolveStatus` | LU, CHO, un-pivoted QR/LQ, triangular solves — no rank concept |
| `RankInfo` | `status`, `rank` | QRCP (`solveInPlace`), CHOP, and the SVD-backed rank-revealing calls (`pinvSolve`, `pseudoInverse`, `nullspaceBasis`, `rangeBasis`) |
| `SolveInfo` | `rnorm`, `iterations`, `status : IterativeSolveStatus` | square iterative solvers (`cg`/`minres`/`biCGStab`/`gmres`) |
| `LstsqInfo` | `rnorm`, `Arnorm`, `xnorm`, `iterations`, `status` | least-squares Krylov solvers (`lsqr`/`lsmr`) |

`DirectSolveStatus`: `Success, Singular, NotPositiveDefinite, Indefinite, RankDeficient,
NotConverged` (the last reported by the SVD-backed rank-revealing calls when the SVD fails to
converge — see [svd.md](svd.md)).
`IterativeSolveStatus`: `Converged, MaxIterations, Breakdown`. All diagnostic fields come from numbers
each solver already tracks internally (residual norms, iteration counts) — never an extra matvec
just to fill in a struct.

Eigensolvers follow this same convention with their own structs (`EigenSolveInfo`, `LanczosInfo`,
`LOBPCGInfo`) — see [eigen.md](eigen.md#diagnostics-structs) rather than duplicating them here.

## Performance

See [decompositions.md](decompositions.md) for the factorization costs each solve is built on. The
triangular-solve step itself is O(n²) and dominated by the O(n³) factorization in every case.

End-to-end "solve `Ax=b`" (`Benchmarks/DirectSolveBenchmark.cs`), square N=1024. LU and CHO time the
explicit `decomp`+`decompSolve` composition (A preserved, distinct from L/U); QR times the fused
`solveInPlace` (A and b destroyed). Ryzen 9 9950X3D, single-thread Burst, median of 9:

| Solve | min(ms) | med(ms) |
|---|---|---|
| `LU.decomp` + `LU.decompSolve` (partial-pivot LU), float | 15.28 | 15.33 |
| `LU.decomp` + `LU.decompSolve`, double | 26.96 | 27.01 |
| `CHO.decomp` + `CHO.decompSolve`, float | 12.00 | 12.08 |
| `CHO.decomp` + `CHO.decompSolve`, double | 16.54 | 16.56 |
| `QR.solveInPlace` (square), float | 36.26 | 36.32 |
| `QR.solveInPlace` (square), double | 62.18 | 62.79 |
