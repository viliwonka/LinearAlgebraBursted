# Direct solvers & the diagnostics-struct convention

The direct (non-iterative) solve entry points live on the factorization classes themselves — each of
the decompositions in [decompositions.md](decompositions.md) exposes its own
`decompSolve`/`solveInPlace`, following the shared `decomp`/`decompInPlace`/`decompSolve`/`solveInPlace`
token grid (see [naming-style-guide](../dev/naming-style-guide.md) and
[spec-solver-api-rework](../dev/spec-solver-api-rework.md)). They are built on the triangular-solve
primitives, which live on [`Blas`](la-primitives.md) as the substitution counterpart to its GEMM/GEMV kernels.
Iterative and least-squares solvers live on `Krylov` and are covered in
[least-squares.md](least-squares.md); this page is the direct (non-iterative) family plus the
diagnostics-struct convention shared by every solver in the library.

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

This is the level-2 → level-3 jump: each substitution step is a contiguous axpy across the `k`
right-hand sides, so each factor entry is loaded once and reused across all of them, and `QᵀB` / `UᵀB`
become GEMMs. Results match the column-by-column vector solve to summation-order rounding.

**End-to-end speedup, the fused `solveInPlace` path** (`Benchmarks/MultiRhsSolveBenchmark.cs`, whole
`AX=B`, square N=512, float, Ryzen 9 9950X3D single-thread Burst). `solveInPlace` fuses factor+solve
into one destructive call, so there is no "factor once then loop" with it — calling it per RHS
re-factorizes every time. The block adds the k-column solve onto a *single* factorization, so its
total is nearly flat in `k`, while looping the one-call API is `k · (factor + solve)`:

| solver | 1-RHS `solveInPlace` (ms) | block, k=16 | block, k=64 | block, k=256 | speedup @ k=256 |
|---|---|---|---|---|---|
| LU | 2.9 | 3.96 | 3.55 | 4.95 | 151× |
| Cholesky | 2.05 | 2.94 | 2.61 | 3.92 | 134× |
| QR | 4.7 | 5.86 | 5.49 | 7.46 | 160× |
| QRCP | 7.0 | 8.47 | 8.05 | 10.22 | 175× |

One block call for *any* `k` costs about the same as a *single* `solveInPlace`; the speedup ≈ `k`
because it eliminates the `k`-fold refactorization, not because the block solve itself is faster. The
fused QR/QRCP block also confirms the no-Q saving: QR's block (~7.5ms) undercuts the form-Q
`decompInPlace` route because `QᵀB` is streamed during factorization and `Q` is never reconstructed.

If you already hold a factorization and only want the marginal cost of more right-hand sides, use the
`decompSolve` block overload instead (`B` preserved): that isolates the level-2 → level-3 TRSM gain
(each factor entry loaded once, reused across all `k` columns) without re-paying the factorization.

## Diagnostics-struct convention

Every solver returns its info struct **by value**, with an implicit `bool` conversion (`info ==
true` reads as "solved") so `if (solve(...))` call shapes keep compiling:

| Struct | Fields | Used by |
|---|---|---|
| `DirectSolveInfo` | `status : DirectSolveStatus` | LU, CHO, un-pivoted QR/LQ, triangular solves — no rank concept |
| `RankInfo` | `status`, `rank` | QRCP (`solveInPlace`), CHOP |
| `SolveInfo` | `rnorm`, `iterations`, `status : IterativeSolveStatus` | square iterative solvers (`cg`/`pcg`/`minres`/`biCGStab`/`cgne`) |
| `LstsqInfo` | `rnorm`, `Arnorm`, `xnorm`, `iterations`, `status` | least-squares Krylov solvers (`cgls`/`lsqr`/`lsmr`) |

`DirectSolveStatus`: `Success, Singular, NotPositiveDefinite, Indefinite, RankDeficient`.
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
