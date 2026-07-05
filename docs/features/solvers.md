# Solvers — direct solve & the diagnostics-struct convention

`Solvers` holds the square/triangular direct-solve primitives; the factorizations in
[decompositions.md](decompositions.md) each expose their own `xxxDirectSolve`/`xxxSolve` entry point
built on top. Iterative and least-squares solvers are covered in
[least-squares.md](least-squares.md); this page is the direct (non-iterative) family plus the
diagnostics-struct convention shared by every solver in the library.

## Direct solve family

- `Solvers.solveUpperTriangular(ref U, ref x)` / `solveLowerTriangular(ref L, ref x)` — in-place
  triangular solves (precondition: non-singular diagonal, unguarded).
- `Solvers.solveQR(ref Q, ref R, ref b, ref x)` — solve from a precomputed QR factorization, reusable
  across multiple `b`; `solveQR(ref A, ref b, ref x)` forwards to `QR.qrDirectSolve`.
- `LU.luSolve(ref LU, in Pivot, ref b)` / `luSolve(ref L, ref U, in Pivot, ref b)` — solve from either
  LU form (compact in-place or split L/U).
- `Cholesky.choleskySolve(ref L, ref b)` — solve from a factor; `choleskySolve(in A, ref L, ref b)`
  factors then solves (destroys `A` even if `L` aliases it — a different aliasing contract than plain
  `choleskyDecomposition`, called out explicitly in the source).
- `QR.qrDirectSolve(ref A, ref b, ref x[, ref u])` — precondition: full column rank (unguarded divide
  on a rank-deficient diagonal). For rank-deficient/wide systems use `QR.qrcpDirectSolve` (truncated
  LS, see [least-squares.md](least-squares.md)) or `SVD.pinvSolve` (minimum-norm, see
  [svd.md](svd.md)).

## Diagnostics-struct convention

Every solver returns its info struct **by value**, with an implicit `bool` conversion (`info ==
true` reads as "solved") so `if (solve(...))` call shapes keep compiling:

| Struct | Fields | Used by |
|---|---|---|
| `DirectSolveInfo` | `status : DirectSolveStatus` | LU, plain Cholesky, un-pivoted QR/LQ, triangular solves — no rank concept |
| `RankRevealingInfo` | `status`, `rank` | QRCP (`qrcpDirectSolve`), pivoted Cholesky |
| `SolveInfo` | `rnorm`, `iterations`, `status : IterativeSolveStatus` | square iterative solvers (`cg`/`pcg`/`minres`/`biCGStab`/`cgne`) |
| `LstsqInfo` | `rnorm`, `Arnorm`, `xnorm`, `iterations`, `status` | least-squares Krylov solvers (`cgls`/`lsqr`/`lsmr`) |

`DirectSolveStatus`: `Success, Singular, NotPositiveDefinite, Indefinite, RankDeficient`.
`IterativeSolveStatus`: `Converged, MaxIterations, Breakdown`. All diagnostic fields come from numbers
each solver already tracks internally (residual norms, iteration counts) — never an extra matvec
just to fill in a struct.

Eigensolvers follow this same convention with their own structs (`EigenSolveInfo`, `LanczosInfo`,
`LOBPCGInfo`) — see [eigen.md](eigen.md#diagnostics-structs) rather than duplicating them here.

## Benchmarks

See [decompositions.md](decompositions.md) for the factorization costs each solve is built on, and
the README's benchmark table for `QR.qrDirectSolve`'s measured 74.7× vectorization win (commit
`eadf6a8`). No standalone benchmark isolates the triangular-solve step itself (it's O(n²), dominated
by the O(n³) factorization in every measured case).

End-to-end "solve `Ax=b`" (factor + triangular solve, `Benchmarks/DirectSolveBenchmark.cs`), square
N=1024. AMD Ryzen 9 9950X3D, single CCD pinned, 2026-07-05, commit `0714c97`, Unity Editor batchmode
(`ENABLE_UNITY_COLLECTIONS_CHECKS` likely on — not the release/player-build shape):

| Solve | min(ms) | med(ms) |
|---|---|---|
| `LU.luSolve` (partial-pivot LU), float | 16.62 | 16.67 |
| `LU.luSolve`, double | 26.42 | 26.51 |
| `Cholesky.choleskySolve`, float | 12.21 | 12.23 |
| `Cholesky.choleskySolve`, double | 16.44 | 16.63 |
| `QR.qrDirectSolve` (square), float | 37.84 | 37.98 |
| `QR.qrDirectSolve` (square), double | 63.21 | 63.52 |
