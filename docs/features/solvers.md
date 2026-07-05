# Solvers — direct solve & the diagnostics-struct convention

`Solvers` holds the square/triangular direct-solve primitives; the factorizations in
[decompositions.md](decompositions.md) each expose their own `decompSolve`/`solveInPlace` entry point
built on top, following the shared `decomp`/`decompInPlace`/`decompSolve`/`solveInPlace` token grid
(see [naming-style-guide](../naming-style-guide.md) and
[spec-solver-api-rework](../spec-solver-api-rework.md)). Iterative and least-squares solvers are
covered in [least-squares.md](least-squares.md); this page is the direct (non-iterative) family plus
the diagnostics-struct convention shared by every solver in the library.

## Direct solve family

- `Solvers.triUpper(ref U, ref b_to_x)` / `Solvers.triLower(ref L, ref b_to_x)` — in-place triangular
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

## Benchmarks

See [decompositions.md](decompositions.md) for the factorization costs each solve is built on, and
the README's benchmark table for `QR.solveInPlace`'s measured 74.7× vectorization win (commit
`eadf6a8`). No standalone benchmark isolates the triangular-solve step itself (it's O(n²), dominated
by the O(n³) factorization in every measured case).

End-to-end "solve `Ax=b`" (`Benchmarks/DirectSolveBenchmark.cs`), square N=1024. LU and CHO time the
explicit `decomp`+`decompSolve` composition (A preserved, distinct from L/U); QR times the fused
`solveInPlace` (A and b destroyed). AMD Ryzen 9 9950X3D, single CCD pinned, 2026-07-05, commit
`0714c97`, Unity Editor batchmode (`ENABLE_UNITY_COLLECTIONS_CHECKS` likely on — not the
release/player-build shape):

| Solve | min(ms) | med(ms) |
|---|---|---|
| `LU.decomp` + `LU.decompSolve` (partial-pivot LU), float | 16.62 | 16.67 |
| `LU.decomp` + `LU.decompSolve`, double | 26.42 | 26.51 |
| `CHO.decomp` + `CHO.decompSolve`, float | 12.21 | 12.23 |
| `CHO.decomp` + `CHO.decompSolve`, double | 16.44 | 16.63 |
| `QR.solveInPlace` (square), float | 37.84 | 37.98 |
| `QR.solveInPlace` (square), double | 63.21 | 63.52 |
