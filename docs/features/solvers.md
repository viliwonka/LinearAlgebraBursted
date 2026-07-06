# Direct solvers & the diagnostics-struct convention

The direct (non-iterative) solve entry points live on the factorization classes themselves — each of
the decompositions in [decompositions.md](decompositions.md) exposes its own
`decompSolve`/`solveInPlace`, following the shared `decomp`/`decompInPlace`/`decompSolve`/`solveInPlace`
token grid (see [naming-style-guide](../naming-style-guide.md) and
[spec-solver-api-rework](../spec-solver-api-rework.md)). They are built on the triangular-solve
primitives, which live on [`Blas`](blas.md) as the substitution counterpart to its GEMM/GEMV kernels.
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
