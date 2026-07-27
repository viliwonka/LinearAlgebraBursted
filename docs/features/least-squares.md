# Least squares — direct routes & iterative Krylov solvers

Three direct routes (pick by what you know about `A`'s rank/shape) plus three iterative Krylov
solvers for large/sparse `A`. All share the diagnostics-struct convention from
[solvers.md](solvers.md).

## Direct routes

- **`QR.solveInPlace`** — full column rank, square or tall. Fastest, no rank safety net; fused kernel,
  `A`/`b` exit as undefined scratch.
- **`QRCP.solveInPlace(ref A_to_Q, ref b, ref x, ref R, ref Pivot P, ref u[, float relTol])`** —
  rank-deficient-safe via QRCP; `relTol < 0` auto-selects a tolerance
  (`max(m,n)·Consts.floatZeroThreshold`). Returns the *basic* (truncated) solution, not minimum-norm.
  Factors `A`'s own buffer directly (no Q scratch, no copy) — `A_to_Q` exits as a usable orthogonal
  factor; `b` is preserved. Returns `RankInfo`.
- **`SVD.pinvSolve`** — minimum-norm least-squares for any shape/rank, via `SVD.thin`. Slowest of the
  three, most robust. See [svd.md](svd.md).

## Iterative (`Krylov`, generic over `IfloatLinearOperator` — dense and [sparse BSR](sparse-bsr.md)
share one body, see that page)

- **`lsqr<TOp>(...)`** — Golub-Kahan bidiagonalization + incremental Givens QR; more robust than
  plain CG on the normal equations for ill-conditioned `A`.
- **`lsmr<TOp>(...)`** — same bidiagonalization as `lsqr`, MINRES-folded onto the normal equations;
  `‖Aᵀr‖` decreases monotonically, giving a cleaner early-stopping signal.

All return `LstsqInfo` (`rnorm`, `Arnorm`, `xnorm`, `iterations`, `status`); `Arnorm =
‖Aᵀ(b-Ax)‖` (damped: `‖Aᵀ(b-Ax) - damp²x‖`) is the true optimality measure and is always "free"
(already-tracked or one dot product, never an extra matvec). `Krylov.lstsqResidual` independently
recomputes a certified `LstsqInfo` for auditing (costs one extra Apply+ApplyT).

## Damping & preconditioning

- **Tikhonov damping** — every Krylov least-squares solver takes a trailing `float damp`, minimizing
  `‖Ax-b‖² + damp²‖x‖²`. `damp == 0` is bit-identical to the undamped solve (unified behind one
  parameter, not a separate code path).
- **Warm start**: `lsqr`/`lsmr` regularize the *correction* `‖x-x₀‖`, not `‖x‖` — mind this when
  warm-starting a damped solve.
- **Jacobi (AᵀA column-equilibration) preconditioning** — `lsqrJacobi`/`lsmrJacobi(in A,
  in b, ref x, ...)` build a column-norm scale via `Blas.columnNormsSquared`/`buildJacobiScale`, solve
  the scaled system, then unscale and re-derive diagnostics in original coordinates. Cold-start only
  (no `damp`/warm-start parameter); use the composable primitives directly for custom control.
- **`cgne<TOp>`** (with `craig`/`craigmr` as siblings) is the complementary case: minimum-norm
  solution of a *consistent* (typically underdetermined) system, vs. the overdetermined/inconsistent
  target above.

## Performance

LSQR's sparse rectangular solve reaches ~7–8× dense at 7% fill, below the ~14× the square
solvers get, because the BSR transpose traversal (`ApplyT`) is less cache-friendly than a forward
`spMV` — see [sparse-bsr.md](sparse-bsr.md).

Direct least-squares solve, overdetermined (tall m×n): plain `QR.solveInPlace` vs. rank-safe
`QRCP.solveInPlace` on the same shapes. Both are fused (neither
reconstructs Q); the remaining gap is the column-pivoting overhead — per-reflector partial-norm
recomputes plus the pivoted panel's extra bookkeeping — about 1.15–1.3× over plain QR. Ryzen 9
9950X3D, single-thread Burst, median of 9:

| Kernel | Shape | float med(ms) | double med(ms) |
|---|---|---|---|
| `QR.solveInPlace` | 2048×512 | 31.24 | 49.01 |
| `QR.solveInPlace` | 2048×1024 | 93.42 | 160.56 |
| `QRCP.solveInPlace` | 2048×512 | 36.16 | 64.68 |
| `QRCP.solveInPlace` | 2048×1024 | 104.58 | 173.96 |
