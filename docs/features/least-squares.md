# Least squares — direct routes & iterative Krylov solvers

Three direct routes (pick by what you know about `A`'s rank/shape) plus three iterative Krylov
solvers for large/sparse `A`. All share the diagnostics-struct convention from
[solvers.md](solvers.md).

## Direct routes

- **`QR.qrDirectSolve`** — full column rank, square or tall. Fastest, no rank safety net.
- **`QR.qrcpDirectSolve(ref A, ref b, ref x, ..., float relTol)`** — rank-deficient-safe via QRCP;
  `relTol < 0` auto-selects a tolerance (`max(m,n)·Consts.floatZeroThreshold`). Returns the *basic*
  (truncated) solution, not minimum-norm. Returns `RankRevealingInfo`.
- **`SVD.pinvSolve`** — minimum-norm least-squares for any shape/rank, via `svdThin`. Slowest of the
  three, most robust. See [svd.md](svd.md).

## Iterative (`Solvers`, generic over `IfloatLinearOperator` — dense and [sparse BSR](sparse-bsr.md)
share one body, see that page)

- **`cgls<TOp>(in A, in b, ref x, ..., float damp)`** — CG on the normal equations, recomputes `Aᵀr`
  fresh each iteration (avoids CGNR drift).
- **`lsqr<TOp>(...)`** — Golub-Kahan bidiagonalization + incremental Givens QR; more robust than
  `cgls` on ill-conditioned `A`.
- **`lsmr<TOp>(...)`** — same bidiagonalization as `lsqr`, MINRES-folded onto the normal equations;
  `‖Aᵀr‖` decreases monotonically, giving a cleaner early-stopping signal.

All three return `LstsqInfo` (`rnorm`, `Arnorm`, `xnorm`, `iterations`, `status`); `Arnorm =
‖Aᵀ(b-Ax)‖` (damped: `‖Aᵀ(b-Ax) - damp²x‖`) is the true optimality measure and is always "free"
(already-tracked or one dot product, never an extra matvec). `Solvers.lstsqResidual` independently
recomputes a certified `LstsqInfo` for auditing (costs one extra Apply+ApplyT).

## Damping & preconditioning

- **Tikhonov damping** — every Krylov least-squares solver takes a trailing `float damp`, minimizing
  `‖Ax-b‖² + damp²‖x‖²`. `damp == 0` is bit-identical to the undamped solve (unified behind one
  parameter, not a separate code path).
- **Warm start** differs by solver: `cgls` regularizes `‖x‖` for *any* starting `x₀`; `lsqr`/`lsmr`
  regularize the *correction* `‖x-x₀‖` instead — pick accordingly if you're warm-starting.
- **Jacobi (AᵀA column-equilibration) preconditioning** — `cglsJacobi`/`lsqrJacobi`/`lsmrJacobi(in A,
  in b, ref x, ...)` build a column-norm scale via `Blas.columnNormsSquared`/`buildJacobiScale`, solve
  the scaled system, then unscale and re-derive diagnostics in original coordinates. Cold-start only
  (no `damp`/warm-start parameter); use the composable primitives directly for custom control.
- **`cgne<TOp>`** (Craig's method) is the complementary case: minimum-norm solution of a *consistent*
  (typically underdetermined) system, vs. `cgls`'s overdetermined/inconsistent target.

## Benchmarks

Not independently benchmarked for the direct or iterative least-squares routes as a group. The one
measured number is structural, not a speed one: CGLS/LSQR's sparse rectangular solve undershoots the
ideal density ratio (~7-8× vs. the ~14× square solvers get at 7% fill) because the BSR transpose
traversal (`ApplyT`) is less cache-friendly than a forward `spMV` — see
[sparse-bsr.md](sparse-bsr.md) for the materialized-transpose mitigation (commit `06035da`, measured
perf-neutral on its own, commit `724ceb0`).

Direct least-squares solve, overdetermined (tall m×n): plain `QR.qrDirectSolve` vs. rank-safe
`QR.qrcpDirectSolve` on the same shapes (`Benchmarks/QRVariantsBenchmark.cs`, "TALL overdetermined
least squares" section — the gap is the column-pivoting overhead: exact partial-norm recomputes plus
Q reconstruction). AMD Ryzen 9 9950X3D, single CCD pinned, 2026-07-05, commit `95a1897`, Unity
Editor batchmode (checks likely on):

| Kernel | Shape | float med(ms) | double med(ms) |
|---|---|---|---|
| `qrDirectSolve` | 2048×512 | 34.46 | 51.99 |
| `qrDirectSolve` | 2048×1024 | 95.62 | 159.63 |
| `qrcpDirectSolve` | 2048×512 | 72.68 | 132.54 |
| `qrcpDirectSolve` | 2048×1024 | 234.71 | 413.11 |
