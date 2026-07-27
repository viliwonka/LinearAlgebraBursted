# Control — discrete-time LQR (`LQR` in namespace `BULA.Control`)

Infinite- and finite-horizon discrete linear-quadratic regulator: for `x_{k+1} = A x_k + B u_k`,
finds the feedback gain `K` minimizing `Σ(xᵀQx + uᵀRu)` under `u = -Kx`, by solving the discrete
algebraic Riccati equation (DARE).

- **`LQR.lqr(in A, in B, in Q, in R, ref K[, maxIter])`** — cold solve via SDA
  (structure-preserving doubling, quadratic convergence, ~10-25 steps typically). `A`/`Q` are `n×n`,
  `B` is `n×m`, `R` is `m×m`, `K` (output) is `m×n`.
- **`LQR.lqr(in A, in B, in Q, in R, ref K, ref floatLQRState state[, maxIter])`** —
  warm-started: reuses the carried Riccati solution `S` across calls, converging in a handful of
  cheap iterations for a slightly-changed `A`/`B` (the per-frame re-linearization case) instead of a
  fresh cold SDA solve. `state` must be constructed via `new floatLQRState(n, allocator)` (double
  build: `doubleLQRState`) before
  the first call (job-safe: this overload never allocates); the terminal `S` is written back only on
  a converged exit, so a non-converged call never corrupts a future warm seed.
- **`LQR.lqrSchedule(in A, in B, in Q, in R, in Qf, int N, ref Kschedule)`** — finite-horizon:
  backward Riccati recursion from a terminal cost `Qf` over `N` steps. `Kschedule` is `(N·m)×n`; the
  gain for step `k` lives in rows `[k·m, (k+1)·m)`.

`Q`/`R` are assumed symmetric PSD (not numerically validated); the `(R + BᵀSB)` solve routes
through the rank-revealing `CHOP`, so a semidefinite `R` degrades to a usable minimum-norm `K`
instead of failing. This is surfaced via `LQRInfo.rankDeficient`.

Returns `RiccatiInfo`: `iterations`, `residual` (relative Frobenius change at the last step),
`status : RiccatiStatus` (`Converged`/`MaxIterations`/`Diverged`), `rankDeficient`. Implicit
`bool` conversion (`== Converged`), so `if (Control.lqr(...))` reads as "did it converge". A
`Diverged` result (blowup detected, or an inner factorization broke down — the system is not
stabilizable/detectable, or the input is degenerate) always returns the last known-good iterate,
never a NaN-polluted one.
