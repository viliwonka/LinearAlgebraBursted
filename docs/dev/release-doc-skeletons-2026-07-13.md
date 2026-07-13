# Facts-only skeletons for missing public prose (2026-07-13)

Raw material for hand-written docs — facts verified against templates; no prose supplied on
purpose. Delete this file once the real docs exist.

## 1. README License section (currently just "[MIT](LICENSE)")

Facts that must reach the reader:
- Package license: MIT.
- `Assets/LinearAlgebra/Source/Third Party Notices.md` ships in the package and governs two
  derived implementations: `LP.ladBR` / `LP.ladFN` (ports of Koenker's GPL quantreg routines).
- Current status per that file: permission to distribute those two under MIT has been REQUESTED;
  until resolved, the package must not be redistributed.
- Everything HiGHS-derived (RevisedSimplex, DualSimplex, QP, MIP) is MIT-licensed and settled.

## 2. README Features list — missing bullets

- Kalman filtering: `Kalman.predict`/`update`, steady-state gain, EKF (user Jacobian functors),
  UKF (Van der Merwe scaled sigma points). Tested + benchmarked.
- Model-predictive control: `MPC.solve` — condensed linear MPC, box/soft constraints, warm-started
  active-set QP core, persistent per-horizon state. Tested + benchmarked.
- Nonlinear least squares / curve fitting: `Optimize.nlsSolve<TF>` / `Optimize.curveFit<TModel>`,
  Levenberg-Marquardt (Nielsen damping), robust losses `L2/Huber/Cauchy/Tukey`. Tested + benchmarked.
- (Pre-existing gap, lower priority: the 1D optimizers on `Optimize` — root-finding, minimization,
  gradient descent — have no README bullet or feature page either.)

## 3. docs/features/kalman.md (new page) — facts

- Class `Kalman`, float/double. State carried in `floatKFState` (`Kalman.State.fProxy.cs`).
- `predict(ref s, in F, in Q[, in B, in u])`, `update(ref s, in H, in R, in z)`.
- `steadyStateGain(...)` — solves the dual DARE via `Control.SDACore` (LQR duality).
- EKF: `ekfPredict<TModel>` / `ekfUpdate<TMeas>` — user supplies model + Jacobian as struct
  functors (interfaces in the same file).
- UKF: `ukfPredict<TModel>` / `ukfUpdate<TMeas>` over `fProxyUKFCache` (Van der Merwe scaled sigma
  points; alpha default 1 — see the ctor doc for the contract). Cache from
  `arena.floatUKFCache(n)`.
- Diagnostics: `KalmanInfo` (`Kalman.Info.cs`).
- Tests: KalmanTests / UKFTests; benchmark: KalmanBenchmark (numbers in TestResults after a run).

## 4. docs/features/mpc.md (new page) — facts

- `MPC.solve(ref floatMPCState s, in x0, in reference, ref u0out[, maxIter])`.
- Condensed formulation (state eliminated; QP over the input sequence), prestabilized dynamics,
  box input constraints + optional soft state walls; warm-started active-set QP core (`QP`).
- `floatMPCState(n, m, N, allocator)` carries the condensed matrices + warm basis across frames;
  its ctor solves a DARE for the prestabilizing gain (throws on non-convergent DARE).
- Returns `MPCInfo` (`iterations`, `activeSetChanges`, `objective`, `status`).
- Tests: MPCTests; benchmark: MPCBenchmark (cold + warm-frame rows).

## 5. docs/features/nls.md (new page) — facts

- `Optimize.nlsSolve<TF>(ref x, ref TF f, ...)` — TF : IfloatResidualFunction (residuals +
  Jacobian); Levenberg-Marquardt, Nielsen damping (Madsen/Nielsen/Tingleff alg. 3.16).
- `Optimize.curveFit<TModel>(in xs, in ys, ref params, ...)` — model functor overload.
- Robust losses via struct functors: `floatL2Loss` (default), `floatHuberLoss`,
  `floatCauchyLoss`, `floatTukeyLoss` (IRLS-style reweighting inside the LM loop).
- Returns `NLSInfo` (`NLS.Info.cs`): iterations, final cost, gradient norm, status.
- Tests: NLSTests; benchmarked alongside Kalman/MPC.

## 6. Small stale spots left for prose judgment (not fixed mechanically)

- `docs/features/generators.md`: the `Window` bullet reads like standalone functions; actual API is
  `enum WindowType { Box, Hann, Hamming, Blackman }` + `Generate.window(ref dest, WindowType)`.
- README vs sparse-bsr.md cite 15.05 vs 15.02 ms for the same CG-dense case (separate runs) —
  pick one number or re-run.
