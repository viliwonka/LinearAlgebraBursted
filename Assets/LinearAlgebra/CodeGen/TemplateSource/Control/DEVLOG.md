# DEVLOG — Control
Code comments state contracts only; history lives here (see CLAUDE.md).

## MPC (removed)

- 2026-07-27 | **MPC removed from the library** (user ruling: overscoped for v1.0, a control
  application rather than a linear algebra primitive, and still churning — recondense shipped the
  same day it was cut, unverified). Deleted: MPC/MPC.State/MPC.Info templates + tests + benchmark
  template, the hand half of MPCBenchmark, the determinism harness's `control/mpc.solve` case
  (control job is 5 hash slots now), and demo 15_HoverTankMPC with its three test files. The QP
  warm seams MPC drove (qpActiveSetCoreWarm / qpActiveSetCorePersist, persistent
  fProxyQPFactorState/fProxyQPReducedState) stay — they are caller-agnostic and tested MPC-free in
  QPFactorStateTests.
  LQR/DARE/Kalman untouched. The MPC entries below are history of deleted code. Revive from git
  (`37c6ee2f` and earlier) if MPC returns post-v1.0; iLQR explicitly ruled out for v1.0 too.

## MPC.recondense

- 2026-07-27 | **The successive-linearization seam.** Demo 15's phase-5 decision routed around full
  actuator-level MPC because rebuilding `fProxyMPCState` per frame costs 7.08 ms at (N=30, n=24, m=8)
  — most of it a COLD DARE for the terminal P plus reallocating every buffer, and a reconstruct also
  throws away the warm plan/working set/factorization. `MPC.recondense(ref s, A, B)` removes exactly
  those costs: the constructor's condensation moved into a shared `Condense(ref s)` fill (constructor
  allocates then fills; recondense refills in place), the terminal DARE re-solves WARM from a carried
  `fProxyLQRState` (`lqrCarry` — the same warm overload `LQR.lqr` already exposed), and the QP
  factorization is invalidated (`qpMeta[0]=0`, `[6]=1`) while `wstatus`/`uPlan`/`z` survive as the
  next solve's warm start. `rowConstRHS` is deliberately NOT rebuilt — `setSoftBound` may have moved
  it — which forced the constructor split: coefficients in `Condense`, RHS/senses written once by the
  constructor in the same row order. Failure contract mirrors the rest of the library: a
  non-converged DARE returns its `RiccatiInfo` and leaves the state byte-identical (the swap happens
  only after convergence; `lqrCarry`'s own only-write-on-Converged contract protects the seed).
  `Q`/`R` are now carried copies for this (they were construction-locals before), and the terminal
  cost split into `autoP`/`Pexplicit` so an explicit caller P stays verbatim while an auto P tracks
  the model. Motivating context: hovertank successive linearization at ~10 Hz against a 50 Hz solve
  loop — the servo slew limits bound how stale the model can go between recondenses by construction.

## MPC.setSoftBound

- 2026-07-26 | Added, driven by demo 15's anti-collision need: a moving obstacle constraint had no
  route short of reconstructing `fProxyMPCState`, and the MPC benchmark prices that at **7.08 ms** at
  (N=30, n=24, m=8) against a **0.45 ms** warm solve — two orders out, and per frame. Only
  `rowConstRHS` depends on `d`; `Phi`/`Gamma`/`H`/`GtQbar`/`Arows`/`qCoupling` and the carried
  `qpFactor`/`qpReduced` are all `d`-free, so the move is O(N·k) writes and reaches the QP through
  `BuildGeneralRHS` exactly the way a moved `x0` does. No warm-start invalidation: `RepairWorkingSet`
  already re-validates every entry against the current RHS each frame.
- 2026-07-26 | The per-stage form is a real capability gain, not sugar. The CONSTRUCTOR's `d` is one
  bound per row shared by every stage, but the condensed layout stores one RHS per (stage, row) —
  `rowConstRHS[k*nSoftPerStage + si]` — so a per-stage bound was always expressible and simply had no
  API. `SetSoftBoundPerStageBindsPerStage` pins a slack value (0.3) that NO constant bound can
  produce, which is what makes that claim testable rather than asserted.
- 2026-07-26 | **The state carried `d` TWICE** and the first cut of `setSoftBound` moved only one of
  them. `BuildWarmStartGuess`'s slack guess read `s.d[si]` while the constraint itself read
  `rowConstRHS`, so a moved wall produced a warm start seeded against the OLD bound — green on every
  pre-existing test, since before this API the two could never disagree. Fixed by deleting the `d`
  field outright rather than syncing it: `rowConstRHS` is indexed per (stage, row) and `s.d` is
  per-row only, so `s.d` structurally cannot represent a per-stage bound and would have stayed a
  stale-copy trap. Its only reader was that one line. Caught by the bit-exactness oracle below on
  the very first run — an approximate tolerance would have let it pass.
- 2026-07-26 | Test oracle worth reusing: build-at-d versus build-elsewhere-then-move are the SAME
  QP, both cold, so the two `u0` values must agree BIT-EXACTLY — an inexact match is the signature of
  `d` having leaked into the condensing. Approximate agreement would not have distinguished the two.

## fProxyMPCState.populated: native-backed, closing the gap with fProxyLQRState
- 2026-07-23 | `fProxyMPCState.populated` was a plain bool whose own doc comment claimed it "mirrors
  fProxyLQRState.populated" without actually doing so -- a job that ran `MPC.solve` via `IJob.Run()`
  (e.g. a receding-horizon frame job holding the state as a struct field) lost the flag on every call,
  forcing every warm tick back onto the cold QP path. Fixed by moving it behind a `NativeReference<int>`
  and re-exposing it as a `bool` property, the exact `fProxyLQRState.populated` idiom -- allocated in the
  main constructor (replacing the old `populated = false;` field-reset line), disposed in `Dispose()` and
  on the constructor's terminal-DARE-non-convergence early-throw path. All call sites (`s.populated =
  true` in MPC.fProxy.cs, `s.populated = false` in MPCBenchmark.fProxy.cs) are unchanged since a property
  assignment has the same syntax as a field assignment. See [[job-struct-copy-warmstate-audit]].

## fProxyLQRState.populated: native-backed so it survives an IJob by-value copy (warm-state fix)
- 2026-07-17 | Bug class: a warm-start flag mutated inside `IJob.Run()`/`Schedule()` is LOST because the
  job runs on a by-VALUE copy of the state struct (native BUFFERS survive — they're pointers — but plain
  fields don't). A worker `.Run()` of an LQR warm solve silently reset `populated`, forcing every warm
  call cold (or worse for LP's counters). Fix = the MPC `qpMeta` idea, but cleaner: `populated` moved
  behind a `NativeReference<int>` and re-exposed as a `bool` PROPERTY, so all call sites are unchanged and
  writes go through the shared handle — no rehydrate/copy-back (the flag is set once per solve, not in a
  hot loop; contrast LP's `etaCount`). `NativeReference` confirmed Burst-job-compatible (ControlLQRTests'
  TestJob runs the warm path through an IJob). Suite 6317/6317. See [[job-struct-copy-warmstate-audit]].

## Riccati (public DARE primitive)
- 2026-07-16 | Extracted the DARE engine out of the LQR facade into a new public
  `Riccati.dare(in A, in B, in Q, in R, ref S, maxIter)` (root `LinearAlgebra`, sibling of
  Eigen/SVD/Krylov). Was `Control.LQR.SDACore` (internal); LQR (control) and Kalman.steadyStateGain
  (estimation, via the Aᵀ/Hᵀ duality) BOTH consume it, so it belonged in a neutral primitive both
  depend DOWN onto -- this deletes the Kalman->Control.LQR reach entirely (Kalman lost its
  `using LinearAlgebra.Control;`). The shared hygiene kernels moved with it (Riccati.SymmetrizeInPlace,
  Riccati.FrobeniusNorm/FrobeniusNormDiff -- double-accumulate, deliberately NOT Norms.L2 which sums in
  fProxy -- Riccati.BlowupThreshold; consts SDA_MAX_ITER/BLOWUP_FACTOR now on Riccati.Info.cs). LQR
  keeps its control-specific mechanics (RiccatiStep = S->K gain kernel, RiccatiIterate warm recursion,
  lqr/lqrSchedule/lqg, fProxyLQRState, WARM_MAX_ITER) and calls Riccati.* for the shared bits; MPC's QP
  Hessian symmetrize now calls Riccati.SymmetrizeInPlace too.
- 2026-07-16 | DEDUP: `LQRInfo`/`LQRStatus`/`LQRStatusExtensions` DELETED, replaced everywhere by
  `RiccatiInfo`/`RiccatiStatus` (identical fields; the DARE result is the DARE result whether used for
  control or estimation). `rankDeficientControl` -> `rankDeficient` (generic: for the Kalman dual it is
  measurement-space, not "control"). LQR.lqr/lqrSchedule/lqg and Kalman.steadyStateGain now return
  RiccatiInfo; LQGInfo bundles two RiccatiInfo. Supersedes the "Control.LQR.SDACore" reach noted in the
  namespace entry below (same day).

## Control namespace (LQR / MPC)
- 2026-07-16 | Moved the control API out of `namespace LinearAlgebra` into a dedicated
  `namespace LinearAlgebra.Control` and renamed the LQR facade class `Control` -> `LQR` (the old
  `Control.lqr(...)` read confusingly next to the `LQ`/`LQRP` matrix decompositions). MPC + all
  companion types (`LQRInfo`/`LQRStatus`/`LQGInfo`/`fProxyLQRState`, `MPCInfo`/`MPCStatus`/
  `fProxyMPCState`) moved into the same sub-namespace. Kalman deliberately stayed in
  `LinearAlgebra` (user ruling); it reaches the internal Riccati helpers as `Control.LQR.SDACore`/
  `SymmetrizeInPlace`/`FrobeniusNorm` (internal = assembly-scoped, so cross-namespace is fine) and
  gained a file-level `using LinearAlgebra.Control;` because it NAMES `LQRInfo`/`LQRStatus` in code
  (`steadyStateGain`'s return type). Nested-namespace files still see every parent `LinearAlgebra`
  type (fProxyMxN/QP/CHOP/Blas/...) with no `using`, which is what kept the move low-risk — only
  external consumers (tests, benchmarks, demos) needed `using LinearAlgebra.Control;`. Method names
  unchanged (`LQR.lqr`/`lqrSchedule`/`lqg`, `MPC.solve`). Suite green post-regen.

## Control symmetric-GEMM reroutes (RiccatiStep / SDACore)
- 2026-07-13 | Riccati/SDA symmetric products moved to dotSym (missed by the first symmetric-GEMM
  pass, which covered QP/MPC only): RiccatiStep's Bᵀ(SB), Aᵀ(SA), BSAᵀK (= BSAᵀR̄⁻¹BSA) and
  SDACore's AkᵀX3 H-update. AkᵀX3 is symmetric only in exact arithmetic (X3 exits an LU solve);
  the mirror picks the upper triangle's roundoff where the full kernel produced O(eps) asymmetry
  that SymmetrizeInPlace then averaged — the existing post-add SymmetrizeInPlace is kept.
  SDACore's GkNext = (AkGk)·X2 does NOT fit either sym kernel form (neither operand of the
  symmetric product is materialized transposed) — left on the full kernel deliberately.

## Kalman / Kalman.UKF TransB + GEMM reroutes
- 2026-07-13 | predict: APAᵀ now Blas.dotSymT(AP, Aeff) — s.At no longer written by predict (still
  UpdateCore's (I-KH)P scratch). update: P·Hᵀ via dot(transposeB) (Ht temp deleted); K = Xtᵀ never
  materialized — K·y via vecMat dot(y, Xt), K·H and K·R via dot(Xt, ·, transposeA: true), IKHt temp
  deleted in favor of dotSymT((I-KH)P, IKH). ukfUpdate: same K elimination via dot(y, Pxzt).
- 2026-07-13 | UKF sigma recombinations GEMM-ified: predict's Σ Wc·d·dᵀ = (WD)ᵀ·D via dotSym
  (D overwrites Y, WD reuses X — both fully consumed by the propagation loop); update's
  Pzz/Pxz via dotSym(dZ, WdZ) + dot(dX, WdZ, transposeA) (dX overwrites X, dZ overwrites Z,
  one npts x m WdZ Temp, dz vector deleted). Results are bitwise different from the scalar
  rank-1 loops (different summation order), suite-validated.

## MPC / MPC.State
- 2026-07-12 | AUDIT POSTMORTEM (release-scan-2026-07-12/30-mpc-qpseam.md): confirmed HIGH --
  prestabilized input-bound rows read Phi/Gamma BLOCK k (x_{k+1}'s coefficients) instead of block
  k-1 (x_k's coefficients) when expressing u_k = -Kstab x_k + v_k, mis-constraining every stage's
  physical input and breaking the warm-start guess's feasible-by-construction property (the guess's
  own v_k = u_k + Kstab*x_k, evaluated with the CORRECT x_k, could not satisfy a row written against
  x_{k+1}). Root cause confirmed by direct read of MPC.State.fProxy.cs's row-assembly loop against
  the file's own Phi/Gamma block-k=x_{k+1} convention. FIX validated in a numpy prototype BEFORE
  editing the template (scratchpad/mpc-proto/mpc_prestab_bugfix.py): the audited off-by-one alone
  (block k -> block k-1, x_0=x0 identity for k=0) drove a deliberately-saturating case's u0 from
  -4.567 (outside [-2,2]) to -2.0 (respects the bound) -- but STILL disagreed with a fresh solve of
  the identical non-prestabilized problem by 0.198, which should be ~0 (prestabilization is a pure
  change of coordinates). Root-caused a SECOND, previously unaudited defect while chasing that
  residual: the condensed Hessian applied R naively to v (Rbar block-diagonal on v_k) instead of
  correctly expanding u_k^T R u_k with u_k = -Kstab x_k + v_k, silently dropping the -Kstab*x_k
  cross-coupling from the cost entirely. Fixed both together via one shared affine map, built once at
  construction and consumed by BOTH the rows and the cost so they cannot drift apart again: u_k =
  M_row_k @ V + c_k (c_k = -KPhiPre_row_k @ x0), M/KPhiPre built from block (k-1) (identity/Kstab
  directly for k=0); H_UU += M^T Rbar M (replacing the naive per-block R add for hasPrestab only);
  new persistent field Rcross = -2 M^T Rbar KPhiPre, applied as `c[0:nu] += Rcross @ x0` every solve
  call (MPC.fProxy.cs's BuildGradient). Extended prototype (mpc_prestab_full_fix.py) confirms the
  FULLY corrected version matches the non-prestabilized reference to ~1e-8 to ~3e-8 across binding,
  inactive, and random x0 -- vs 0.198-0.68 with only the row fix and no cost fix. Added
  PrestabBindingBoundMatchesNonPrestab to MPCTests.fProxy.cs (x0=(3,1.9), the SAME saturating case as
  SaturatedMatchesOracle) asserting both properties the coordinator specified: (i) u0 reconstructed
  independently from the state's own public Kstab/z fields respects the physical bound, (ii) matches
  a fresh non-prestabilized solve of the identical (A,B,Q,R,uLo,uHi) problem to tight tolerance --
  discriminates against BOTH the original off-by-one and the newly-found cost defect (verified by
  mentally/numerically reintroducing each).
- 2026-07-12 | AUDIT POSTMORTEM, low finding (same scan): MPC.solve's Fallback comment claimed
  state.z/wstatus are left untouched on "Infeasible/Unbounded", but QP.qpActiveSetCoreWarm only
  short-circuits before touching either on Infeasible -- on Unbounded (defensive-only, should not
  happen for MPC's genuinely PD H given R PD, but not structurally impossible) it runs the full
  active-set loop (which can mutate x = state.z via prior accepted steps) and unconditionally persists
  wstatus. Fixed by capturing u0out from the pre-solve warm-start guess BEFORE calling
  qpActiveSetCoreWarm (not re-derived from state.z afterward), so the "returns the shifted previous
  plan's first input" contract holds regardless of which failure status fires; state.uPlan/populated
  were already never written on this path (no change needed there). state.wstatus may still be
  perturbed on the (unreachable-in-practice) Unbounded path -- left as documented behavior rather than
  short-circuited in the QP seam itself, since RepairWorkingSet already re-validates every entry
  against the next frame's own state regardless, making a stale/perturbed persisted entry harmless.
  MPCStatus.Fallback's XML doc corrected to match (was overclaiming the same "both statuses" guarantee).
- 2026-07-12 | NEW feature: linear MPC over the standard batch/dense condensing (Borrelli-Bemporad-
  Morari, "Predictive Control for Linear and Hybrid Systems", ch. 2). acados/HPIPM (BSD-2) and TinyMPC
  (MIT) condensing routines were read for PRODUCT SHAPE reference only (decision-vector layout, the
  general idea of a fixed-at-construction condensed Hessian) -- no source line from either was
  transcribed; the actual Phi/Gamma/H assembly here is an original derivation verified against a from-
  scratch numpy/scipy prototype (scratchpad, not shipped) before this file was written. Soft-row exact
  penalty follows Kerrigan & Maciejowski, "Soft Constraints and Exact Penalty Functions in Model
  Predictive Control" (2000). qpOASES's MANUAL/thesis (warm-start strategy framing, Ferreau/Bock/Diehl
  2008) was read; qpOASES's SOURCE (LGPL) was not. DAQP (MIT) was read for active-set warm-start
  mechanics only.
- 2026-07-12 | Validation (numpy/scipy prototype, double integrator A=[[1,1],[0,1]], B=[[0],[1]],
  Q=I2, R=1 throughout): (a) unconstrained condensed MPC's u0 matched Control-style infinite-horizon
  LQR to ~1e-13 (double) / ~1.6e-7 (float32) across N in {1,3,10,30} -- a stationary DARE terminal cost
  makes ANY horizon reproduce the infinite-horizon law exactly, the correctness anchor. (b) input-
  saturated case matched scipy.optimize.minimize(method='trust-constr') on the identical condensed QP
  to ~1.6e-5 (its own convergence floor), independently cross-checked against a 3^n box-active-set
  brute-force enumeration. (c) soft wall: inactive case matched the unconstrained solution to ~5e-10;
  active-but-avoidable (input saturates but the wall itself is never touched) matched a hard-constrained
  trust-constr solve to ~1.6e-7 with zero slack, INSENSITIVE to rho1 across [0.5, 200] (all agreed to
  ~1e-7) -- the library's chosen default (rho1=1e3) sits well inside this margin; active-and-unavoidable
  (a double integrator's control has a one-step lag onto position, so the FIRST predicted stage's
  position is fixed by x0 alone) reproduced a hand-derived minimal-violation closed form exactly
  (0.3 then 0.6 over two stages). (d) receding-horizon active-set churn: [3,3,3,2,1,0,0,...,0] over 40
  frames -- collapses to 0 after frame 5, matching the "0-3 after the first" expectation. (f)
  prestabilization: rho(A)=1.2, N=40 raw condensing reached cond(H)~2.4e9 (float32-risky, though not yet
  NaN/inf) vs prestabilized cond(H_cl)~3.2 -- confirms the conditioning-insurance framing, not a
  strict correctness requirement at this rho/N.
- 2026-07-12 | Prestabilization (u_k = -Kstab x_k + v_k, condense the closed loop A-B*Kstab) turns hard
  input bounds into GENERAL rows (2*N*m of them) instead of a box on the decision vector, since u_k's
  bound becomes state-dependent (state depends on v through Gamma_cl) -- verified analytically that
  forward-simulating the warm-start guess with the REAL (A,B) and u_k, then deriving v_k = u_k +
  Kstab@x_k from that SAME trajectory, reproduces exactly the closed-loop condensing's own implied
  trajectory (x_{k+1} = A x_k + B u_k = (A-B Kstab) x_k + B v_k by construction). Combining
  prestabilization with the deltaU penalty is NOT supported in v1 (deltaU would need to couple to the
  state through the SAME substitution, compounding both derivations) -- throws at construction rather
  than silently dropping one feature. QR up/downdate for the per-iteration re-factorization was
  evidence-gated OUT of scope per the task brief; qpActiveSetCoreWarm re-factorizes the working set from
  scratch every pivot, same as qpActiveSetCore, fine warm at the target sizes (d <= 160).
- 2026-07-12 | Constructor overload ladder: deltaU-only and prestabilization-only convenience overloads
  were NOT added -- both would need an extra fProxyMxN-typed parameter (S / Kstab) in the exact same
  position as the "explicit terminal P" overload's own P parameter, a genuine C# overload-signature
  collision (parameter names never participate in overload resolution). Verified this is a real
  constructor-only distinction (methods can't disambiguate on names) via a standalone dotnet repro
  before writing the constructor ladder. Reach the full (17-parameter) constructor directly for those
  two features, passing `default` for the unused optional matrix params.
- 2026-07-12 | H (the condensed QP Hessian) is explicitly re-symmetrized via Control.SymmetrizeInPlace
  (reused directly, not reimplemented) after assembly -- Gamma^T Qbar Gamma accumulates through
  Blas.dot's own summation order, which can leave a tiny roundoff asymmetry even though the true
  mathematical result is exactly symmetric whenever Q/R/P/S are.

## Control
- 2026-07-12 | lqg() added: convenience solving BOTH the LQR control DARE (existing lqr) and the
  KF filter DARE (new Kalman.steadyStateGain) from the same A, returning a thin LQGInfo pair. Zero
  new Riccati math -- both calls reuse Control.SDACore, the filter side via the LQR/KF duality
  mapping (Kalman.fProxy.cs's file header). SymmetrizeInPlace widened private -> internal (no
  behavior change) so Kalman's PredictCovarianceCore/UpdateCore reuse the exact same
  symmetrize-after-roundoff hygiene instead of a second copy of the loop.

- 2026-07-11 | SDA recurrences implemented (Chiang-Fan-Lin Algorithm 2.1, no-cross-term/nonsingular-R case): A0=A, G0=BR⁻¹Bᵀ, H0=Q, A_{k+1}=Ak(I+GkHk)⁻¹Ak, G_{k+1}=Gk+AkGk(I+HkGk)⁻¹Akᵀ, H_{k+1}=Hk+Akᵀ(I+HkGk)⁻¹HkAk, Hk→S. The (I+GH)/(I+HG) solves are nonsymmetric n×n via LU (compact in-place + multi-RHS decompSolve), not Cholesky; G0=BR⁻¹Bᵀ is built via CHOP on R (not a bare inverse) so a semidefinite R degrades gracefully there too. (was Control.fProxy.cs:10-40, :164)

## Kalman
- 2026-07-12 | Release-scan perf fix: UpdateCore's Xt = Smeas^-1 * (H P) recomputed H*P via a
  fresh GEMM even though PHt = P Hᵀ (already computed for Smeas, still live) equals (H P)ᵀ
  exactly since P is symmetric at every call site. Xt is now Blas.trans(in PHt, ref Xt) --
  O(m·n) instead of O(m·n²), same result.
- 2026-07-12 | Bug found by the test suite (float only): SteadyStateGainVsOracle got a Kss ~98%
  relatively wrong (0.9765909 vs the 2e-3 float tolerance); FixedPathMatchesConverged missed its
  tracking bound by 0.109 downstream of the same bad gain. Root-caused with a float32 numpy harness
  transliterating Control.SDACore and the test's own OracleGain literally on the test's exact CV
  system (A=[[1,1],[0,1]], H=[[1,0]], Q=diag(1e-4,1e-4), R=[[0.05]]): SDACore's convergence test
  (residual = diffNorm / max(1.0, ‖Hk‖)) reported Converged after ONE doubling step in float
  (residual 2.644e-4, just under Consts.floatSqrtEps=3.4527e-4) while the true fixed point needs
  ~8 steps (confirmed independently by both the double-precision SDA run and the test's own
  fixed-point oracle, which agree with each other to ~1e-16/2e-7). The `max(1.0, ...)` floor is a
  reasonable absolute backstop for LQR's typically-O(1) cost weights, but Kalman process/
  measurement covariances are routinely << 1 (here ‖Q‖+‖R‖ ~ 0.05), so the floor turns the
  RELATIVE tolerance into an ABSOLUTE one at roughly the SAME scale as the quantities being
  tracked -- one tiny absolute step off Sigma0=Q satisfies it immediately, before the recursion
  has moved at all. Fixed in steadyStateGain (not in Control.SDACore itself, to avoid touching the
  shared LQR cold-solve path and its own test suite): jointly rescale Q/R by
  1/max(‖Q‖+‖R‖, Consts.fProxyZeroThreshold) before the SDA call and unscale Sigma after -- proven
  exactly invariant for Kss (scaling Q and R by the same c scales Sigma by c, leaving
  Sigma Hᵀ(H Sigma Hᵀ+R)⁻¹ unchanged), confirmed in the float32 harness: relImplCorrect
  0.9765909 -> 3.385e-7, iterations 1 -> 6, while the wrong-orientation discrimination margin
  (relOraclePair/relImplWrong ~3.57) is untouched. Also confirmed harmless for double (already
  exact, scaling doesn't change the converged answer, iteration count unchanged at 8).
  Control.FrobeniusNorm widened private -> internal for this (no behavior change).
- 2026-07-12 | NEW feature. Algorithm reference: FilterPy (rlabbe/filterpy, MIT) -- predict/update
  equations (x=Ax+Bu, P=APAᵀ+Q; y=z-Hx, S=HPHᵀ+R, K=PHᵀS⁻¹, x+=Ky, Joseph-form
  P=(I-KH)P(I-KH)ᵀ+KRKᵀ) fetched and verified line-by-line against kalman_filter.py/EKF.py.
  Interface-shape reference: mherb/kalman (MIT) -- separate propagation/measurement function plus
  a separate Jacobian, not one fused updateJacobians() call. FORBIDDEN sources (per owner ruling,
  not used): MathNet.Filtering Kalman (LGPL despite MIT-labeled repo), TinyEKF historical
  snapshots (LGPL then).
- 2026-07-12 | K is never formed via an explicit S inverse anywhere in this file: every gain
  computation (UpdateCore, steadyStateGain) solves the TRANSPOSED system S·Kᵀ = (PHᵀ)ᵀ = HP via
  CHOP (pivoted Cholesky), so a rank-deficient S degrades to a minimum-norm K instead of a hard
  failure or a divide-by-near-zero.
- 2026-07-12 | steadyStateGain's SDA-duality mapping (Ã=Aᵀ, B̃=Hᵀ, S↔Σ) was validated against an
  INDEPENDENT ground truth before this file was written (Python prototype, plain fixed-point
  iteration of the KF predicted-covariance Riccati equation from Σ0=Q, no SDA/doubling involved):
  agreement to ~1e-16 relative Frobenius norm on a 2-state CV tracker, AND against a THIRD
  independent path (the actual predict/update Joseph-form recursion iterated to steady state,
  gain extracted from its last update call) to ~1e-16. A deliberately-wrong mapping (forgetting
  the A transpose, i.e. Ã=A instead of Aᵀ) was also run and diverges from ground truth by ~1e-2
  relative -- confirms the test is actually discriminating, not passing by coincidence.
- 2026-07-12 | EKF interface choice: analytic Jacobian REQUIRED on IfProxyKFModel/
  IfProxyKFMeasurement (JacobianF/JacobianH), no numeric-differentiation fallback baked into the
  interface itself. A wrapper-functor design (an fProxyNumericKFModel<TInner> auto-computing the
  Jacobian for a Jacobian-less inner model) was considered and rejected: it needs a nested generic
  struct implementing IfProxyKFModel while itself being generic over another IfProxyKFModel-minus-
  Jacobian shape, which has no precedent elsewhere in this codebase's struct-functor family
  (IfProxyLinearOperator's wrappers like fProxyColScaledOperator wrap ONE inner operator of the
  SAME interface, not a different, smaller interface) and adds a layer of generic indirection for
  a case (no analytic Jacobian available) that is the exception, not the rule. Shipped instead:
  Kalman.numericJacobianF/numericJacobianH, plain central-difference helpers a user calls FROM
  INSIDE their own JacobianF/JacobianH when hand-differentiating is impractical -- same
  "provide the primitive, not a forced wrapper" shape as QRCP's tol3z reuse of Consts.fProxySqrtEps.
- 2026-07-12 | fProxyKFState's own scratch is genuinely zero-Allocator.Temp for predict/ekfPredict/
  predictFixed/updateFixed (every intermediate is a pre-allocated field, sized once at n or n x n
  at construction). The general update()/ekfUpdate<TMeas> path does NOT extend this to its
  measurement-shaped intermediates (Hᵀ, PHᵀ, S, the CHOP factor, K) -- these are per-call
  Allocator.Temp, sized to that call's actual H.M_Rows, deliberately mirroring
  Control.RiccatiStep's own R+BᵀSB solve (also per-call Temp, also variably shaped). Considered and
  rejected: pre-allocating update()'s scratch at fProxyKFState.MMax and reinterpreting a smaller
  logical sub-block of it per call -- the library's dot/CHOP primitives all validate EXACT
  dimension equality (no stride/logical-sub-size concept anywhere), so this would need either
  mutating fProxyMxN.M_Rows/N_Cols post-construction (undocumented elsewhere, and fProxyMxN.Length
  is a readonly field that would then disagree with M_Rows*N_Cols) or a raw NativeArray-view
  reinterpretation via NativeArrayUnsafeUtility (safety-handle bookkeeping for no proven benefit --
  a per-call CHOP factorization is O(m³), already far more expensive than one Temp bump-allocator
  vector/matrix allocation). MMax is used only by the fixed-gain fast path (predictFixed/
  updateFixed), which genuinely needs and gets zero-Temp-alloc treatment since Kss is fixed-shape
  for the state's whole lifetime.

## Kalman.UKF
- 2026-07-12 | NEW feature (UKF, next increment after the linear/EKF Kalman filter). Algorithm
  reference: FilterPy (rlabbe/filterpy, MIT) -- MerweScaledSigmaPoints.sigma_points/_compute_weights
  and UnscentedKalmanFilter.predict/update, fetched and verified line-by-line. ukfPredict/ukfUpdate
  reuse the SAME IfProxyKFModel/IfProxyKFMeasurement functors ekfPredict/ekfUpdate use, calling ONLY
  F/H -- JacobianF/JacobianH are never read, which is the whole point of the unscented transform
  (no linearization at all, not even an approximate one).
- 2026-07-12 | FLOAT-RISK FINDING, deviation from FilterPy's cited default: Van der Merwe's classic
  write-up (and FilterPy's docstring) recommends alpha ~1e-3. Measured in the float32 numpy
  prototype (CV tracker, UKF vs the exact linear-KF oracle -- sigma points are exact for a LINEAR
  F/H, the strongest correctness check available): alpha=1e-3 gives max|x diff|=0.86 (catastrophic
  -- worse than useless) and max|P diff|=5.8, vs alpha=1.0 giving 1.9e-6/2.0e-6 (both essentially at
  the float32 precision floor for this problem's scale). Root cause: n+lambda = alpha²(n+kappa)
  shrinks the sigma-point spread by alpha while lambda/(n+lambda) (and every other weight, which is
  ∝ 1/(n+lambda)) grows by roughly 1/alpha² -- at alpha=1e-3 the weights reach ~±1e6 (see the
  concrete numbers in the fProxyUKFCache DEVLOG entry) and the covariance recombination becomes a
  weighted sum of near-identical numbers with huge opposite-signed weights, i.e. textbook
  catastrophic cancellation. This library's DEFAULT is alpha=1, beta=2, kappa=0 instead --
  confirmed (same harness) that UKF then tracks a nonlinear pendulum AS WELL AS OR BETTER than EKF
  in both precisions (double: EKF 0.00718 vs UKF 0.00713; float: EKF 0.00718 vs UKF 0.00579 --
  UKF actually wins in float, matching the "UKF should track as well or better" acceptance bar).
  Double precision also improves under the new default (3.6e-15 vs 4.7e-9 relative agreement with
  the linear-KF oracle at alpha=1e-3), so this is not purely a float32-only trade. A caller can
  still construct <see cref="fProxyUKFCache"/> with an explicit smaller alpha via the 4-arg
  constructor; the algorithm remains correct there (validated: alpha=0.1 and 0.05 both keep P
  exactly symmetric and PSD, min eigenvalue ~4e-4, over 2000 steps in both precisions despite
  Wc[0] reaching -96 / -396) -- just with markedly less numerical margin, which is now a documented,
  deliberate caller choice rather than a silent trap.
- 2026-07-12 | GenerateSigmaPoints regenerates sigma points FRESH at the start of BOTH ukfPredict
  and ukfUpdate -- a deliberate deviation from FilterPy's UnscentedKalmanFilter, which reuses
  predict()'s propagated `sigmas_f` directly inside update() (a documented perf shortcut in
  FilterPy's own code, not part of Van der Merwe's original algorithm). Reasoning: this library's
  own Kalman.update already supports being called more than once per predict (multi-sensor fusion
  between predicts); reusing stale sigma points across a second ukfUpdate call in the same pattern
  would silently under-represent the covariance change the first update just made. Regenerating
  costs one extra O(n³) Cholesky per ukfUpdate call, and is mathematically IDENTICAL to FilterPy's
  result in the common case (update immediately follows predict, nothing else in between).
- 2026-07-12 | Permutation-aware sigma-point scatter: CHOP factors Pᵀ·Σ·P = L·Lᵀ (P the pivot
  permutation, Σ the state covariance -- disambiguated from CHOP's own P-for-permutation in
  comments as "the permutation"), so L's COLUMNS are in PIVOTED order, not the original state-index
  order. The Van der Merwe spread vector for column k is therefore built by SCATTERING L's column
  through the permutation (v[Piv[i]] = L[i,k]·scale), not read off directly -- verified in the
  Python prototype's own pivoted-Cholesky emulation (which deliberately pivots, unlike numpy's
  plain `cholesky`, specifically to exercise this scatter logic before it was ported to C#).
  Getting this backwards (reading L[k,i] or skipping the permutation) would silently produce a
  valid-LOOKING but WRONG sigma spread for any P that actually pivots (i.e. essentially always,
  since CHOP pivots greedily by largest remaining diagonal even for well-conditioned input).

## Kalman.UKFCache
- 2026-07-12 | Chose a SEPARATE fProxyUKFCache over folding sigma-point buffers into fProxyKFState
  (the spec's other offered option): keeps the linear/EKF/fixed-gain paths (which never need sigma
  points) free of (2n+1)-sized memory, mirrors the house Cache convention (fProxyCHOPCache,
  fProxySVDThinCache) of a workspace struct paired with -- not merged into -- the data it operates
  on, and lets a caller reconfigure alpha/beta/kappa (a UKF-only concept) without touching
  fProxyKFState's own constructor arity.
- 2026-07-12 | Nests CHOP's own fProxyCHOPCache (`chopWs`) rather than calling CHOP.decomp's
  convenience (non-workspace) overload, which allocates an n x n Allocator.Temp buffer internally
  every call -- caught by re-reading CHOP.decomp's own source after first wiring GenerateSigmaPoints
  to the convenience overload, which would have silently broken the "ukfPredict is zero-Temp-alloc"
  claim. `bt` (CHOP's solve-side scratch) is deliberately left uncreated -- sigma-point generation
  only ever calls `decomp`, never `decompSolve`.
- 2026-07-12 | See the concrete alpha=1e-3 default-negative-Wc[0] numbers this defaults choice
  avoids: n=2, alpha=1e-3, beta=2, kappa=0 gives lambda=-1.999998, Wm[0]=Wc[0]≈-1e6, every other
  weight ≈+2.5e5 (computed in the float32 prototype). alpha=1 (this library's default) instead
  gives lambda=0=kappa, Wm[0]=0, Wc[0]=2 -- non-negative for the default case, though a caller-
  chosen alpha&lt;1 can still drive Wc[0] negative (by design; see Kalman.UKF's own DEVLOG entry).

## Kalman.State
- 2026-07-12 | Scratch fields (xNext/Bu/AP/APAt/At/J/yFast) are `public`, not `internal`, matching
  the house Cache/State convention (fProxyCHOPCache, fProxyLQRState both use public fields) rather
  than hiding them -- these are workspace buffers, not encapsulated implementation state.

## Kalman.Info
- 2026-07-12 | KFStatus has only two members (Ok / InnovationSolveFailed) because CHOP.decomp on
  the innovation covariance S = HPHᵀ+R has only two outcomes worth distinguishing here:
  Success/RankDeficient (both usable -- S is generically PSD whenever P is, so RankDeficient is
  expected on a redundant/collinear sensor row, not an error) collapse to Ok, and Indefinite (S
  numerically broken) is the only real failure.

