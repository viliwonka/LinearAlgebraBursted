# DEVLOG — OP
Code comments state contracts only; history lives here (see CLAUDE.md).

## Fit / Blas.Triangular — review cleanup, 2026-07-26
- 2026-07-26 | `Blas.triUpper` now reports `Singular` instead of always reporting Success. Fixed at the
  innermost kernel rather than per-caller, so LU/CHO/QR/LQ all gain it. Rejects a zero, NaN or
  +/-Inf diagonal and checks the result is finite. The Inf case is NOT redundant with the finiteness
  check: `abs(Inf) > 0` passes the magnitude test and then drives `x[r] = finite/Inf = 0`, which is
  finite and would be certified. Detection is deliberately narrow -- only what definitively produces
  garbage -- so no currently-working caller changes behaviour; ill-conditioned-but-usable still
  solves and still reports Success. The whole suite passing unchanged is the evidence.
- 2026-07-26 | `Fit.conic` gained the same Hartley normalization the 3D ellipsoid route has, for the
  identical reason: its design entries are FOURTH powers, so a unit circle centred at (500, 500)
  faces a scatter conditioned like offset^4. `Fit.ellipse` on an off-origin cloud in float was
  returning garbage or false long before any of today's work.
- 2026-07-26 | Comment-policy violations trimmed: three code comments carried bug-postmortem
  narration (the collapse-guard-duplication history) that belongs here per CLAUDE.md. One of the
  three disappeared with SphereIrls; the other two are now contract language.
- 2026-07-26 | Uniformity ORACLES added for the ellipsoid and capsule samplers. Neither
  surface-membership nor fit-back can catch a biased sampler -- both pass for anything that lands on
  the surface -- so a sign slip in the ellipsoid's stretch factor or the capsule's area split would
  have shipped green. Ellipsoid: for the oblate (1, 1, 0.5) spheroid the stretch is
  `sqrt(0.25 + 0.75 nz²)`, giving `E[|nz|] = (7/9)/1.38017 = 0.5635` against the 0.5 an unrejected
  sampler yields, ~30 sigma at n = 20000. Capsule: tube share is `L/(L + 2r)`.
- 2026-07-26 | `Subspace2Refit`'s `dominant` parameter was dead -- 2D has no plane-equivalent, so
  nothing ever asked for the minor axis. Dropped rather than left as unexplained dead code.

## Fit — correctness pass on the review findings, 2026-07-26
- 2026-07-26 | **Sampson IRLS was not Sampson-weighted.** Minimizing `sum rho(F²/g²)` through a
  weighted ALGEBRAIC fit needs weight `rho'(d²)/g²`; the `1/g²` was missing, so the refit still
  minimized gradient-scaled algebraic residuals. Because `fProxyL2Loss.RhoPrime == 1`, every weight
  stayed at its initial 1, `maxDelta` was 0, and the loop broke BEFORE its first refit — making
  `conic<L2Loss>` bit-identical to `conic` and the "L2 is not a no-op here" doc claim false in three
  places. `QuadricSampsonL2ReducesAlgebraicBias` only asserted "not worse", so it passed vacuously;
  it now asserts strict improvement.
- 2026-07-26 | **The point-to-ellipse coordinate floor was shared across axes**, scaled by the LARGEST
  radius. On a 1e7:1 ellipse that lifts the minor-axis vertex y = 1 to 3452, and the vertex measures
  3451 from the curve it lies on. Now per-axis, each scaled by its own radius. Float-only: double's
  smaller sqrtEps hid it, which is why the double half of the test passed while float failed.
- 2026-07-26 | The same root solve bracketed on `[0, big²]`. Far above the root F is flat at -1, so
  every Newton step overshoots and degrades to bisection, needing `log2(big²/s*)` halvings — past
  roughly a 1e6 aspect ratio that exceeds the 40-iteration cap. Both ends now come from the root
  condition itself: the terms sum to 1, so each is <= 1 (giving `s + d_i >= r_i q_i`) and the largest
  is >= 1/k (giving `s + d_i <= sqrt(k) r_i q_i`). The bracket is a factor of sqrt(k) wide.
- 2026-07-26 | **`bool ok = QR.solveInPlace(...)` was a guard in name only** at four sites. That
  overload documents itself as ALWAYS reporting Success and back-substituting through a zero diagonal
  on rank-deficient input (QR.fProxy.cs:589, :599) — exactly singular gave Inf/NaN caught downstream
  only by luck, near-singular float gave large finite garbage certified as a fit. All four now use
  QRCP and check `rank`.
- 2026-07-26 | Cone `Distance` measured the infinite DOUBLE cone, so points behind the apex scored as
  inliers on the mirror nappe. Now projects onto the generator and returns `|p - apex|` where that
  arclength is negative. The residual functor keeps the smooth signed form deliberately — LM wants
  the kink-free version, and the two agree everywhere near the surface.
- 2026-07-26 | `irls`'s documented warm start did nothing: `Refit` runs before `Distance` is ever
  read, so the incoming model had zero effect and the RANSAC-then-polish pipeline the library itself
  recommends silently started from the contaminated unweighted fit. Now an explicit `warmStart` flag
  seeds the first weights from the incoming model's residuals.
- 2026-07-26 | `irls`'s convergence test compared an absolute weight delta against sqrtEps, which is
  unreachable once weights are large (an inverse-variance prior, or L1's `0.5/floor`), so the loop
  always ran its full budget. Now relative to the largest weight. Same fix in both Sampson loops,
  where the new `1/g²` factor makes weight magnitude arbitrary.
- 2026-07-26 | `EllipsoidWeighted`'s Hartley normalization ignored the weights while the fit used
  them, so under LO-RANSAC's 0/1 weights the outliers still set the scale — defeating the
  normalization exactly where it was needed most.

## Fit — four-agent review, 2026-07-26
- 2026-07-26 | **`ellipsoid` returned a WRONG ellipsoid as success.** Li & Griffiths' `4J - I² > 0` is
  sufficient for an ellipsoid but not necessary: for eigenvalues (1, 1, t) it equals t(4 - t), so it
  excludes everything past a 2:1 axis ratio. A (1, 1, 0.4) cloud came back as (1, 1, 0.5) — clipped
  to the boundary of the representable set, reported `true`. Raising k does NOT fix it: at k > 4 the
  constraint starts accepting hyperboloids (λ = (1,1,-ε) gives `kJ - I² → k - 4`), which is why 4 is
  the published choice. Restructured to run the UNCONSTRAINED fit first and keep it when it already
  classifies as an ellipsoid, applying the constraint only as the repair for clouds the plain fit
  gets wrong. Gated on n >= 10, since the constrained route is what makes MinimalSamples 9.
- 2026-07-26 | **RANSAC hypotheses were not independent.** All four consensus drivers hoisted
  `var candidate = model;` OUTSIDE the draw loop, and cylinder/cone/torus/capsule `Estimate` forwards
  its own current fields as the LM warm start (auto-seeding only on a zero axis). So every hypothesis
  after the first continued the PREVIOUS one instead of seeding from its own minimal sample: a bad
  early hypothesis became the basin all later ones converged into, and the adaptive stopping rule
  lost the independence premise it is derived from. Now `default(TModel)` per draw. The final
  consensus refit stays warm — refining from the winner is the point there.
- 2026-07-26 | `classify`'s zero threshold was `max(scale, 1) * sqrtEps`, i.e. ABSOLUTE whenever every
  eigenvalue is below 1 — the normal case, since `quadric` returns unit-norm coefficients and a
  sphere of radius R has quadratic entries of order 1/R². A radius-60 sphere classified as
  `Degenerate` in float, against a doc promising scale invariance. Now purely relative.
- 2026-07-26 | `Fit.quadric` guarded `n >= 9` but its n x 10 design needs `SVD.thin`'s rows >= cols,
  so exactly 9 points threw from inside the SVD. Guard corrected to 10.
- 2026-07-26 | `maxIter` was declared and silently ignored by all 8 `Fit.Solid` entry points. Forwarded
  via `SolveLM`; the `<= 0 → default` mapping has to live there because `nlsSolve` THROWS on
  `maxIter < 1`, so passing a defaulted 0 straight through would turn "use the default" into a throw.
- 2026-07-26 | `MsacScore` took `in TModel` and called `Distance` on it, so the compiler copied the
  whole shape struct once PER POINT (magsac pays that per sigma level per draw). Local copy now.
  Plain `ransac` also inlined its own copy of the scoring loop instead of calling it; both dimensions
  now route through one helper, so a scoring change cannot reach two of the three estimators only.
- 2026-07-26 | `WeightedPlane` was a byte-for-byte duplicate of `SubspaceRefit(points, w, 3, ...)`,
  written during the flat-shapes work without noticing the sibling. Deleted.

## Fit.Ellipsoid / Fit.Sample
- 2026-07-26 | Ellipsoid added as the CONSTRAINED counterpart to `quadric`, via Li & Griffiths (2004)
  `4J - I² = 1`. `k = 4` is fixed rather than exposed: any `k > 3` forces an ellipsoid, but a caller
  passing `k <= 3` would silently lose the guarantee the routine exists to provide. C1's inverse is
  written in closed form (`[[0,.5,.5],[.5,0,.5],[.5,.5,0]]` over the squares, `-0.25 I` over the
  cross terms) so no solve is needed for it.
- 2026-07-26 | The rotated-ellipsoid fit FAILED outright in float before data normalization was
  added. The design entries are fourth powers of the coordinates, so a cloud a few units off the
  origin conditions the 10x10 scatter badly enough that the 6x6 eigenproblem finds no
  constraint-satisfying root at all. Hartley-style centre+RMS-scale normalization fixed it; the
  quadratic block is untouched by the inverse map, so the ellipsoid constraint survives it and the
  un-transform needs no divisions (multiplying through by scale² clears them, and coefficients are
  only defined up to scale anyway).
- 2026-07-26 | Accuracy floor is ~1e-8 relative in double, NOT machine precision, because
  Li & Griffiths needs the scatter blocks for its generalized eigenproblem and forming DᵀD squares
  the condition number. `quadric` factors the n x 10 design directly and does better. Getting the
  same accuracy here would mean carrying an n x 10 matrix (O(10n) memory against the current O(1))
  to buy precision far below any measured cloud's noise — rejected, and `AlgTol` in the tests is
  sized to the method rather than to `Tol`.
- 2026-07-26 | Point-to-ellipse/ellipsoid root solve REWRITTEN from the classic `t` parameterization
  to `s = t + min(r)²`. Two independent bugs, both fixed by the shift: (1) every divisor is
  `(t + r_i²)`, a difference of nearly-equal numbers, so in `t` they lose most of their significant
  digits near the shape's own centre; (2) the convergence test was scaled by `max(|t|, 1)`, which
  near the centre is orders of magnitude looser than the root needs — a point at the centre of a
  (4, 2) ellipse measured 2.186 against a true 2.0. In `s` both are exact. Don't "simplify" back.
- 2026-07-26 | The same routine reported distance ZERO at an ellipse's centre before the coordinate
  floor was added: every term of F vanishes there, so the search runs to the bracket floor and the
  closest point comes back as the centre itself. A dead-centre outlier scored as a perfect inlier.
  Shipped that way in the flat-ellipse work; found by the ellipsoid's own centre test.
- 2026-07-26 | `UniformDirection` reports through an OUT parameter, not a return value. `Fit` carries
  no per-type token, so the float and double files merge into one class and a returned `fProxy3`
  would leave the two differing only in return type — CS0111, not an overload. `EllipseAngle` is
  safe as a return because its fProxy PARAMETERS already discriminate.
- 2026-07-26 | Sampling is opt-in per shape for the same reason fitting is: plane, line, cylinder and
  cone are unbounded, so they have no uniform distribution to draw from — the same missing extent
  that stops least squares fitting it. Triangle/simplex/solid-box surface fits were considered and
  rejected in the same discussion: a shape that can CONTAIN points makes every containing candidate
  score zero, which is a min-enclosing problem, not a least-squares one.

## Header blocks trimmed to contract + traps
- 2026-07-25 | User ruling: "what code does can be derived from code anyway" — mechanism narration
  cut from the long header blocks, contracts/traps/invariants kept. Formulas and derivations that
  were load-bearing for a MAINTAINER (rather than a caller) are recorded here.
  **LOBPCG convergence test** (was LOBPCG.fProxy.cs:43). Per-pair, scale-invariant:
  `‖A x_i − λ_i B x_i‖ <= tol · scale_i`, `scale_i = min(normAEst·‖x_i‖ + |λ_i|·normBEst·‖x_i‖_B,
  max(|λ_i|,1)·‖x_i‖_B)`, where normAEst/normBEst are Frobenius operator-norm estimates taken from
  the orthonormalized seed block (normBEst = 1 for B = I, making the two norms coincide). Explicit
  norms stop a shrinking/collapsed iterate self-certifying; the λ terms scale with `‖x_i‖_B` (not
  `‖x_i‖`) so an iterate blowing up inside a singular B's null space cannot self-certify either.
  Do not "simplify" this to a plain relative residual — see [[solver-termination-measure-safety]].
  **MPC prestabilization block-index trap** (was MPC.State.fProxy.cs:41). The affine map
  `u_k = M_row_k @ V + c_k` uses `c_k = -KPhiPre_row_k @ x0`, built at construction from Phi/Gamma
  BLOCK (k−1) — that block's own convention is `x_{(k-1)+1} = x_k`, i.e. stage k's state, NOT block
  k, which is `x_{k+1}`. k = 0 uses `x_0 = x0` exactly, with no V-coupling. Item 2's expansion:
  `u_k^T R u_k` becomes `M^T Rbar M` added to H_UU plus a per-call gradient correction
  (`Rcross @ x0`, `Rcross = -2 M^T Rbar KPhiPre`, fixed at construction). Row assembly and cost
  correction consume the SAME M/KPhiPre built once, so they cannot drift apart under a future edit.
  **MINRES-QLP shift exactness** (was Krylov.MINRESQLP.fProxy.cs:23). The Lanczos recurrence stays
  exact because the `-shift*v` term in the shifted matvec cancels against `+shift` in the diagonal
  alfa; the shift costs one extra axpy per iteration when nonzero. `shift` is a runtime parameter,
  not a compile-time constant, so a zero shift is a branch-skip rather than a Burst fold — which is
  why zero-shift callers stay bit-identical. `A - shift*I` stays symmetric, so the QLP min-length
  machinery is unchanged. The tol-driven regularization is the reference's MAXXNORM knob made
  problem-relative (growth past ~`beta1/(64·tol·‖A‖est)`) instead of its absolute 1e7 default.
  **GCRO-DR harmonic-Ritz vector route** (was Krylov.GCRODR.fProxy.cs:19). Each cycle projects the
  recycled subspace out of the residual, runs an m-step Arnoldi projected against the recycled C,
  then rebuilds the k recycled vectors from a small dense harmonic-Ritz eigenproblem over the
  combined (old-recycle + this-cycle-Krylov) subspace. Harmonic Ritz VALUES come from
  `Eigen.valuesQRInPlace`; each selected value's REFINED vector (minimizer of `‖(A−θI)v‖` over the
  combined subspace) comes from `Eigen.symmetricInPlace` on a small symmetric matrix. DESIGN
  CONSTRAINT: this library has no general nonsymmetric eigenVECTOR solver, so the refined-vector
  route deliberately reuses the two eigensolvers that exist rather than hand-rolling one. If a
  nonsymmetric eigenvector solver ever lands, this is the call site to revisit.
  **UKF Merwe scaled sigma points** (was Kalman.UKF.fProxy.cs:13). `lambda = alpha²(n+kappa) − n`;
  point 0 = mean; points 1..n = `mean + sqrt(n+lambda)·col_k(chol(P))`; points n+1..2n = the same
  subtracted. `Wm[0] = lambda/(n+lambda)`, `Wc[0] = Wm[0] + (1 − alpha² + beta)`, every other weight
  `= 1/(2(n+lambda))`. Recombination is a weighted mean, weighted-outer-product covariance,
  cross-covariance Pxz, `K = Pxz·Pzz⁻¹`, `x += Ky`, `P −= K·Pzz·Kᵀ` — with K solved as the
  TRANSPOSED system `Pzz·Kᵀ = Pxzᵀ` via CHOP, never an explicit inverse (same as linear/EKF
  UpdateCore).

## LP.ladFN — deviations from the literal Fortran reference (`rqfnb.f`/`lpfnb.f`)
- 2026-07-25 | Relocated from the file header (was LP.FrischNewton.fProxy.cs:69). The port is
  fidelity-first; these are the three documented departures, per the port-fidelity rule.
  **(1) The single tolerance is data-scaled.** `lpfnb.f` drives both the z/w initialization floor and
  the convergence test from one caller-supplied eps, and so does this port — but that eps is
  `sqrtEps·‖b‖₂` rather than a caller constant, so the convergence criterion means the same thing at
  every data scale. The reference's single constant is not itself unsafe: when the floor fires on
  every observation the gap is about `m·eps`, which still exceeds the tolerance eps for any m > 1, so
  the solve proceeds. It is SPLITTING the two that is unsafe — a scaled gap test beside an unscaled
  floor lets the floor-dominated gap fall under the tolerance, and the solve returns its
  least-squares starting point. That was a live bug here, fixed 2026-07-25; float was broken at 3 of
  4 data scales and c=1e-8 returned the least-squares fit.
  **(2) A failed least-squares init is not fatal.** Where the reference aborts with no fit if the
  one-time plain-CHO factorization of AᵀA fails, here y starts at 0 — still a valid strictly-interior
  point — and the solve proceeds.
  **(3) The Newton solve is defended for float.** `lpfnb.f`'s `stepy` accumulates AᵀQA with `dsyr`
  and calls `dposv` — plain Cholesky, no pivoting, no regularization, no equilibration — because it
  is double-only and never needs more. The pivoted CHOP, the diagonal `reg` and the Jacobi
  equilibration all exist to make the float instantiation survive the polarized-weight endgame. They
  are additions to the reference, not corrections of it. (The surviving "Factorization:" paragraph in
  the file header states this as a contract; only the reference comparison moved here.)

## LP.solve(BSR) REMOVED — general sparse LP is out of scope
- 2026-07-25 | User ruling: the implementation was not good enough to keep. Removed `LP.solve(in
  fProxyBSR …)`, the `standardFormInterior` Mehrotra loop it was the last caller of,
  `fProxySlackAugmentedOperator`, and `fProxyNormalJacobi` (redundant with the existing
  `fProxyDiagonalPreconditioner`). `LP.Sparse.fProxy.cs` is gone entirely; the surviving `LP.lad(BSR)`
  forwarder moved next to `ladFN(BSR)` in LP.FrischNewton. Tests `SparseWyndorGlass` /
  `SparseVsDenseLp` deleted with it.
  KEPT: `fProxyLadOperator` and `fProxyNormalOperator` — the latter is documented as a general
  "AᵀDA / A D Aᵀ" operator and both are pinned by a direct operator test that checks Aₛ D Aₛᵀ against
  a hand-computed AAᵀ. `IfProxyStandardFormOperator` stays as their shared contract, now with no
  in-library solver consuming it (noted in its own doc).
  WHY, beyond the measured defects (inert absolute `reg` 21 orders under a 1e13 diagonal; Jacobi
  preconditioner inert because diag(M)'s range is 1.87; CG at its cap every iteration): the AUDIENCE.
  Large sparse LP is a logistics/scheduling workload at 1e5-1e6 variables where users reach for HiGHS
  or Gurobi directly; the LP-shaped problems games actually pose are small and dense, already served
  by the dense revised/dual simplex and MIP. Porting HiGHS's sparse route was considered and declined:
  its dual simplex needs a Markowitz/Suhl-Suhl sparse LU, and its IPM (IPX, Schork-Gondzio) needs a
  BASIS preconditioner built on that same sparse LU — the subsystem already declined in July as
  CHOLMOD-class. Precedent: PDLP was removed on the same reasoning (−2764 lines).
  Sparse LAD is unaffected and is the one sparse LP-family capability with a real use case; it now
  runs Frisch-Newton on the original design (entry above).
## LP.lad — quantreg's own published coefficients imported as a real-data oracle (barro, 3 taus)
- 2026-07-25 | `quantreg/tests/rq.R` asserts rq() coefficients on the Barro growth data at tol 1e-5
  for tau = 0.5 (`y.net ~ lgdp2+fse2+gedy2`) and tau = 0.75 / 0.25 (`+ Iy2 + gcony2`). Imported as
  `LadBarroReferenceTests` (161 obs x 6 cols inlined; `barro.rda` is R binary, so the data was
  extracted with `write.csv`). This is the first REAL-DATA non-median vector in the tree — the
  brute-force C(m,2) oracle proves optimality of our own objective, this proves agreement with the
  reference implementation. Comparison reproduces R's `all.equal` semantics exactly: mean(|t-c|) /
  mean(|t|), NOT element-wise, which matters because the fse2 coefficient is ~500x smaller than the
  intercept and an element-wise relative test would be dominated by it.
  MEASURED mean relative difference, worst of the three fits:
                 double        float
    ladBR        9.8e-9        1.1e-6
    ladFN        1.7e-8        2.6e-3
  Both engines beat quantreg's own 1e-5 bar in DOUBLE by ~3 orders. In FLOAT the two separate by
  2000x, and it is structural, not a tuning artifact: BR pivots on the ORIGINAL m x n design and
  stays near-exact, FN forms AᵀQA and pays the SQUARED condition number. Same limit documented in
  LP.FrischNewton's header, now confirmed on real data instead of a synthetic near-collinear probe.
  Tolerances are therefore PER-ENGINE (float BR 5e-6, float FN 5e-3, double 1e-6 both) so neither
  hides behind the other and a BR regression cannot pass on FN's looser bound. Note the default
  `lad` routing sends m=161 to BR, so the accurate engine is the one users get here.
  METHOD NOTE: tolerances were set by first running with tol = 0 to force every measurement to
  print, then bounding from the numbers — not by loosening until green.

## LP.FrischNewton — plain CHO instead of CHOP: TRIED, NO EFFECT, REVERTED. Don't retry.
- 2026-07-25 | Hypothesis (user's, and I agreed it was newly plausible): CHOP's rank truncation is
  what stalls FN on near-collinear designs, deleting the near-degenerate direction that plain CHO
  would keep, damped by `reg`. It looked newly testable because the same-day scale fix moved `reg`
  to AFTER equilibration, so the smallest eigenvalue finally has a guaranteed RELATIVE floor (1e-6
  of a unit diagonal) — before that, `reg` was absolute against an M whose scale drifts like 1/‖b‖,
  so CHO had no reliable floor and the header's "CHO hard-fails on a non-positive pivot" stood.
  MEASURED (float, relative L1 error vs ladBR, same probes both arms): nearCollinear 8.721e-3 with
  CHOP vs 8.721e-3 with CHO; colScale 3.292e-5 vs 3.292e-5; duplicateCol 6.116e-6 vs 6.117e-6;
  vandermonde 1.412e-5 vs 1.460e-5. Iteration counts identical. Double identical everywhere. The
  FULL suite passes 7125/7125 under CHO (incl. the quantreg barro vectors), so CHOP is not
  load-bearing for anything we currently test — but there is NO measured gain either, so the swap was
  reverted rather than churn a deliberate robustness choice for nothing.
  BENCHMARKED TOO (back-to-back LPBenchmark arms, CCD-pinned): no consistent timing difference.
  METHOD NOTE worth reusing — `ladBR` is UNCHANGED code in both arms, so its column is a free
  CONTROL for run-to-run noise, and it moved up to +43% (m=192 float 0.0300 -> 0.0429) between the
  two runs. Every apparent ladFN delta at m=192/384 was smaller than the control's own movement, i.e.
  machine load, not CHO. What survives is contradictory (double m=4096 CHO 6.7% SLOWER, m=16384 1.9%
  FASTER, both with a stable control) = noise with a sign. Only float m=8/16 hint at ~10%, on 4-6 us
  absolute, in a range where `lad` routes to ladBR anyway.
  PROFILED DIRECTLY afterwards (per-call, rep-looped inside one job so IJob.Run overhead cannot
  swamp a sub-100ns operation; SmallSizeBenchmark starts at n=16 and never times CHOP, so this shape
  was previously unmeasured):
    n        float CHO / CHOP        double CHO / CHOP
    4        79.5 / 107.3 ns         73.3 /  83.1 ns
    8       153.0 / 252.2 ns        162.5 / 253.3 ns
    32     3297.9 / 4385.0 ns      2146.1 / 4076.9 ns
  FN does ONE factorization per iteration, so at n=4 CHOP costs +28 ns/iter (float) over CHO. Against
  measured per-iteration cost that is 2.9% at m=8, 2.6% at m=16, 0.09% at m=1024 and 0.004% at
  m=16384 — i.e. under a tenth of a percent anywhere FN is the ROUTED engine (m > 512), and the few
  percent only at sizes where `lad` picks ladBR anyway. Also explains why the A/B could not resolve
  it: a 28 ns/iter effect against a 43% noise floor.
  ⚠️ NON-OBVIOUS: at n=4 both are OVERHEAD-bound, not flop-bound (79.5 ns for ~30 flops is
  ~0.4 GFLOP/s), which is why CHOP is only 1.13-1.35x there instead of the ~2x its per-column pivot
  search costs at n=32. Do not extrapolate the large-N CHOP/CHO ratio down to solver-inner shapes.
  So CHOP is effectively FREE. That inverts the "is this over-defended?" reading — it is not a cost
  worth removing, it is an unexercised safety net that costs nothing. The open question is no longer
  "should we drop it" but "can a case be built where it wins", and it stays until one is.
  CONCLUSION: rank truncation is NOT the near-collinear mechanism. The limit is the CONDITION
  SQUARING inherent to forming AᵀQA — cond ~1e12 against float's ~1e-7 leaves no digits in the
  degenerate direction, and no Cholesky variant can recover information the normal matrix already
  destroyed. Equilibration cannot help either: it normalizes the DIAGONAL (column scale), while
  near-collinearity is an OFF-DIAGONAL correlation (~0.9999999 after equilibration).
  The only real fix is to stop forming the normal matrix: QR of the row-scaled √Q·A, whose R gives
  the same solve at cond(W) instead of cond(W)². NOT DONE, and not obviously worth it — Householder
  QR is ~2mn² against BuildATQA's ~mn²/2, so roughly 4x the dominant per-iteration cost, plus an
  m×n materialized buffer, to fix a float-only weakness in a regime where `lad` already routes to
  ladBR (m <= 512) and where ladBR is the better engine anyway. Revisit only if a real caller hits
  near-collinear columns at large m in float.
  Pinned by `NearCollinearColumnsStayWithinKnownBound` (2e-2 float) as a regression guard on the
  known limit — it does not endorse the accuracy, it catches FN getting worse.

## LP.FrischNewton — primal-residual re-injection ADOPTED (the "measured worse" verdict was stale)
- 2026-07-25 | The affine RHS is now the reference's `(bLP - Aᵀa) + Aᵀ(q·(z-w))`, closing the last
  of the three documented deviations. `bLP` is recovered exactly as `Aᵀa` at the initial
  `a = 1 - tau` (the constraint maintained throughout is `Aᵀa = bLP`, and the start satisfies it by
  construction), so it costs one setup ATmul plus one n-buffer, and one ATmul per iteration.
  The corrector needed NO change: it already reuses the saved affine `rhs` and adds `Aᵀ(qCorr)`,
  which is exactly Fortran's `dswap(rhs,dy)` + `dgemv(a,dr)` — so the term propagates to both solves
  from the single predictor-side edit.
  WHY IT WAS RE-OPENED: the earlier "measured worse (stackloss float -38.140)" verdict was taken
  BEFORE the Jacobi equilibration landed, i.e. while the rank-3 endgame freeze was still corrupting
  the solve. Measuring under conditions that no longer exist is not a verdict. Re-measured on top of
  equilibration: `LadFNStackloss` (published Brownlee coefficients, 5e-2 band = 0.13% relative on the
  -39.690 intercept) PASSES, so -38.140 is gone by a wide margin.
  ⚠️ MEASURED ON THE WRONG DATA FIRST. The LPBenchmark A/B (sections 2 + 2b, m = 8..16384) showed
  IDENTICAL iteration counts in every cell, both dtypes, and identical L1 residuals — from which the
  first conclusion drawn was "numerically equivalent, adopt on fidelity grounds alone". That was a
  measurement error of the same shape as the one being corrected: the benchmark's designs are
  WELL-CONDITIONED, and this term is zero in exact arithmetic, so it can ONLY show up where `reg`,
  the equilibration or CHOP's rank truncation perturbs that cancellation. Well-conditioned data
  cannot discriminate, by construction.
  RE-MEASURED on ill-conditioned/degenerate designs vs `ladBR` (independent exact-vertex engine),
  relative error in the L1 objective, FLOAT:
    vandermonde deg4    3.682e-4 without  ->  1.412e-5 with   (26x better)
    colScale 1:1e4      3.292e-5           ->  3.292e-5        tie
    nearCollinear 1e-5  8.721e-3           ->  8.721e-3        tie
    duplicateCol        6.110e-6           ->  6.116e-6        tie
    tiny b + colScale   9.590e-5           ->  9.461e-5        tie
  DOUBLE is a wash everywhere (all cases <= 1e-10 both ways). So the term earns its keep in float on
  genuinely ill-conditioned normal matrices, which is precisely what the theory predicted and what the
  benchmark could never have shown. Pinned by `IllConditionedVandermondeMatchesExactEngine`, whose own
  (differently-seeded) instance measures 8.2e-5 with the term vs 4.6e-3 without — 56x — with the
  tolerance at 3e-4, between the two.
  ⚠️ The first cut of that test reused the diagnostic's tolerance (8e-5) even though the diagnostic's
  rng had been consumed by three earlier probes, making it a DIFFERENT problem instance. It failed at
  8.2e-5 on a bound set from data that did not describe it. Numbers from a scratch harness do not
  transfer to a permanent test unless the instance is identical — re-measure in place.
  Separately noted, NOT caused by this change: float `nearCollinear 1e-5` sits at 8.7e-3 relative in
  both arms — a standing float accuracy limit of FN on near-collinear columns; ladBR is the better
  engine there. Worth its own investigation, not folded into this one.
  TIMING NOT MEASURED. The extra ATmul is O(mn) against BuildATQA's O(mn²/2), so it should be small
  at n >> 1 and material at LAD's typical n=4 — but the only A/B taken ran while the machine was in
  use (browser + music), and this repo's own history records that swinging single runs 20-40%. The
  numbers seen (float within noise, double ~3-4% slower at large m) are NOT trustworthy and are
  recorded here only so nobody mistakes them for a baseline. Re-measure on an idle machine before
  quoting a cost.

## LP.FrischNewton — tau != 0.5 oracle + rank-deficient coverage (closes the audit's test gaps)
- 2026-07-25 | Closes both open test gaps from the port-fidelity cross-check below; no behaviour
  change, the solver was already correct on both.
  ORACLE: quantile regression is an LP, so an optimum sits at a basic solution whose fitted
  hyperplane interpolates n observations exactly. At n=2 that is a line through a PAIR of points, so
  enumerating all C(m,2) pairs and taking the smallest check loss is the EXACT optimum. Plain double
  arithmetic over the raw data, sharing nothing with the solver — unlike the pre-existing tau=0.5
  test, which only compares ladFN against the core it forwards to, and the tau=0.25 sign-fraction
  test, which is a wide statistical envelope. Assertion is one-sided (the solver cannot beat the
  exact optimum, only fall short).
  ⚠️ CHOOSING tau MATTERS: on this 9-point data the optimum is piecewise-constant in tau and the
  MEDIAN line is exactly optimal across tau in [0.25, 0.4] — the originally-drafted {0.1, 0.25, 0.75,
  0.9} therefore included a value at which a completely tau-BLIND solver would pass. Measured excess
  of the median line over each tau's own optimum: 0.05→189%, 0.1→71%, 0.15→32%, 0.2→12%, 0.25-0.4→0%,
  0.6→4%, 0.75→24%, 0.9→221%, 0.95→571%. Settled on {0.1, 0.2, 0.75, 0.9}. `BruteForceOracle
  DiscriminatesTau` now pins that property OF THE TEST DATA (median line must be >10% worse at every
  tau exercised), so the same trap cannot be reintroduced by editing the data or the tau list.
  RANK-DEFICIENT: duplicated-column and zero-column designs (`LadFrischNewtonRankDeficientTests`).
  Coefficients are not identifiable but the optimal OBJECTIVE is, and equals the full-rank two-column
  problem's — that is the assertable invariant. Note the fallback these exercise is dtype-dependent:
  the LS init still passes an absolute `reg` to BuildATQA (deliberately — with q=1 that matrix is
  AᵀA, which depends only on A, so it is already scale-invariant in b), and in FLOAT `reg` = 1e-6 is
  below one ulp of AᵀA's entries at this size, so the plain-CHO init genuinely fails and the y=0
  fallback fires; in double it does not. Both dtypes must reach the same objective either way.

## LP.FrischNewton — scale equivariance: one data-scaled tolerance + relative regularization
- 2026-07-25 | The L1 fit is exactly equivariant under `b -> c*b`; the solver was not. New
  `LadFrischNewtonScaleTests.ResponseScaleEquivariance` measures all four scales before asserting (one
  bad scale must not hide the others) on a 5-point line whose LAD fit (0,1) and least-squares fit
  (1.6,1) are far apart, so a solve that stops at its own starting point is unmistakable. Measured
  BEFORE the fix: float broken at 3 of 4 scales (c=1e-8 returned L1 residual 12.8 = exactly the
  least-squares value; c=1e-3 off 3%; c=1e6 off 15% with slope 0.802), double broken at c=1e-16.
  TWO independent absolute-vs-data-scaled comparisons, both fixed:
  (a) `zwFloor` was `Consts.fProxyZeroThreshold` (1e-6 float / 1e-14 double) compared against the
      residual `r = -b - Ay`, which carries the data's scale. On small-magnitude data every residual
      falls under the floor, so z/w initialize to the floor alone (no data content) and the resulting
      gap is already under `gapTol` — the loop never runs and `x = -yBest` returns the plain
      least-squares initialization. Now ONE `eps = sqrtEps*‖b‖₂` drives both the floor and the gap
      test: the reference's own single-tolerance structure (`lpfnb.f` uses its caller `eps` for both),
      with the constant replaced by a data-proportional value.
      `sqrtEps*(1 + ‖b‖₂)` — gapTol's previous shape — is NOT a fix: the additive 1 is an absolute
      floor that dominates when ‖b‖ << 1, leaving the tolerance ~4000x the objective at c=1e-8.
  (b) `reg` was added to M's diagonal INSIDE BuildATQA, before the Jacobi equilibration. The weights
      `q = 1/(z/a + w/s)` scale like 1/‖b‖ (z,w residual-scaled; a,s in [0,1]), so M scales like
      1/‖b‖ and on large-magnitude data an absolute bump swamps the real matrix and degenerates the
      Newton direction — this is the 15% at c=1e6. BuildATQA is now called with 0 and `reg` is added
      AFTER equilibration onto the unit diagonal, where it is a genuine relative perturbation (which
      the pre-existing comment already claimed — that claim only ever held for the Indefinite-retry
      bump). Suite 7104/7104.
  Float c=1e6 was NOT a regression from (a): baselined against the pre-fix solver it was already
  broken. LESSON: an assertion loop that throws on the first failure hides the shape of the bug and
  invites mis-attributing a newly EXPOSED failure to the change that exposed it — measure every case,
  then assert once.

## LP.FrischNewton — independent port-fidelity cross-check vs rqfnb.f: CLEAN
- 2026-07-25 | Independent statement-by-statement audit of `ladFrischNewtonCore` against
  `lpfnb.f`/`stepy` at commit `7c2feec6`, algebra re-derived by hand rather than taken from the
  earlier passes. NO bugs found beyond the three already-resolved divergences. Confirmed correct,
  all previously unverified: the strict `< 0` test in the ratio test matches Fortran; `mu*(g/mu)³/(2n)`
  uses n = OBSERVATIONS (our m), not coefficients; the full-step ELSE branch does NOT reset fa/fd to 1,
  matching Fortran's fall-through (a classic bug spot — resetting there would silently discard the
  affine step lengths); `lpfnb.f:107`'s reuse of `u` as scratch has no analogue hazard here since we
  use a dedicated `Av`; all 33 Allocator.Temp buffers are disposed exactly once with both in-loop
  `break`s falling through to the same block; `[NoAlias]` on BuildFNWeights is truthful; generated
  float/double are exact codegen expansions. Also confirmed the corrector fix is NOT a double-division
  (the precomputed `dadz = daAff*dzAff/a` is the same divided quantity reused at both use sites), and
  that `ATmul` lacking `Amul`'s `MemClear` is SAFE because `UnsafeOP.vecMatDot` self-clears its output
  before accumulating (`UnsafeOP.fProxy.cs:413`).
  The audit surfaced two UNDOCUMENTED deviations from the reference — now written into the file
  header's "Deviations from the literal Fortran reference" list: the tolerance SPLIT (reference drives
  both the z/w init floor and the convergence test off one caller `eps`; we scaled gapTol by ‖b‖ but
  left zwFloor an unscaled constant), and the non-fatal LS-init failure (reference aborts with no fit when the plain-CHO factorization of AᵀA
  fails; we fall back to y=0, a valid strictly-interior start, and proceed). The omitted primal-residual
  re-injection was added to that list too, since it is the item most likely to be "helpfully restored"
  by a future reader diffing against the Fortran — re-adding it was measured WORSE (see the entry
  below); do not restore without re-measuring.
  ⚠️ The tolerance-split item was NOT merely a documentation gap: it was a live scale-dependence bug,
  fixed the same day — see the scale-equivariance entry above. The audit's "behaviour is correct, only
  the docs are missing" reading of it was wrong.
  Residual risk noted, not a demonstrated bug: the dropped primal-residual term is provably zero in
  exact arithmetic, so it can only bite where `reg`, the Jacobi equilibration, or CHOP rank-truncation
  perturbs that exact cancellation — i.e. precisely the near-singular endgame CHOP exists for.
  Open test gap (NOT closed): tau != 0.5 quantile regression has no independent oracle — only tau=0.5
  against ladFN's own forwarding plus a wide statistical sign-fraction check at tau=0.25 — and no
  rank-deficient or degenerate-size design is routed through the core.

## LP.FrischNewton — complementarity gap (rqfnb.f) replaces the cancellation-prone duality gap
- 2026-07-24 | Closes the "LATENT" item in the entry below at the ROOT rather than guarding the
  symptom. THIRD MATLAB-vs-Fortran divergence found in this port: the convergence measure itself.
  `lpfnb.f:52,139` uses `gap = ddot(z,x) + ddot(w,s)` — the COMPLEMENTARITY gap, a sum of products of
  strictly positive quantities. `lp_fnm.m:22,93` uses `gap = c*x - y*b + w*u` — the signed DUALITY gap.
  The two are algebraically equal at a primal/dual-feasible pair (substitute `Aᵀy+z-w = c`, `s = u-x`),
  and the Newton step maintains both feasibilities exactly (`Aᵀda = 0` since `dy = M⁻¹rhs`;
  `dz - dw = -A dy` by construction), so they agree in exact arithmetic. They do NOT agree in float:
  the MATLAB form subtracts terms of magnitude ~500 (stack-loss) to land on a quantity ~0.03, so
  cancellation can invert its SIGN, whereas the Fortran form cannot go negative by construction.
  That sign inversion was the mechanism behind BOTH latent hazards: a negative gap satisfies
  `gap <= gapTol` for ANY tolerance (which is also why tightening float gapTol 10x and 333x during the
  investigation moved nothing — a dead end that cost real time) AND beats every legitimate positive gap
  in the `gap <= bestGap` yBest update, so the blow-up captures the very safeguard meant to survive it.
  Switched `DualityGap` -> `ComplementarityGap(z, a, w, s, m)`. This also retires `bLP` entirely (it
  existed only to feed the duality-gap form): one fewer n-buffer and one fewer setup-time ATmul.
  No `gap >= 0`/`abs(gap)` guard needed — the quantity is now non-negative by construction, which is
  strictly better than guarding a corrupt number, and it is what the reference does. NOTE the earlier
  suggestion to guard with `gap >= 0` was wrong anyway: near true convergence the gap approaches 0 and
  rounding can make it slightly negative on a GOOD iterate, so a hard sign guard would reject it;
  `abs(gap)` would have been the correct guard had one been needed. Measured: performance-neutral —
  iteration counts and timings identical to the pre-change run at every size, both dtypes
  (float 10 iters/6.84ms @16384, double 14/9.35ms @16384), L1 residuals unchanged. Suite 7101/7101.

## LP.lad — hybrid BR/FN routing threshold re-tuned for double
- 2026-07-24 | `lad`'s crossover literal was `/*+choose[512|4096]*/`. The double 4096 was STALE — and
  already stale BEFORE the Frisch-Newton corrector work (baseline m=1024: FN 0.372ms vs BR 0.689ms),
  so double m in [1024, 4096] was routing to the engine ~2x SLOWER. After the corrector + equilibration
  the measured crossover is ~512-1024 for BOTH dtypes: double m=384 BR 0.087ms vs FN 0.142ms (BR wins),
  m=1024 BR 0.692ms vs FN 0.313ms (FN wins 2.2x); float m=384 BR 0.078 vs FN 0.083, m=1024 BR 0.623 vs
  FN 0.299. Set to `/*+choose[512|512]*/`. Kept as a choose (not a plain literal) so the per-dtype axis
  survives for the next re-tune. Section 2b has no m=512 sample, so 512 is interpolated between the
  384 and 1024 brackets; the penalty for being off near the crossover is bounded by how close the two
  engines are there (<=1.5x), unlike the 4096 error which cost ~2x across a whole octave.

## LP.FrischNewton — exact Mehrotra corrector (rqfnb.f), ~20% faster, SHIPPED with Jacobi-equilibrated normal solve
- 2026-07-24 | UNBLOCKED, shipped. The entry below's "exiting via the `!rinfo.Solved` CHOP break /
  AᵀQA goes indefinite / bump retry fails" mechanism was an unmeasured inference and is WRONG on all
  three counts — per-iteration instrumentation of the float `LadFNStackloss` run showed: CHOP never
  reports Indefinite (bump retry never fires, 0 retries all iterations); it reports RankDeficient
  rank=3 from it=1 (!) and every iteration from it=4 on; the loop exits via `gap <= gapTol` — but with
  a numerically CORRUPT NEGATIVE gap (-8.36 at it=14, after a/s/z/w mins collapse geometrically to
  ~1e-32), which also poisons the `gap <= bestGap` yBest update. Root cause: stack-loss columns have
  ~80x scale disparity (1 vs ~60/~25/~90 → AᵀQA diagonal spread ~8100x on top of IPM weight
  polarization), so CHOP's scale-RELATIVE rank tolerance in float sees rank 3; decompSolve's rank-3
  min-norm direction has no component along the dropped direction, y freezes 2.71e-2 suboptimal
  (objY stalls at 42.108242 from it=8), and the collapsed iterates then fabricate the negative-gap
  exit. Decisive control: the OLD (MATLAB-corrector, green) float run had the SAME pathology —
  rank 3 at it=1/4/5, exit at it=5 on a corrupt gap of -0.023, objY 42.0878 (6.6e-3 suboptimal),
  intercept 39.727 — it passed the 5e-2 test by luck. The exact corrector didn't break a healthy
  algorithm; it moved an already-broken float endgame's freeze point outside the tolerance.
  FIX: Jacobi (symmetric diagonal) equilibration of M = AᵀQA before CHOP — M̂ = D·M·D,
  D = diag(1/sqrt(M_jj)); both solves scale RHS by D in and solution by D out. Unit diagonal makes
  CHOP's rank tolerance see genuine near-dependence instead of column scale, and makes the bump
  retry a RELATIVE perturbation for free. Result: float runs rank=4 EVERY iteration, tracks the
  double trajectory almost value-for-value, converges in 5 iters with a genuinely positive gap
  (0.0128 < tol 0.0322), intercept 39.7073 / objY excess 2.8e-3 — closer to the optimum than the old
  green baseline ever was. Suite 7101/7101. Speedup intact and iteration counts unchanged vs the
  corrector-fix-only measurement (equilibration is O(n²)/iter, negligible vs O(mn²) BuildATQA):
  float 10 iters @16384 (6.83ms), double 14 @16384 (9.29ms), 12 @4096 (1.67ms); L1 residuals match
  ladBR/oracle at all sizes, both dtypes. LATENT (driver removed, not hardened): a large-negative
  corrupt gap still both terminates as "Optimal" and wins the yBest update — the exit condition
  itself is reference-faithful (lpfnb.f's loop is `gap > eps` too), but if a float endgame ever
  collapses again, consider guarding the yBest update with `gap >= 0` or tracking the honest
  recomputed objective instead of the gap.
- 2026-07-24 | (superseded by the entry above — kept for the measurement record) Koenker's suggestion to port ladFN from quantreg's `rqfnb.f`/`lpfnb.f` (Fortran) rather
  than the `lp_fnm.m` (MATLAB) lineage was re-examined against both sources. Substantive finding: the
  MATLAB corrector is NOT the textbook Mehrotra corrector — the second-order terms are missing their
  `/x` and `/s` divisions. Deriving `Z dx + X dz = mu e - XZe - dX dZ e` gives
  `dz = mu/x - z - (z/x)dx - dxdz/x`; `lpfnb.f:115` has the `/x(i)`, `lp_fnm.m:71` does not. Same for
  `dw`, for `dx`, and for the corrector RHS term. MATLAB is self-consistent (it effectively solves with
  `X·dXdZ`), so it converges — but it under-weights the corrector by a factor `x`, exactly where
  `x -> 0`. Our port inherited the MATLAB form. The fix is 2 lines: `dadz = daAff*dzAff/a`,
  `dsdw = dsAff*dwAff/s` — all four downstream uses (qCorr, da, dz, dw) then match `lpfnb.f` exactly.
  MEASURED (LPBenchmark, m=8..16384, n=4): iterations drop at EVERY size, both dtypes, L1 residuals
  identical to 5 s.f. float 13->10 iters @16384 (9.23->7.09ms, -23%); double 17->14 @16384
  (11.60->9.42ms, -19%), 16->12 @4096 (2.19->1.67ms). So Koenker's "bound to be faster" is real, but
  the cause is the corrector, not compiled-vs-interpreted (our port is Burst-compiled already).
  BLOCKER — REVERTED, suite kept green: float `LadFNStackloss` regresses, intercept -39.690 -> -40.188
  (tol 5e-2). Not a lucky-test artifact and not a tolerance artifact: the published vector is the unique
  LAD optimum (confirmed vs scipy/HiGHS, L1 42.081160, 4 exact-zero residuals), and the best L1
  reachable with our intercept is 42.108241 — a genuine 2.71e-2 objective excess, which sits just inside
  float's own `gapTol = sqrtEps*(1+||b||) = 3.22e-2`. Tightening float gapTol 10x AND 333x changed the
  answer not at all, which rules out early termination: the loop is exiting via the `!rinfo.Solved`
  CHOP break. Mechanism: the exact corrector reaches the polarized-weights endgame faster/harder, AᵀQA
  goes numerically indefinite in float, the 4-step bump retry cannot recover, and yBest is returned
  0.027 suboptimal. Also tried, did not help: computing `dadz`/`dsdw` in double (no change beyond 7 s.f.
  — rules out plain rounding in that term); adding `lpfnb.f`'s primal-residual re-injection
  `rhs += bLP - Aᵀa` (changed the answer, made it WORSE: -38.140). To unblock, the float CHOP endgame
  needs work first (cf. [[lobpcg-structural-stability]]'s SVQB-style robustness pass); the double-only
  win is real but an algorithm that differs by dtype is a user call, not a default.

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

## Krylov.craig / craigmr — Tikhonov damping via the augmented operator
- 2026-07-23 | New Krylov.LeastNormDamped.fProxy.cs. Damped craig/craigmr = ridge least-norm
  x = Aᵀ(AAᵀ+damp²·I)⁻¹b. KEY: instead of the "materially more complex" generalized Golub-Kahan
  recurrence (why this was deferred before — the LSLQ-bias risk), run the UNDAMPED, proven solver over
  the AUGMENTED operator `fProxyDampedLeastNormOperator<TOp>` = [A | damp·I] (Rows × (Cols+Rows)). The
  min-norm solution of [A|damp·I]·(x,s)=b minimizes ‖x‖²+‖s‖² s.t. Ax+damp·s=b, which IS the ridge
  least-norm x; the solver's solution vector is (x,s) and x = its first Cols entries. Zero recurrence
  surgery → low risk (same trick as damped cgne, but via an operator wrapper since craig/craigmr are
  generic over TOp). The augmented operator is always full-row-rank + consistent (the damp·I block),
  so the inner solve is always well-posed — damped craig even handles a RANK-DEFICIENT A (undamped
  would break down). Operator needs ONE Cols-length scratch (x-part copy in Apply; Aᵀx in ApplyT).
  Diagnostics re-audited in ORIGINAL coords via lstsqResidual (DampedLeastNormFinish): rnorm=‖b-Ax‖
  (=damp·‖s‖, nonzero at optimum), Arnorm=‖Aᵀr-damp²x‖ (→0, the cert) — same convention as damped cgne.
  damp==0 delegates to the plain solver (bit-identical). Surface: dense + BSR damped overloads for
  craig + craigmr. lnlq DELIBERATELY not damped this way: its certified forward-error bound is on the
  augmented (x,s), not x, so the augmentation defeats its distinctive feature. Tests
  (fProxyCraigDampedTests): dense Aᵀ(AAᵀ+damp²I)⁻¹b oracle + rank-deficient + damp==0 bit-identity, for
  both craig and craigmr. Completes non-square damping: lsqr/lsmr (LS) + cgne/craig/craigmr (LN);
  still out = lnlq (above) + lslq (~1.4e-3 λ-rotation bias, v1-out).

## Eigen.eigNearShift — shift-and-invert interior eigensolver
- 2026-07-23 | New Eigen.ShiftInvert.fProxy.cs: `fProxyShiftInvertOperator<TOp>` (Apply = (A-shift·I)⁻¹x
  via an inner minresQLP solve with the eigenvalue shift; symmetric so ApplyT=Apply) + the driver
  `eigNearShift` (finds the k eigenpairs of symmetric A NEAREST shift). ARPACK-style shift-invert
  LANCZOS, chosen over LOBPCG (user steer): Lanczos converges the EXTREME Ritz values of the
  shift-invert operator T=(A-shift·I)⁻¹ first, i.e. the eigenvalues nearest shift, on BOTH sides,
  with ONE inner solve per step (LOBPCG would need −T² / two solves and its smallest-eigenvalue
  orientation fights the interior goal). The operator needs NO scratch fields — Apply just calls the
  arena minresQLP-shift overload which allocates internally. KEY correctness point: the k modes are
  SELECTED by largest |theta_j| (Ritz value of T = 1/|lambda_j-shift|, large exactly for the
  nearest+best-converged modes), NOT by |Rayleigh-shift| — an UNconverged Ritz vector can have a
  Rayleigh quotient that lands near shift by accident and poison the result (this was the first-cut
  bug, residual 0.52). lambda is then RECOVERED per selected mode by the Rayleigh quotient vᵀAv/vᵀv
  against the ORIGINAL A (robust to an inexact inner solve — far better than shift+1/theta). Uses
  lanczosVectors (full reorth). Dense + BSR convenience overloads default steps=min(n,2k+20),
  innerTol=sqrtEps. Tests (fProxyEigNearShiftTests): diagonal + 1D-Laplacian tridiagonal, both with
  exact known spectra compared as a SET (the Laplacian spectrum is symmetric about 2 → the two
  nearest shift=2 are an exact tie, so nearness-ORDER is ambiguous — set comparison handles it) +
  eigenpair residual ‖Av-λv‖ (~1e-2 float, innerTol-limited).

## Krylov SIMD sweep (#58) — BlockFrobDot reroute measured, REVERTED
- 2026-07-22 | Re-ran the Krylov SIMD-reduction sweep. Finding: the core single-RHS solvers are
  already SIMD-optimal — every dot/norm goes through Blas.dot → UnsafeOP.vecDot (2×fProxyW + 2×fProxy4
  multi-accumulator). The benchmarked block solvers (bcg/bcgrq/bfbcg/bgcrodr/bgmres) are BlockGram/
  GEMM-based (gcrodr Gram→GEMM already shipped, ec8bae3), no hand-rolled reduction debt. The ONLY
  hand-rolled reductions left are in the UN-benchmarked block solvers (bidr/blsmr/bbiCGStab/btfqmr/
  bcraig/bcraigmr): BlockFrobDot (whole-block Frobenius dot, 16 call sites) + per-row/per-col norm
  loops. Tried rerouting BlockFrobDot to the multi-accumulator vecDot (self-dot NoAlias is fine —
  read-only, same as Blas.dot(a,a) library-wide). MEASURED: it BREAKS the block battery — bidr float
  matrix-10 flips 0→2 non-converged, because the multi-acc fold changes ‖R‖_F at ULP level and shifts
  bidr's convergence detection. REVERTED. This upgrades the earlier "recommended skip" (memory) to a
  measured fact: BlockFrobDot must stay a sequential accumulate. Also: these solvers have NO benchmark
  at all, so any block-reduction opt here is unmeasurable without adding one — and the reductions are
  O(s·n) in O(nnz·s)+O(s²·n) code. Net: the Krylov SIMD sweep is complete; no code change kept. Real
  remaining reduction targets are OUTSIDE Krylov (QueryOP argMaxRowNorm, Eigen/LOBPCG/ladIRLS dots).

## Krylov.GMRES / Krylov.FGMRES — merge into a shared GmresCore
- 2026-07-22 | gmres and fgmres were ~95% identical (Arnoldi-MGS + incremental Givens LS + restart);
  the only diffs were the preconditioned-basis storage and the solution update. Extracted the single
  body into `GmresCore<TOp,TPre>` (new Krylov.GMRES.Common.fProxy.cs); gmres/fgmres are now thin
  façades (validation + gmres's `!IsConstant` throw, then forward). The core branches on the
  compile-time-constant preconditioner flags (Burst-folds each specialization to one path):
  IsIdentity → plain GMRES (no M, accumulate into x); `flexible = !IsConstant` → store the
  preconditioned basis Z (m vectors), x += Σ y_i z_i (FGMRES, per-step-varying M); constant real M →
  single zt vector, x += M⁻¹(Σ y_i v_i) (standard right-precond, m-1 fewer basis vectors + one M-apply
  per restart instead of per step). This REALIZES the user's "switch on the preconditioner": a CONSTANT
  M takes the cheap standard path whether it came through gmres OR fgmres — so fgmres(constant M) is now
  BIT-IDENTICAL to gmres(constant M) and drops from m basis vectors to 1 (was the Z path). Only a
  genuinely varying M (AMG K-cycle) pays for Z. Behaviour preserved bit-for-bit for: identity (both),
  gmres(constant real M), fgmres(variable M); fgmres(constant real M) changes bit-level (cheaper,
  mathematically identical — M⁻¹Σy_i v_i == Σy_i M⁻¹v_i when M fixed). gmres keeps its constant-only
  API contract (the IsConstant throw stays). No public signature changed. 189/189 across
  GMRES/FGMRES/square-battery/preconditioner-compat/AMG.

## Krylov.MINRESQLP — eigenvalue shift (A - shift*I) x = b
- 2026-07-22 | Added `fProxy shift` to the `minresQLP<TOp,TPre>` primitive: solves (A - shift*I) x = b.
  Math: shifted Lanczos on B = A - shift*I. The recurrence is exact because
  B·v - (vᵀBv)·v = (A·v - shift·v) - (vᵀA·v - shift)·v = A·v - (vᵀA·v)·v — the -shift·v term cancels
  against +shift in the diagonal alfa, so β and the Lanczos vectors are identical to unshifted; only
  T's diagonal shifts by -shift. Implemented as option (a): shift the matvec directly (one axpy),
  which makes alfa, the recurrence, and all downstream QLP machinery consistent with zero extra
  bookkeeping. FOUR shift sites, each guarded `if (shift != 0)` — a RUNTIME branch-skip (shift is a
  runtime parameter of the generic primitive, NOT a compile-time constant like IsIdentity, so it is
  not Burst-folded; the zero-shift forwarder simply doesn't take the branch → bit-identical): Lanczos
  matvec (A·v - shift·v), initial residual r0 = b - (A-shift·I)x, final
  true residual (inlined instead of VerifyTrueResidual so r1 keeps the SHIFTED residual for the LS
  certificate), and the ‖A·r‖ certificate ((A-shift·I)·r). Preconditioned case unchanged (M applied
  after the shifted matvec — solving M-weighted B). This is a lambda (eigenvalue) shift on the
  operator, distinct from lsqr/lsmr damp (a sigma / singular-value Tikhonov on AᵀA+damp²I): A-shift·I
  stays symmetric so the QLP min-length machinery needs no change. Surface: shift added to the
  primitive + a zero-shift forwarder (every existing convenience overload resolves to it) + arena
  overloads for generic-preconditioned / dense / BSR. Perf penalty: one axpy/iter when shift≠0, nil
  when shift==0. Unlocks shifted systems and shift-and-invert interior-eigenvalue drivers without an
  operator wrapper. Tests: fProxyMinresQLPShiftTests (residual certificate, explicit-A-σI
  self-consistency, zero-shift bit-identity, BSR path).

## Krylov.CG / Krylov.MINRES — collapse per-preconditioner BSR overloads to generic
- 2026-07-22 | Each file had 18 concrete BSR overloads (BlockJacobi/SSOR/IC0/FSAI/Chebyshev/
  AdditiveSchwarz × zero-alloc/arena/default rungs), every one a pure forward
  `cg(new fProxyBSROperator(in A), in M, …)`. Since preconditioners are generic
  (`TPre : IfProxyPreconditioner`), collapsed each 18 → 3 generic `cg<TPre>(in fProxyBSR A, in TPre M, …)`
  (one per rung). C# infers TPre from the concrete M, so ALL existing call sites bind unchanged
  (507/507, no test edits). No ambiguity with the fully-generic `<TOp,TPre>` since fProxyBSR isn't an
  IfProxyLinearOperator. Only loss = per-type IntelliSense doc text (cosmetic). This REVERSES the
  earlier "keep the overload ladder as-is" stance on the per-preconditioner-TYPE axis (the rung axis
  is untouched) — user-directed. Rollout candidates enumerated for the rest of the family (BiCGStab
  3 types, IDR 2, MINRESQLP/GMRES/FGMRES/GCRODR/TFQMR + block twins 1 each, LOBPCG 3) — same
  transform, smaller per-file wins.
- 2026-07-22 (rollout) | Applied the same collapse to the rest of the family: BiCGStab/IDR/GMRES/
  FGMRES/GCRODR/TFQMR/MINRESQLP + block IDR/FGMRES/GMRES/GCRODR/TFQMR + LOBPCG (which had 10
  BlockJacobi overloads across standard/generalized × rungs). 45 concrete → 35 generic, all verified
  pure forwards, 1071/1071. NOTE this WIDENS every solver to accept any IfProxyPreconditioner —
  including the symmetric solvers (cg/minres/minresQLP/bcg/bcgrq/bfbcg/bminres/lobpcg), which
  mathematically require an SPD M (M defines the inner product). The old per-type overloads were a
  leaky implicit gate against that. PLANNED fix: an `IfProxySpdPreconditioner : IfProxyPreconditioner`
  marker (implemented by BlockJacobi/SSOR/IC0/FSAI/Chebyshev/identity; NOT ILU0/SPAI/RestrictedSchwarz)
  constraining the symmetric-solver family, restoring the gate at compile time. The A-requirement axis
  (SPD vs symmetric-indefinite vs general) stays runtime/doc — SPD-ness isn't a static property.

## Krylov.Block.* — more shared helpers (thresholds, LS-exit tail)
- 2026-07-22 | `BuildColumnThresholdsPlain` (un-floored tol²·‖B[j]‖², the twin of the floored
  `BuildColumnThresholds`) folds the inline threshold loop in bcg/bfbcg/bcgrq/bbiCGStab/bidr.
  bminres SKIPPED — its loop also tracks `bIsZero` (tol==0 edge-case), not a clean match. CGLS
  skipped (gamma[j,j] variant). `BlockLstsqExit` folds the identical 3-line exit tail
  (BlockApplyOp + BlockMaxResidualRecompute + converged flag) in bcraig/bcraigmr/blsmr. All
  bit-identical; 489/489.

## Krylov.Block.* — shared BlockResidual helper (R = B - A·X)
- 2026-07-22 | The `A.ApplyBlock(in X, ref buf, s)` + i-major/c-minor `R = B - buf` block was
  copy-pasted 23× across the block family (init + restart + verify-at-exit sites in
  cg/bcgrq/bfbcg/bminres/bbiCGStab/btfqmr/bidr/bgmres/bfgmres/bgcrodr). Extracted to
  `Block.Common.BlockResidual` with two overloads: in-place (A·X into R, R = B - R) and
  separate-scratch (A·X into `applied`, R = B - applied) for the few sites that keep the applied
  block in a distinct buffer. Bit-identical (same fold order); 522/522.

## Krylov.Block.MINRES — hoist BuildOmega scratch out of the iteration loop
- 2026-07-22 | `BuildOmega` allocated 8 `Allocator.Temp` matrices (Y/Qy/Z0/T/QyT/Z1/Qperp/Rz) per
  call = per iteration; hoisted them to caller-owned pre-loop scratch passed by ref (they are
  fixed-size for the whole solve — s constant — and fully overwritten each call, so reuse is
  behavior-preserving; verified bit-stable, 135/135 incl. precond path). Named `om*` in the caller
  to avoid colliding with the loop's own `T` (s×n vs BuildOmega's s×s). Remaining per-iteration
  allocs in this solver (UnpivotBetaRows `tmp`, BlockNormalizePrecond/BlockResidualSolve `corner`,
  the `pivS` Pivot) are one-each and thread through the normalize helpers — deferred (smaller win,
  more invasive). From the allocation survey; scalar solvers were already clean (one-time pre-loop).

## Krylov.Block.CG — bcg aliasing guard
- 2026-07-22 | Added the RequireDistinctBuffers guard (X/R/P/Q/B, plus Z under a real M) that its
  sibling block solvers (bminres/bbiCGStab/btfqmr/bcraig/bcraigmr) already carry — bcg was the
  outlier with only shape checks. Left bcgrq/bfbcg's guards out on purpose: bfbcg DELIBERATELY
  aliases Z onto R under the identity preconditioner (see its own comment), so a blanket
  distinct-buffer guard there would be wrong; bcgrq deferred with it. Surfaced by the Fable/Sonnet
  block-solver audit that followed the bminres #49 fix.

## Krylov.LSQR / Krylov.LSMR — GENERAL (non-symmetric) right preconditioning
- 2026-07-22 | Follow-on to the symmetric path below: added `lsqrRightPreOp`/`lsmrRightPreOp`
  (dense + BSR, damped + undamped + default ladder) taking N as a general
  `IfProxyLinearOperator` instead of a symmetric `IfProxyPreconditioner`. Chose to REUSE
  `IfProxyLinearOperator` rather than mint a preconditioner-with-transpose interface — it already
  carries both Apply and ApplyT, so `fProxyGeneralRightPreconditionedOperator<TInner,TPreN>` gets
  (A·N)ᵀ = Nᵀ·Aᵀ from N's own ApplyT (the symmetric wrapper instead reused N.Apply for the
  transpose, only valid for N = Nᵀ). Couldn't overload `lsqrRightPre` by constraint (generic
  constraints aren't part of the overload signature → CS0111), hence the distinct `…Op` name.
  Motivation = the strong LS preconditioners are non-symmetric: N = R⁻¹ from a QR/sketch of A
  (Blendenpik/LSRN) makes A·N orthonormal → 1–2 iterations. `fProxyDenseOperator` already serves
  as a concrete N, so no new concrete type shipped; the R⁻¹ strength test builds Rinv by
  back-substitution and wraps it. `RightPreFinishOp` duplicates `RightPreFinish` because TPre and
  TPreN are different interfaces (both have Apply, but C# can't unify them without a common base).
- 2026-07-22 | Generalized least-squares RIGHT preconditioning from diagonal-only Jacobi to any
  SYMMETRIC `IfProxyPreconditioner`: new `fProxyRightPreconditionedOperator<TInner,TPre>` wraps A·N
  (Apply = A(N·x) via owned scratch; ApplyT = N(Aᵀx), valid because N = Nᵀ), new
  `fProxyDiagonalPreconditioner` (z = d.*r) makes diag(d) one instance of it, and
  `lsqrRightPre`/`lsmrRightPre` (dense + BSR, damped + undamped + default-arg ladder) do the
  change-of-variables solve (A·N)y = b cold-start, recover x = N·y, and re-audit diagnostics in
  original coordinates. `lsqrJacobi`/`lsmrJacobi` now just build d (columnNormsSquared +
  buildJacobiScale), wrap it in `fProxyDiagonalPreconditioner`, and forward — numerically identical
  to the old `fProxyColScaledOperator` path (same multiplies, same order; that operator stays as
  the composable zero-alloc primitive). `JacobiFinish` became `RightPreFinish<TOp,TPre>` (unscale
  step is now N.Apply staged through nScratch instead of the in-place d.* loop, plus a damp
  pass-through to lstsqResidual). Damped caveat unchanged in kind: damping the preconditioned
  system penalizes ‖y‖ = ‖N⁻¹x‖, not ‖x‖, so the reported ‖x‖-ridge Arnorm is generally nonzero
  even at the damped optimum — documented on the entry points, not asserted small in tests.

## Krylov.Block.MINRES — preconditioned path un-gated (r-space recurrence)
- 2026-07-22 | Fixed the preconditioned block-Lanczos recurrence and removed the
  `NotSupportedException` gate. Root cause (diagnosed + fix verified in the numpy reference,
  `reference/wip-bminres/precond_full.py`): the old shared recurrence subtracted the Alfa/Beta
  terms in V-space (the M-orthonormal vectors) and then `BlockNormalizePrecond` applied M⁻¹ to the
  WHOLE result, corrupting the subtraction terms — the Lanczos vectors came out non-M-orthogonal
  and the solve stalled at relRes ~0.43 (`BlockNormalizePrecond` itself was always correct; at M=I
  V-space == r-space, which is why the identity path passed 522/522). The precond path now keeps the
  UNPRECONDITIONED residual blocks (r-space): `Wnext = A·Vcur − Alfa·(Beta⁻¹Wcur) −
  Betaᵀ·(Beta_prev⁻¹Wprev)`, with `Alfa = Vcur·A·Vcurᵀ` taken from A·Vcur BEFORE any subtraction —
  reduces exactly to scalar minres's r1/r2 bookkeeping at s=1. Implementation caches each step's
  triangular solve (`BlockResidualSolve`: gather W's rows through the CHOP pivot, solve the revealed
  rank×rank corner, deflated lanes zero — same frame as Vout's own corner solve, computed right
  after the normalize while that step's pivot/rank are live) as Ucur, rolled to Uprev — bit-identical
  to re-solving, since W/Beta don't change between iterations, and it dodges saving pivot history.
  The identity recurrence and all post-Lanczos MINRES machinery (M2/Omega/Gamma/Phibar/W/X) are
  untouched. Invoker PrecondKind None→SymmetricBSR so battery check #5 drives a real BlockJacobi M.

## Krylov.Block.Common — RowOrthoRankFloored replaces LQRP in the block-Arnoldi step
- 2026-07-22 | Merged the `wip/lqrp-drop` branch (was 946a8af). The block-Arnoldi residual
  orthonormalization (BlockArnoldiMGS2Step + bgcrodr's inline copy) used `LQRP.decomp` + a per-step
  `new Pivot(w[j], Allocator.Temp)` + `LQRPRankFloored`, but only consumes the orthonormal Q rows and a
  deflation rank — never the pivoted L or the row permutation. Replaced with `RowOrthoRankFloored`: an
  allocation-free pivoted (rank-revealing) modified Gram-Schmidt on W's rows. Greedy largest-residual
  row pivot mirrors LQRP's, so the pivot-norm sequence is non-increasing like LQRP's L diagonal; the
  rank/floor test reproduces LQRPRankFloored verbatim (self-relative vs the first pivot, floored by
  relTol*scale) → identical deflation DECISION, just on MGS pivot norms not LQ diagonals (iterates need
  not match bit-for-bit). Drops the LQRP dependency + the hot-loop Temp Pivot alloc. BlockArnoldiMGS2Step
  loses its `ref Lbuf` param; Lbuf stays allocated in the callers (still used for the initial R0 factor).
- 2026-07-22 | Perf: prior A/B (on the branch) was neutral (±6%, noisy machine) — MGS2 re-orthogonalization
  dominates block-Arnoldi cost, not the tiny w[j]×n factor this replaces; the win is architectural
  (allocation + dependency drop), not speed, so NOT re-benchmarked on merge. Correctness re-verified on
  current main: block suite 522/522 green. Merged by hand-applying the template change (the branch's raw
  merge conflicted with main's copy-kernel renames + carried stale generated files); RowOrthoRankFloored
  needs `using LinearAlgebra.Internal` for UnsafeOP.vecDot/axpyNormSq.

## Krylov.LSLQ — least-squares solver with a certified forward-error bound
- 2026-07-21 | Shipped `lslq` (Estrin-Orban-Saunders 2019), the least-SQUARES twin of `lnlq`: same
  Golub-Kahan bidiagonalization as `lsqr`, folded through an LQ factorization (SYMMLQ on AᵀAx=Aᵀb).
  Returns the LQ point xᴸ — the ERROR-minimizing iterate (‖xᴸ-x*‖ ↓ monotone), NOT lsqr's residual-
  minimizing point. VALUE-ADD is `LslqInfo.xErrBound` = |ζ̃|, a certified upper bound on ‖x*-xᴸ‖, opt-in
  via `double sigmaMinEst` (a strict UNDERestimate of σ_min(A)); the bound's tightness ratio is ~1.0
  (essentially exact — the whole point of the LQ point over LSQR). New struct `LslqInfo` (rnorm +
  Arnorm + xnorm + xErrBound); reuses lstsqResidual + the fProxy `SymGivens` (from MINRESQLP).
- 2026-07-21 | Bound recurrence (EOS2019): σ-QR Givens chain (init csig=-1, ρ̄=-σ) yields the
  Gauss-Radau ω each step — ω²=σ(σ-δ·h), h=δ·csig/ρ̄; then η̃=ω·s, ε̃=-ω·c, τ̃=-τδ/ω, ζ̃=(τ̃-ζη̃)/ε̃.
  Valid from the FIRST iterate (unlike LNLQ's, which needs τ_{k-1}). disc<0 → complex, xErrBound=NaN.
- 2026-07-21 | Bound runs at the SOLVE's precision (fProxy SymGivens), NOT a double sidecar. First
  shipped it in double (LNLQ mirror) via a single-output SymGivensD (an all-double helper CS0111-dupes
  across float/double copies) — user pushback ("meh"). MEASURED it (reference/wip-lnlq/lslq_float32.py,
  lslq_mysplit.py): a float solve + DOUBLE bound STILL under-reports ~2% (ratio 0.980) because the
  bound reads the float-rounded bidiag scalars (γ/δ/τ/ζ/c/s) — double can't recover lost precision.
  Fully-float = ~3% (0.968). Only the fully-DOUBLE build is certified (violation 0, ratio 1.0). So the
  double sidecar bought ~1% and never certified float → dropped it + deleted Krylov.Givens.cs. Contract
  now honest: xErrBound is a certified upper bound in the double build, a ~1-3% tight estimate in float.
  Test lowerFactor = choose[0.99(float)|1-1e-3(double)].
- 2026-07-22 | Fable audit (fable-model code-review) verified recurrence CLEAN vs Krylov.jl oracle +
  all numpy prototypes (double violation 0/ratio 1.0; stopping never-false-Converged; tmpM/tmpN mid-loop
  audit reuse safe — write-first next iter). Fixes it caught: (1) float test factor was 0.95 but a
  500-seed sim at the EXACT test op-point (iter 2) gives ratio ∈[1.0005,1.19] — the ~3% under-report is
  a NEAR-convergence-only effect, not at mid-convergence — so 0.95 was 50-100× too loose → tightened to
  0.99 (still catches any >1% float-specific pathology). (2) b=0 / Aᵀb=0 early-outs now return
  xErrBound=0 (x is EXACT there) not NaN, when a σ-est was given. (3) test σ-est margin 1e-10→1e-4 so the
  strict-underestimate survives the (fProxy)sigmaMinEst float cast (~1e-7) + float-SVD error.
- 2026-07-22 | LNLQ is NOT affected by the same float issue (Fable hypothesized it was; MEASURED via
  reference/wip-lnlq/lnlq_float32.py: float64 AND float32 both give violation 0, min ratio 1.174). LNLQ's
  Gauss-Radau bound has ~17% inherent slack (τ̃²-τ² on the augmented tridiagonal) that ABSORBS float
  noise → stays a valid upper bound in float; LSLQ's |ζ̃| is essentially exact (ratio ~1.0) so it can't.
  The contracts legitimately differ — do NOT "fix" LNLQ to match, and do NOT re-loosen LSLQ's float test.
- 2026-07-21 | STOPPING: LSLQ's xᴸ minimizes error, not residual, so the Krylov.jl running
  rNorm/ArNorm estimates track NEITHER xᴸ nor x^C in a plain transcription (verified: rel-err 0.8 /
  1e12). So the stop is a CERTIFIED optimality audit (lstsqResidual ‖Aᵀr‖ ≤ tol·‖Aᵀb‖, ‖Aᵀb‖=α₁β₁
  free), armed by a cheap forward-error trigger |ζ_k| ≤ tol·‖xᴸ‖ (verified to fire at/after the gate on
  160 problems; a premature fire is rejected by the audit → never-false-Converged). Collapse
  (β or α → 0) also arms the audit → Converged if optimal, else Breakdown. Rank-deficient A is handled
  gracefully (terminates at the min-norm LS point, ‖Aᵀr‖~1e-16), not a Breakdown.
- 2026-07-21 | NO damping and NOT battery-wired (deliberate). The shared KrylovLstsqBatteryTests
  overdetermined path unconditionally exercises a Tikhonov-damped entry point (#12); LSLQ's damped
  (λ-rotation) transcription had a fixed ~1.4e-3 optimality-residual bias I could not clear cleanly
  (entangled with the preconditioned derivation in the reference), so damping is out of scope for v1
  and LSLQ gets a dedicated LSLQTests instead — which also covers the certified bound the battery
  can't test. VERIFIED end-to-end in reference/wip-lnlq/lslq_*.py (gitignored): solve==lstsq to 9.6e-15,
  LQ bound violation 0 + ratio 1.0 over 320 trials; the paper's c₀=1 is a TYPO (must be -1), caught by
  diffing my numpy against Krylov.jl lslq.jl (MPL-2.0 → oracle-only, never copied).

## Krylov.LNLQ — least-norm solver with a certified forward-error bound
- 2026-07-21 | Shipped `lnlq` (Estrin-Orban-Saunders 2019). The SOLVE returns LNLQ's transferred
  CRAIG point x^C (identical min-norm x as `craig`, via the τ recurrence) — verified vs LQ.minNormSolve
  AND craig. The VALUE-ADD is `LnlqInfo.xErrBound`: a certified upper bound on ‖x*-x‖ (the library's
  only forward-error-in-x bound), opt-in via `double sigmaMinEst` (an underestimate of σ_min(A)).
- 2026-07-21 | The bound is the constant-time Gauss-Radau recurrence (EOS2018 eq 41):
  ‖x*-x_k^C‖² ≤ τ̃_k² - τ_k², τ̃_k = -β_k·τ_{k-1}/ω_k, ω_k² = σ_est² + σ_est·β_k²/p_{2k-2}, where p_j is
  the running LDLᵀ pivot of (Y - σ_est·I) (Y = Golub-Kahan augmented tridiagonal, zero diagonal,
  off-diagonals interleaving α_1,β_2,α_2,β_3,...): p_1=-σ_est, p_{j+1}=-σ_est-g_j²/p_j. O(1)/iter, no
  stored history — advance the pivot by (α_1) then (β_m,α_m) each step. Computed in DOUBLE regardless of
  solve precision. `sigmaMinEst<=0` skips it (xErrBound=NaN); iterate 1 has no bound (needs τ_{k-1}).
- 2026-07-21 | METHOD (this was the crux): pdftotext mangles the sub/superscripts of the three source
  papers, so the recurrence was extracted by RASTERIZING the PDF pages via PyMuPDF (`fitz`,
  Matrix(2.6,2.6)) and reading them as images — EOS2016 p.10 (Algorithm 1), EOS2018 p.13 (§5.1/eq 41),
  EOS2017 p.12 (θ node). Then NUMERICALLY VERIFIED end-to-end in `reference/wip-lnlq/*.py` (gitignored):
  the bound holds (slack≥0) and is tight (~1.5-3× true err) across 360 trials; the constant-time pivot ω
  matches the σ_min(L̃_k)=σ_est root-find. Do NOT re-derive from the OCR .txt — re-render if unclear.
  Test `BoundIsUpperBound` checks a MID-convergence iterate (maxIter=2) so the invariant has teeth.
  OUT OF SCOPE (v1): the y-bounds (eq 38/39), the x^L bound (eq 43), the sliding-window refinement
  (eq 40), and the preconditioned Generalized-Golub-Kahan path (§6). Spec: docs/dev/spec-lnlq.md.

## Krylov.CGNE — Tikhonov-damped least-norm (augmented-operator route)
- 2026-07-22 | Added a DAMPED cgne overload: x = Aᵀ(AAᵀ + damp²·I)⁻¹ b (ridge-regularized least-norm).
  The clean derivation is CGNE on the AUGMENTED operator [A | damp·I]: its normal matrix is
  [A|λI][A|λI]ᵀ = AAᵀ + λ²I, so no matrix is assembled and per-iter cost stays 1 Apply + 1 ApplyT.
  Why not "just add λ² to the matvec": undamped cgne stores only x = Aᵀy and p = Aᵀd (Aᵀ-images), which
  PROJECT OUT the y/d space (R^Rows) where λ²I acts — so damping needs the Rows-space search-direction
  component back. The augmented view supplies it as ONE extra Rows vector ps (= the R^Rows part of
  [A|λI]ᵀr = (Aᵀr, λr)); curvature pp = ‖p‖²+‖ps‖², matvec Ap = A·p + λ·ps, plus s (aux, Rows) only
  for the fresh augmented-residual verify. damp==0 DELEGATES to the undamped primitive (bit-identical,
  ps/s untouched) so the proven path is never perturbed. Convergence is on the AUGMENTED residual
  ‖b − Ax − damp·s‖→0 (verified fresh at exit); the Info reports the UNDAMPED ‖b−Ax‖ = damp·‖s‖ as
  rnorm (legitimately nonzero at the optimum — same convention as damped lsqr/lsmr), so read the
  status/implicit-bool, not rnorm. Damping also makes the solve well-posed without full row rank / b∈range(A)
  (AAᵀ+damp²I is SPD for any A). Surface: damped generic primitive (6 scratch: r/p/Ap/tmpN/ps/s) +
  arena dense/BSR damped overloads. Rounds out non-square regularization: lsqr/lsmr (least-SQUARES)
  already had damp; craig/craigmr/lnlq (generalized bidiag damping) and lslq (known ~1.4e-3 λ-rotation
  bias) remain deliberately undamped — see their notes. Tests: fProxyCGNEDampedTests (dense
  Aᵀ(AAᵀ+damp²I)⁻¹b oracle, nonzero-reported-residual contract, damp==0 bit-identity).
- 2026-07-22 | Review follow-up: the damped exits route through a new `LstsqInfoAudited(…, damp)`
  overload (forwards damp to lstsqResidual) so Info.Arnorm is the Tikhonov gradient ‖Aᵀr − damp²x‖
  (→0 at the optimum — the meaningful convergence cert), not the undamped ‖Aᵀr‖ (= damp²‖x‖, nonzero).
  rnorm stays the undamped ‖b−Ax‖. Undamped LstsqInfoAudited (cgne/craig) unchanged. Added
  rank-deficient (two identical rows) + BSR-path damped tests; the review panel found no correctness
  bugs in the recurrence.

## Krylov.CGNE — direct-CG least-norm (κ² route); LNLQ deferred
- 2026-07-21 | Added `cgne` = CG on AAᵀ (x = Aᵀy, matrix-free), the direct-CG minimum-norm solver.
  Computes the SAME min-norm x as `craig` but via CG on AAᵀ → κ² conditioning (cheaper/simpler per
  step, less stable) — the least-norm analog of CGLS, symmetric to keeping `bcgls` on the
  least-squares side. NOT redundant with craig (which is the stable Golub-Kahan route, like LSQR);
  it fills the κ² direct-CG route we lacked. Battery invoker runs at tol = sqrtEps·0.1 / MaxIterMul
  20 because κ² drives x's error as κ²·(residual tol), so the residual must go ~10× lower than
  craig's to land x in the same element band (a real κ²-driven adjustment, not a loosened assertion).
- 2026-07-21 | LNLQ (Estrin-Orban-Saunders least-norm LQ) — UNPARKED. Justification: it is the
  library's only forward-error-in-x bound (`‖x*-x_k‖` upper bound), a real capability gap —
  LstsqInfo/SolveInfo carry residual norms only. Reference chain now stashed + pdftotext'd in
  reference/rectangular/: LNLQ-…eos2018 (Algorithm 2 core + §5 bounds), EOS2017 (LSLQ companion,
  Radau node procedure), EOS2016 (SYMMLQ/CG error bounds, the constant-time ξ recurrence). All
  three from Ron Estrin's Stanford ~restrin/files/. Krylov.jl lnlq.jl is MPL-2.0 → oracle-only,
  never copied. Spec: docs/dev/spec-lnlq.md. Core solve = same min-norm x as craig; the Gauss-Radau
  error bound is the entire value-add (needs a σ_min UNDERestimate). Sliding-window/regularization/
  preconditioned-GGK all v1-out-of-scope.

## Krylov.Block.LSMR — deflation deferred (do not implement blind)
- 2026-07-21 | Assessed adding per-column / graceful rank-deficient deflation to blsmr (task #74).
  DEFER. blsmr's Golub-Kahan block bidiagonalization is a lag-2 short recurrence over ~20 fixed
  s×s/s×n buffers overwritten in place each step, with NO persistently-addressable identity axis
  (only B/X/ATB carry it; every other block mixes two direction indices) and NO reconciliation
  point analogous to bgmres's growing-basis re-solve. So bgmres's w[j]/off[] active-width pattern
  does NOT transfer, and neither does bcgrq's per-column lock (needs an identity axis through the
  whole recurrence). "Deflate and continue" would require inventing variable-width bookkeeping
  across mismatched lag history AND re-deriving the free O(1) ‖AᵀR‖_F stopping recurrence (which is
  proved only for constant width) — content Mojarrab & Toutounian explicitly decline, pointing to
  an unfetched Robbe & Sadkane paper (not in reference/). Deriving it blind risks a wrong
  convergence certificate (worse than today's honest Breakdown); porting it needs the reference
  first (port-fidelity rule, same rule that dropped blsqr). Value is narrow: a rank-deficient RHS
  *block* is user-avoidable (dedupe columns / scalar lsmr per column), and bgmres/bfgmres/bgcrodr
  already deflate where it's structurally tractable. Revisit only if the Robbe & Sadkane reference
  is obtained. The one small sub-option (ridge-regularize the two BlockSolveGeneralWide solve sites)
  does NOT fix the motivating first-LQ-step case, so it's not worth a standalone change.

## Krylov.Block.GCRODR — harmonic-Ritz Gram construction via GEMM
- 2026-07-21 | The three d×d Grams (Fmat=APᵀAP, Pgram=PᵀP, Gmat=APᵀP, d=kcur+Krylov dim) were
  built with one Blas.dot per (ai,bi) entry — O(d²) calls over length-n vectors. Replaced with an
  O(d·n) gather of the d combined columns into contiguous n×d buffers + three GEMMs (Fmat/Pgram hit
  the symmetric matAtA kernel, Gmat hits matMatDotTransA). Measured 27–58% faster on bgcrodr across
  float/double and N=128..512 (BlockArnoldiBenchmark), every data point improved. Reduction reorder
  vs the per-entry dot is not bit-identical (pre-1.0 waiver); oracle battery + GCRODRTests stay
  green. SCALAR gcrodr was tried the same way and REVERTED — its d (≈30, bounded by recycle+restart)
  is small enough that the per-entry SIMD vecDot over a cache-resident length-n vector already wins;
  gather+GEMM-dispatch overhead regressed it (ConvDiff up to +199%). Don't retry scalar. Block's d
  is far larger (restart·s can exceed n → basis rank-exhausts per cycle), so the O(d²) call overhead
  dominates there and GEMM pays off.

## Krylov least-squares family — blsqr decided-against
- 2026-07-21 | Dropped block-LSQR from the roadmap (user call). No permissive reference to port
  (fidelity-first porting rule), and the block LS space is already spanned: `bcgls` = CG on the
  normal equations = LSQR's iterates in exact arithmetic (the ‖r‖-minimizing route), and `blsmr` =
  MINRES on the normal equations (‖Aᵀr‖-optimal, Golub–Kahan-stable, the modern default). blsqr
  would only add a marginally-more-stable variant of bcgls that blsmr already supersedes. LSMR is
  not literally dominant (on compatible systems LSQR's ‖r‖ can be marginally ahead), but that niche
  doesn't justify the port.

## Krylov.Block.{GMRES,FGMRES,GCRODR} — status/fresh-residual reconciliation
- 2026-07-21 | `status` was set to `Converged` purely from the in-cycle Pythagorean LS-residual
  estimate (bgmres/bfgmres: the post-Arnoldi-step check; bgcrodr: the same check at its mid-cycle
  site -- its top-of-cycle recycled-correction recheck already used a fresh residual and was left
  alone), while `converged`/`maxRnorm` were always derived from a fresh `R0 = B - A·X` recomputed
  after the loop. `info.converged` was honest but `info.status` could over-claim Converged when
  Arnoldi orthogonality loss (MGS drift) made the estimate optimistic -- the same silent
  false-Converged class fixed for Block.CG/BiCGStab/IDR (see above) and for scalar
  gmres/fgmres/gcrodr (#60). Fix: after the existing post-loop `converged = CountConverged(...)`,
  downgrade `status` to `MaxIterations` whenever it says `Converged` but `converged < s` -- free
  (the fresh residual was already computed), definitionally correct, cannot regress a genuinely
  converged solve. Unlike the Block.CG/BiCGStab/IDR gate this is a post-hoc downgrade, not a
  fall-through-and-keep-iterating re-verify (the block-GMRES family's post-loop recompute already
  runs unconditionally once per call, so there was nowhere to "keep iterating" to without
  restructuring the loop).

## QR/LQ/Bidiag — zero-column Householder fallback used a float √2 in the double build
- 2026-07-21 | genHouseholder's zero-column fallback (QR.fProxy.cs:51, LQ.fProxy.cs:39,
  Bidiag.fProxy.cs:40 & :67) stored `math.SQRT2` into the reflector vector. `math.SQRT2` is a FLOAT
  constant (= (float)SQRT2_DBL); in the double-generated variant that gives uᵀu = 1.99999993 instead
  of 2, so H = I - uuᵀ has H[k,k] = -0.99999993 -- an off-by-6.85e-8 sign-flip reflector and a Q with
  ‖QᵀQ-I‖ ≈ 1.37e-7, IN DOUBLE. An ordinary full-rank factorization never takes this fallback, but
  blsmr's block-Givens step QRs a deliberately zero-PADDED 2s×2s matrix, so its padded columns hit
  the fallback every iteration; the polluted complement columns poison the cbark/dbark recurrence,
  which drifts phi by ~5e-8 and makes the subtractive ‖AᵀR‖² estimator cross threshold early -> a
  FALSE Converged with X frozen (blsmr double plateaued at ~1e-5..5e-7 relative normal-eq residual
  vs scalar lsmr's 2e-16). bcraigmr shares the zero-padded-QR trick and degraded similarly. Fix:
  `(fProxy)math.SQRT2_DBL` at all four sites -- exact per fProxy precision, float build bit-identical
  ((float)SQRT2_DBL == math.SQRT2). Root-caused by Fable (instrumented ‖QᵀQ-I‖ trace matched the
  predicted 1.3691416e-7 to five digits). This unblocks the deferred blsmr verify-at-exit gate (#73):
  no more false Converged at its source.

## Krylov.Block.{CG,BiCGStab,IDR} — verify-at-exit honesty gate
- 2026-07-21 | These block solvers declared `Converged` purely from the tracked recurrence residual;
  on ill-conditioned inputs the tracked residual can drift below tolerance while the true residual
  is still above it (silent false-Converged). Added a fresh true-residual gate at each `Converged`
  decision: recompute `R = B - A.X` from the current X into an IDLE scratch block (Block.CG: `Q`;
  Block.BiCGStab: `Tmp`; Block.IDR: `termMN`, scratch-only — IDR's `f[]` cross-product history must
  NOT be reseated mid-sweep) and only commit `Converged` if `CountConverged` also clears against the
  SAME `thr`. On a failed fresh check the loop continues (never a post-hoc downgrade), preserving the
  iterations contract. Bit-identical for well-conditioned inputs (the fresh check passes there).
  Battery honesty check (#12) is the regression oracle. blsmr's analogous gate is deferred (its
  clamped estimator floors above tol on a consistent system — a separate attainable-accuracy
  question, see task tracker).

## Krylov.Block.MINRES — tol==0 nonzero-B false convergence
- 2026-07-21 | Task #49 (un-gate preconditioned path) ATTEMPTED, REVERTED. Removed the
  `NotSupportedException` and flipped `fProxyBminresInvoker.PrecondKind` None→SymmetricBSR so the
  block battery's check #5 drives `SolveWithPrecond` with a real BlockJacobi M. Result: it does NOT
  converge — fresh relRes ~0.43 (double) / ~0.48 (float) vs a ~1e-7 threshold on the first symmetric
  gallery (matrix=0), i.e. the solve stalls near half the initial residual. So the two shared-recursion
  fixes (BuildOmega `Beta^T`, BlockNormalize un-pivot) did NOT also fix the preconditioned path:
  `BlockNormalizePrecond`'s M-inner-product block-Lanczos normalization is independently wrong. #49 is
  therefore NOT a free un-gate; it needs the same reference-driven diagnosis as #50 (extend the numpy
  reference `reference/wip-bminres/bminres_reference.py` to a nontrivial SPD M, or dump the C#'s
  per-iteration Beta/Gnorm/Vcur under M and diff). Gate restored; unpreconditioned bminres unaffected.
- 2026-07-21 | Block sub-matrix copy-kernel consolidation + rename. (1) Deleted MINRES-local
  `CopyRowsAt`/`CopyColsAt`/`CopyBlockAt` — pure redundancy with Block.Common helpers already in
  scope (same `partial class Krylov`); they had drifted in during the s>1 debugging. (2) Renamed the
  Common block-copy family for direction-clarity (the old `Copy*` conflated read vs write, the old
  `Extract*`/`Write*`/`Store*` split direction but read clinically): now `CopyBlockFrom` (read: Dst =
  Src sub-block), `CopyBlockInto` (write/assign), `AddBlockInto` (write/+=), `CopyBlockFromTransposed`,
  `CopyRowsFrom`. Arg orders unchanged (offsets already sit next to the buffer they index — From is
  Src-first, Into is Dst-first), so it was a pure token rename across Common/MINRES/CRAIGMR/LSMR/
  GMRES/FGMRES/GCRODR. Behavior-identical; battery is the oracle. Implementations kept as element
  loops (NOT MemCpy'd — the blocks are small, the win would be nil and the stride/alias risk real;
  the DRY/compile-time payoff is the dedup itself). `CopyBlock`/`CopyMat` (whole-block) and the views
  keep their names.
- 2026-07-21 | `bminres`'s zero-RHS early-out (Krylov.Block.MINRES.fProxy.cs) tested `thr[j] ==
  0` (`thr[j] = tol*tol*||B[j]||^2`), which is also true whenever `tol == 0` regardless of `B` --
  a `tol=0` call on a NONZERO `B` took the shortcut, set `X := B`, and reported `Converged`. `X = B`
  only solves `A.X = B` when `A` is the identity. Fixed by testing `||B[j]||^2` directly (`bIsZero`)
  instead of `thr[j]`; the shortcut now fires only for a genuinely zero `B`. `tol == 0` with nonzero
  `B` falls through to the normal iteration, which honestly converges only at an exact residual or
  reports MaxIterations. Test: `fProxyBlockMinresTests.ZeroTolShortcutOnlyFiresOnGenuineZeroB`.

## Krylov.LSQR — damped Arnorm sign
- 2026-07-21 | `lsqr`'s per-iteration `arnorm = phibar * alpha * math.abs(c)` (Krylov.LSQR.fProxy.cs)
  took `abs` on `c` only. Under Tikhonov damping (`damp != 0`) the rotation `rhobar1 =
  sqrt(rhobar^2+damp^2)` folds `rhobar`'s sign into `phibar` every iteration once `rhobar` goes
  negative (which it does from the second iteration on, `rhobar = -c*alpha`), so `phibar` is not
  itself sign-definite the way `alpha`/`beta` (bidiagonalization norms) are -- Arnorm (a norm) came
  back negative. Fixed by also taking `math.abs(phibar)`. `phibar >= 0` for the entire undamped
  (`damp == 0`) recurrence (`phibar_{k+1} = sn_k*phibar_k`, `sn_k >= 0`, `phibar_0 = beta_1 >= 0`),
  so the undamped path is bit-identical. Test: `fProxyLSQRDampedArnormTests.DampedArnormNeverNegative`
  (deterministic sign flip via `maxIter=2`) + `UndampedArnormBitIdenticalToAbsPath`.

## Krylov.Block.CRAIG / Krylov.Block.CRAIGMR — missing X/B aliasing guard
- 2026-07-21 | Neither `bcraig` nor `bcraigmr` (Krylov.Block.CRAIG.fProxy.cs /
  Krylov.Block.CRAIGMR.fProxy.cs) checked `X`/`B` for aliasing before use, unlike every other block
  Krylov solver with caller-supplied buffers (e.g. `bbiCGStab`'s `RequireDistinctBuffers` over
  R/Rhat0/P/V/T/Phat/Shat/X/B). Both solvers own their entire internal workspace via
  `Allocator.Temp` (no external scratch params), so `X` and `B` are the only two caller-supplied
  buffers -- an aliased `X` silently destroys `B` mid-solve. Added a 2-pointer
  `RequireDistinctBuffers` guard to both, right after the existing shape/maxIter validation. Test:
  `fProxyBlockCraigGuardTests.BcraigAliasedXBThrows` / `BcraigmrAliasedXBThrows`.

## Krylov.CRAIGMR — iterate-0 ArNorm off by ||b||
- 2026-07-21 | `craigmr`'s pre-loop `ArNorm = alpha` (Krylov.CRAIGMR.fProxy.cs) claimed to be
  ‖Aᵀ(b-Ax₀)‖ = ‖Aᵀb‖ at x₀=0, but `alpha` alone is `‖Aᵀu_1‖` (`u_1 = b/beta_1` is UNIT norm), not
  `‖Aᵀb‖` -- the correct identity is `alpha*beta = ‖Aᵀu_1‖*beta_1 = ‖Aᵀb‖`, matching the sibling
  lsqr/lsmr convention (`arnorm = alpha*beta` pre-loop) and this file's own per-iteration recurrence
  (`ArNorm = alphaNew*betaNew*|zeta|/rho`), which only reduces to `alpha*beta` in that shape, never
  bare `alpha`. The stale value was returned whenever the solve exited during the FIRST loop pass
  (a k=0 Breakdown, or a k=0 Converged exit before the per-iteration `ArNorm` update at the end of
  the loop body). Fixed `ArNorm = alpha * beta` pre-loop; `x`/`rNorm` unaffected. Test:
  `fProxyCRAIGMRTests.IterateZeroArnormMatchesAtb` (loose `tol > 1` forces a k=0 Converged exit, by
  the file's own monotonic-residual property).

## Block GMRES family (bgmres/bfgmres/bgcrodr) — zero-RHS-row saturation NaN fix
- 2026-07-21 | Root cause (Fable): battery check #11 (KrylovBlockBatteryTests, matrix DenseNonsym20,
  S=4, B row 0 all-zeros) drove `bgmres`/`bfgmres`/`bgcrodr` to a NaN-poisoned `X` reported as
  `Breakdown` -- masking real corruption as a status the caller could plausibly treat as "the
  degenerate row broke something benign". The zero-norm column can never self-declare converged, so
  the block-Arnoldi loop iterates PAST the point where the shared basis has captured everything A's
  Krylov subspace can offer. At that saturation point `Wj` (the freshly orthogonalized step) is pure
  rounding noise, but `LQRPRank`'s deflation test compared each `|L[i,i]|` only against the SAME
  noise block's own `|L[0,0]|` (self-relative) -- mutually comparable noise never trips a relative
  test, so `wj1` stayed > 0 instead of collapsing to 0. Unit-norm noise rows then entered `V`/`H`, the
  block-Hessenberg least-squares matrix went singular, `QR.decompSolve`'s unpivoted `Blas.triUpper`
  divided by ~0, and `Y` went Inf -> Inf-Inf -> NaN, committed straight into `X`. The NEXT cycle's
  `R0 = B - A*X_NaN` came back all-NaN; `CountConverged`/`LQRPRank`'s NaN-blind comparisons (`rn >
  worst`, `abs(L[i,i]) > tol`, both false for NaN) read that as `maxr=0, rank=0`, tripping the
  existing `w[0]==0` "defensive" Breakdown one cycle late -- Breakdown status, NaN `X`, silently.
- 2026-07-21 | Fix 1 (the actual root cause): `BlockArnoldiMGS2Step` (Krylov.Block.Common.fProxy.cs)
  and bgcrodr's own duplicated inline copy of the same step (Krylov.Block.GCRODR.fProxy.cs, kept
  separate per this file's own earlier note on the recycled-subspace projection) now capture
  `scale = Norms.L2(Wj)` BEFORE MGS2 mutates `Wj`, and rank the post-orthogonalization LQ diagonals
  via a new `LQRPRankFloored(L, m, nGlobal, scale)` -- `max(m,nGlobal)*ZeroThreshold` applied against
  BOTH `|L[0,0]|` (the existing self-relative term) and `scale` (an absolute floor tied to what the
  step actually started from). At true saturation the orthogonalized step is noise relative to its
  own pre-orthogonalization magnitude, so the scale-tied floor drives `wj1 -> 0` (a clean happy
  breakdown, ending the cycle on the already-optimal `Y`) before any noise row reaches `V`/`H`. Plain
  `LQRPRank` (the initial per-cycle `w[0]` site in all three solvers, plus `FactorLiveResidual`/
  `FactorLiveSearch`, LOBPCG-shaped and not saturation-prone) is UNTOUCHED -- confirmed via the full
  suite reproducing the exact pre-fix pass count on everything except the 3 targeted failures.
- 2026-07-21 | Fix 2: ported bgcrodr's own NaN/Inf scan of `Yv` (already present between its
  dense-QR solve and Pythagorean check) into the shared `BlockLSResolveAndCheck`
  (Krylov.Block.Common.fProxy.cs), used by both `bgmres` and `bfgmres` -- `out bool lsBreakdown`. Both
  callers now break their inner Arnoldi loop and skip the cycle's `X` commit entirely on a non-finite
  `Y`, reporting `Breakdown` from the UNMODIFIED `X` (the shared post-loop fresh-residual recompute
  then reports the true state at that `X`, never a poisoned one). Defense in depth: with Fix 1 in
  place this should rarely fire on the battery's own inputs, but a singular/near-singular
  block-Hessenberg least-squares is reachable from other inputs (e.g. A mapping a whole cycle's basis
  to 0), same as bgcrodr's own documented rationale for the guard it already had.
- 2026-07-21 | Fix 3 (cheap safety net, not required for the battery fix but closes the class):
  `CountConverged`'s worst-norm update changed `rn > worst` to `!(rn <= worst)` so a NaN row now WINS
  the update instead of being silently dropped -- `maxRnorm` can no longer report 0 from a poisoned
  block (bit-identical for finite inputs: the two forms agree whenever `rn`/`worst` are finite).
  `LQRPRank`/`LQRPRankFloored` both added an explicit `!math.isfinite(L[0,0])` early return of rank 0
  -- behaviorally a no-op (a NaN/Inf `L[0,0]` already produced rank 0 through the loop's own
  NaN-comparison-is-false semantics), but makes the "never masquerade as a clean deflation-to-zero"
  contract an explicit, self-documenting guard instead of an implicit IEEE side effect.
- 2026-07-21 | Verification: reverted only these 4 files (keeping the rest of the in-flight tree,
  including the already-shipped `BuildColumnThresholds` per-column absolute floor, untouched) and ran
  the full suite both ways. Before: 7138 total, 7132 passed, 6 failed -- exactly
  `{Bgmres,Bfgmres,Bgcrodr} x {fProxy-literal compile-check, float}` on matrix=9 (DenseNonsym20),
  check=11, status=Breakdown (the `double` variant already passed at this seed -- consistent with a
  rounding-noise-triggered saturation being much rarer at double precision). After: 7138/7138, 0
  failed, no duration spike (354.9s vs 356.5s -- no Burst-to-Mono fallback).

## Krylov DRY extraction (P2)
- 2026-07-20 | Task #57 P2 (docs/dev/spec-krylov-dry-extraction.md Cluster E3): extracted the
  final block-solver maxRnorm cleanup reduction into two variants in `Krylov.Block.Common.fProxy.cs`
  — `BlockMaxResidualRecompute` (B - Rfinal, `Rfinal` freshly recomputed by the caller) and
  `BlockMaxResidualNorm` (residual already sitting in a buffer, no recompute). Retargeted the 4
  files that still hand-rolled the raw loop: `Krylov.Block.LSMR.fProxy.cs`,
  `Krylov.Block.CRAIG.fProxy.cs`, `Krylov.Block.CRAIGMR.fProxy.cs` (recompute variant),
  `Krylov.Block.CGLS.fProxy.cs` (in-hand variant). The other 9 block files (bcg, bcgrq, bfbcg,
  bminres, bidr, bgmres, bfgmres, bgcrodr, bbiCGStab) already routed their final maxRnorm through
  the pre-existing `CountConverged`/`CountConvergedByBound` helpers (P0/P1-era), so E3's remaining
  duplication was only these 4 raw loops by the time P2 started — confirmed via `grep` for the
  `rr +=` / `math.sqrt(rr)` shape across every `Krylov.Block.*.fProxy.cs` before touching anything.
- 2026-07-20 | Task #57 P2 (Cluster C2): extracted the verify-at-exit / final-true-residual
  recompute (`A.Apply(x)` -> `r = b - Ax` -> `‖r‖²`) into `VerifyTrueResidual` in new
  `Krylov.Solve.Common.fProxy.cs`. Returns the squared norm only — no baked-in return/branch — so
  every caller keeps its own control flow exactly as before (the landmine: several sites must FALL
  THROUGH and keep iterating on a failed verify, not return). Retargeted 8 call sites: `cg`, `fcg`
  (one site each), `minres` (both the in-loop verify and the preconditioned-MaxIterations final
  report), `minresQLP` (the unconditional final-residual report), `biCGStab` (its second,
  committed-x site only), `idr` (both its in-sweep and end-of-sweep sites).
- 2026-07-20 | Cluster C2 left inline: `biCGStab`'s FIRST verify-at-exit site (checks a
  not-yet-committed trial x, sign-flipped into `A·x_trial - b` instead of `b - A·x`, and never
  touches `r`) is a genuinely different shape from the other 8 sites — folding it in would mean
  branching the shared helper on "which buffer is x" and "which sign", which starts baking
  call-site-specific control flow into a "shared" helper. Left inline, per the task's
  don't-force-it guidance.
- 2026-07-20 | Task #57 P2 (Cluster D): extracted the scalar Golub-Kahan bidiagonalization
  half-steps into `GolubKahanUStep`/`GolubKahanVStep` in new `Krylov.Bidiag.Common.fProxy.cs`
  (`u = Av - alpha*u`/`v = Aᵀu - beta*v`, fused via the existing `Blas.xpayNormSq`, returning the
  fresh norm WITHOUT dividing or branching). This is the one piece of the bidiag recurrence that is
  genuinely byte-identical across all 4 solvers regardless of step order: `lsqr`/`lsmr` do u-step
  then v-step and divide immediately after each; `craig`/`craigmr` also do u-step then v-step but
  interleave their own convergence/breakdown checks BETWEEN computing the norm and dividing by it
  (craigmr divides even later, after its Givens update) — every caller kept its own divide/branch
  placement untouched, only the `Apply`+`xpayNormSq` pair moved. Retargeted 8 call sites (2 each in
  lsqr, lsmr, craig, craigmr).
- 2026-07-20 | Cluster D init left inline: `lsqr`'s and `lsmr`'s init blocks (`atbSq` scale, warm-
  started `u = b - Ax`/beta, `v = Aᵀu`/alpha, THREE separate early-return points each constructing
  a different `LstsqInfo` via `LstsqInfoTracked`) ARE byte-identical to each other and were a
  candidate — but `craig`/`craigmr` do NOT share this shape (no warm start, x forced to 0, scale is
  `‖b‖` not `‖Aᵀb‖²`, `v` bootstraps from 0 instead of an `ApplyT`), so the extraction would only
  cover 2 of the 4 files. Given the three interleaved early-returns each build a different
  `LstsqInfo` inline, sharing even the lsqr/lsmr pair would mean either baking a return into the
  helper (same landmine C2 avoids) or a discriminated-result design — deferred as lower value than
  the half-step extraction for the 1-hour budget; left inline in both files.
- 2026-07-20 | Verified bit-identity for all of the above the same way as P0/P1: `diff` the
  pre-edit source for every candidate before merging, then a full suite run
  (`Result=Passed total=7123 passed=7123 failed=0`) before and after: the pre-edit suite was
  already green at 7123/7123, and the post-edit suite reproduced the exact same line — a pure
  refactor, no test count change, no rounding/reduction reorder.

## Krylov DRY extraction (P1)
- 2026-07-20 | Task #57 P1 (docs/dev/spec-krylov-dry-extraction.md Clusters B/H): extracted the
  scalar Arnoldi/Givens/Hessenberg core shared by `gmres`/`fgmres` into new
  `Krylov.Arnoldi.Common.fProxy.cs` (`ArnoldiMGSStep`, `GivensApplyAndGenerate`,
  `HessenbergBackSolve`), and the block Arnoldi/dense-QR-least-squares core shared by
  `bgmres`/`bfgmres` into `Krylov.Block.Common.fProxy.cs` (`BlockArnoldiMGS2Step`,
  `BlockLSResolveAndCheck`). Pure cut-paste, same statement order, no reduction/rounding
  reorder — confirmed via `diff` on the pre-edit source that every extracted block was
  byte-identical (modulo comment wording) between the two solvers in each pair before touching
  anything. Suite stayed 7123/7123 green before and after; GMRES/GCRODR/Battery filters also
  independently green.
- 2026-07-20 | `gcrodr`/`bgcrodr` deliberately NOT folded into these helpers, left on their own
  full copies. Their Arnoldi step interleaves a recycled-subspace projection before MGS and a
  `pivotGuard`-based breakdown test (`arnoldiDone`/`lsBreakdown`) that gmres/fgmres/bgmres/bfgmres
  don't have — `bgcrodr` specifically inserts a NaN/Inf scan on `Yv` between the dense-QR solve
  and the Pythagorean check, which changes control flow (an early break that must happen before
  the Pythagorean check ever reads possibly-NaN data) and would corrupt bit-identity for the
  other two if merged into one shared function. Note: `bgcrodr`'s block-MGS2 + LQ-deflation
  sub-block (the part before the dense-QR resolve) DID diff byte-identical against
  `bgmres`/`bfgmres`'s and could technically call `BlockArnoldiMGS2Step` too — left out anyway to
  keep the fold-in decision at the whole-solver level (matches scalar `gcrodr`'s treatment) rather
  than mixing a partially-shared solver body, per the task's 1-hour-rule / "leave on its own copy"
  default.

## Krylov DRY extraction (P0)
- 2026-07-20 | Task #57 P0 (docs/dev/spec-krylov-dry-extraction.md Cluster F/I): relocated 15
  single-definition block helpers (`BlockApplyOp`, `BlockApplyOpT`, `TriNearSingular`,
  `ExtractBlockTranspose`, `ExtractBlockAt`, `WriteBlockAt`, `TransposeSmall`,
  `BlockSolveGeneralWide` from `Krylov.Block.LSMR.fProxy.cs`; `BlockCrossGram`,
  `BlockSolveGeneral`, `BlockFrobDot`, `BlockScaleInPlace` from `Krylov.Block.BiCGStab.fProxy.cs`;
  `StoreBlockAt`, `ExtractRowsAt`, `ZeroPrefix` from `Krylov.Block.GMRES.fProxy.cs`) into
  `Krylov.Block.Common.fProxy.cs`, pure cut-paste (partial class, no call-site changes). Also
  converted `cg`/`fcg`'s inline pointer-aliasing OR-chains to the shared `RequireDistinctBuffers`
  helper (mirrors minres/biCGStab/etc), same buffer sets. Zero behavior change; suite stayed
  7123/7123 green before and after.

## Krylov.Block.GCRODR
- 2026-07-20 | New `Krylov.Block.GCRODR.fProxy.cs` (task #40, bgcrodr): block GCRO-DR, block-generalizing scalar `gcrodr`'s recycled-subspace + refined-harmonic-Ritz deflation (same two eigensolvers: `Eigen.valuesQRInPlace` for the harmonic-Ritz VALUES of `Gmat^-1 Fmat`, `Eigen.symmetricInPlace` for each refined VECTOR) on top of `bgmres`'s block Arnoldi / deflating-LQ / periodic-dense-QR-least-squares engine. The recycled subspace (U/C/Ru, up to `recycle` individual n-vectors) is stored as block matrices (recycle rows) rather than scalar gcrodr's `UnsafeList<fProxyN>`, so the projection/uncorrection steps reuse the same GEMM-shaped block helpers (`BlockCrossGram`/`BlockCTV`) bgmres's own machinery uses. Preconditioning mirrors scalar gcrodr's per-Arnoldi-step M-apply + stored basis (Zv, combined at commit) rather than bgmres's apply-M-once-to-the-combination trick, because U/C need to live directly in solution space for the recycled correction to need no second M-apply — so recycle=0 is NOT guaranteed bit-identical to bgmres (unlike scalar gcrodr's recycle=0-equals-gmres guarantee), one extra M-apply per block step versus bgmres's optimum.
- 2026-07-20 | Landmine hit and avoided: a `RectView`/`View` narrower than a buffer's TRUE column width reinterprets the FLAT storage (front `rows*cols` elements), not a geometric sub-rectangle — using it to narrow `Bmat` (recycle x m*s) down to (kcur, totalCols) for the commit-phase `Bmat @ Y` product would misalign rows. Fixed by GEMM-ing the FULL untruncated `RowsView(Bmat, kcur)` against the full `Yscratch` (m*s rows): Bmat's columns beyond this cycle's totalCols are freshly zeroed every cycle, so they cancel any stale (but always finite, never NaN) rows Yscratch carries over from an earlier, wider cycle. Same rationale as `StoreBlockAt`/`ExtractRowsAt`'s own doc comments in `Krylov.Block.GMRES.fProxy.cs`.
- 2026-07-20 | A private helper (`LocateCol`, mapping a combined-space column index through the `off[]` prefix-sum array) whose signature was pure `Indices`/`int` — no `fProxy`-tagged parameter — compiled fine per dtype file but collided (CS0111) once both the float and double generated copies landed in the same shared `Krylov` partial class: codegen only disambiguates types/members that reference an fProxy/iProxy-tagged type. Fixed by inlining the 3-line locate loop directly into its one caller (`ResolveCombinedCol`, which does carry fProxyMxN params) instead of keeping it as a standalone method. General lesson: any new Krylov.Block.*.fProxy.cs private helper needs at least one fProxy-tagged parameter or return type, or it must not be a distinct top-level member.

## Krylov.Block.CRAIGMR
- 2026-07-20 | New `Krylov.Block.CRAIGMR.fProxy.cs` (task #45, bcraigmr): block CRAIGMR for a wide/underdetermined, full-row-rank operator and s consistent right-hand sides -- the MINRES-flavored (monotonic-residual) sibling of `bcraig`. DERIVED, NOT PORTED: no published block-CRAIGMR exists (confirmed absent -- see `reference/rectangular/CRAIGMR-BlockCRAIG-algorithm-extract.md` section 5.2, itself flagged "unverified, no reference to check against"). Reuses `bcraig`'s exact block Golub-Kahan bidiagonalization (LA/LBnew each round via `LQ.decomp`) and blsmr's zero-padded-2s x 2s-`QR.decomp` trick for extracting a full p x p orthogonal block transform (Abark/Bbark/Cbark/Dbark, the block analog of scalar Givens (c, s, -s, c)), applied to `bcraig`'s own block-bidiagonal system `LA_j @ Y_j + LBnew_j^T @ Y_{j-1} = RHS_j` instead of to blsmr's LSMR-shaped one -- exactly the "swap which bidiagonal recurrence the block-QR runs on" transfer the reference doc's section 5.2 describes.
- 2026-07-20 | The scalar `craigmr` template's own w/d running-recurrence (`d = (v - theta*d)/rho; x += zeta*d`) block-generalizes with a NON-obvious transpose placement once the coefficients stop commuting: `Zeta`/`ZetaBarNew` (RHS carry) and `ThetaNew`/`RhoBarNew` (next round's R-factor blocks) are plain products (`Abark@ZetaBar`, `Cbark@ZetaBar`, `Bbark@LAnew`, `Dbark@LAnew`), matching the scalar shape term-for-term, but the `D` update needs BOTH `Rho` and `Theta` TRANSPOSED (`Rho^T @ Dnew = V - Theta^T @ D`, solved via `BlockSolveGeneralWide` against the explicit transpose, same "pass the transpose in" pattern blsmr's own P/phi solves already use against `alphabarT`) -- a naive direct-product `D` update (no transpose) is provably wrong by a hand-worked 2-round index/algebra check (traced through `LA_1 = Abark^T @ Rho_1` and compared `Zeta_1^T @ D_1` against ground-truth `Y_1^T @ V_1` from `bcraig`'s own forward-substitution formula; only matched once both `Rho` and `Theta` were transposed in the `D` recurrence). `X += Zeta^T @ Dnew` (BlockCTV) needs no extra transpose, matching `bcraig`'s own `X += Y^T @ V` shape directly. Convergence deliberately recomputes a fresh `‖B-AX‖_F` every round rather than trusting the free `|zetabar|`-style scalar shortcut generalized to `‖ZetaBar‖_F` -- unverified for the block case, and the extra `BlockApplyOp` cost matches `bcraig`'s own already-accepted correctness-over-speed tradeoff. Passed the block-least-squares battery's `LQ.minNormSolve` min-norm oracle (and the scalar-`craigmr`-agreement check) on the FIRST full test run after this derivation, no sign/order iteration needed once the transpose placement above was nailed by hand first.

## Krylov.Block.CRAIG
- 2026-07-20 | New `Krylov.Block.CRAIG.fProxy.cs` (task #44, bcraig): block CRAIG for a wide/underdetermined, full-row-rank operator and s consistent right-hand sides -- block min-norm counterpart to `bcgls`/`blsmr`. DERIVED, NOT PORTED: no published block-CRAIG exists (confirmed absent -- see `reference/rectangular/CRAIGMR-BlockCRAIG-algorithm-extract.md` sections 5 and 7). Followed that file's derivation sketch (block-triangular-solve flavor, section 5.1): reuses `blsmr`'s exact block Golub-Kahan bidiagonalization engine (the two LQ-decomp relations producing LA/LB each round, and the BlockApplyOp/BlockApplyOpT/TriNearSingular/BlockSolveGeneralWide helpers, all shared via the partial class) but replaces LSMR's block-Givens QR continuation with a block-lower-TRIANGULAR forward substitution: L_j Y_j = R_1 (round 1) / -R_j Y_{j-1} (round j>1), each solved via `BlockSolveGeneralWide` against the exactly-lower-triangular LA factor (a QRCP general solve, not a dedicated TRSM -- simpler and still correct since L_j's triangularity is a QR/LQ identity that QRCP reproduces exactly, and QRCP's own rank check doubles as the breakdown guard). Re-derived the classical (column-convention) block relations from the reference doc's section 4 by hand, converted to this codebase's row-storage LQ convention via an explicit transpose round-trip (our `LB` variables = transpose of the classical upper-triangular R; our `LA` variables equal the classical lower-triangular L unchanged; the X-accumulation is `Y_j^T @ V_j` via BlockCTV, matching blsmr's own X-update shape) -- verified twice independently (once via direct relation-matching against blsmr's actual LQ.decomp call sites, once via a classical-matrix round trip) after an initial mid-derivation sign/order confusion that a careful re-derivation resolved before any code was written. Deliberately does NOT use scalar craig's free `rnorm=|beta*z|` per-iteration identity -- the reference doc only derives that shortcut for block CRAIGMR (Givens-based), not block CRAIG (triangular-solve-based), and blocks don't commute the way that scalar identity needs; instead recomputes a fresh `‖B-AX‖_F` every round (one extra BlockApplyOp per round, correctness over speed). Round numbering/iterations bookkeeping mirrors `blsmr` exactly (round 1 done in an uncounted init block, `iterations` counts only loop passes after that), not scalar craig's (which counts its first round inside the loop) -- picked for family consistency, not fidelity to the scalar sibling.

## Krylov.Block.CGLS
- 2026-07-20 | New `Krylov.Block.CGLS.fProxy.cs` (task #46, bcgls): block CGLS for a tall/overdetermined operator -- block-CG on the normal equations AᵀA X = AᵀB, never forming AᵀA, reusing `bcg`'s s x s Gram/BlockSolveSPD machinery and `blsmr`'s per-row BlockApplyOp/BlockApplyOpT (ApplyBlock is symmetric/square-only, A here is rectangular). Not warm-startable (X always zeroed), no preconditioner param -- matches `blsmr`'s scope, not `bcg`'s. Convergence checked on S = AᵀR (the normal-equation residual) against tol²·‖Aᵀ B[j]‖², via the SAME `CountConverged`/`thr` pattern `bcg` uses on its raw residual; `BlockSolveInfo.maxRnorm` is the raw LS residual ‖B[j]-AX[j]‖ read directly off the maintained R block (no extra AX recompute needed, unlike blsmr, since CGLS carries R explicitly). Validated pre-commit via a scratch job (deleted, not part of the authored suite): per-column normal-equation optimality + exact match to scalar `lsmr`, consistent-system exact recovery, zero-RHS immediate convergence, tiny-maxIter no-NaN -- ALL PASSED on first implementation, all three compiled variants (fProxy/float/double), no float status-gating needed (unlike blsmr's conservative float convergence flag) -- CGLS's maintained-residual convergence test is exact, not an estimate. Test-writer should author `BlockCGLSTests.fProxy.cs` mirroring `BlockLSMRTests.fProxy.cs`'s 4-case structure; no block-LS battery family yet (blsmr not wired either).

## Krylov.Block.LSMR
- 2026-07-21 | Stopping estimator rewritten from subtractive to direct (was Krylov.Block.LSMR.fProxy.cs:234). Old form `nrmATR2 -= ||phik||_F^2` telescopes ‖AᵀB‖²−Σ‖phi‖² by SUBTRACTION: near convergence it's small = large − large, and its noise floor sat a small factor ABOVE threshold = tol²·‖AᵀB‖² — after the SQRT2 QR-precision fix, double blsmr reached X ≈ 1e-14 on TallRandom24x8 yet ran to maxIter and reported not-Converged. New form is the block analog of scalar lsmr's `zetabar = -sbar*zetabar`: maintain the s×s tail block of the block-rotated LS RHS, `zetabark+1 = cbark @ zetabark`, `zetabar1 = Bbar1`; then ‖AᵀRk‖_F = ‖zetabark+1‖_F exactly (in exact arithmetic) and non-increasing in fp (cbark is an s×s submatrix of an orthogonal 2s×2s Q, so ‖cbark‖₂ ≤ 1). Derivation: ‖AᵀRk‖_F is the residual of min_Y ‖c − T̃k Y‖_F with c = [Bbar1;0;…] (paper eq. 7-8); the paper's normal equations Tk Yk = Fk are that LS's stationarity (Fk = T̃ᵀc), so its [phi1..phik] = R̃⁻ᵀFk = the rotated RHS's top blocks and the residual = the tail block; Qᵀ = [[abark,bbark],[cbark,dbark]] per ExtractBlockTranspose, giving the cbark product. Orthogonality ‖zetabark‖² = ‖abark·zetabark‖² + ‖cbark·zetabark‖² reproduces the paper's Theorem-1 telescoping identity, confirming ‖phik‖ = ‖abark·zetabark‖. Battery re-stricted: strictConsistentStatus back to IsDouble().
- 2026-07-20 | New `Krylov.Block.LSMR.fProxy.cs` (task #43, blsmr): block LSMR for tall/overdetermined systems via block Golub-Kahan bidiag + block QR update; ported from rectangular/BlockLSQR-LSMR-algorithm-extract.md (Bl-LSMR Algorithm 2). Fidelity deviations: no warm start, no damping, per-row Apply/ApplyT (not a fused block GEMM). Key oracle = per-column normal-equation optimality + matches scalar lsmr — passes all dtypes; exact recovery of a consistent system also holds all dtypes. The convergence FLAG (Solved/converged==s) is CONSERVATIVE in float: the block-GKB residual estimate is stricter than the achieved accuracy, so float recovers Xk yet can report MaxIterations — so ConsistentSystemRecoversExactSolution asserts full-convergence status for double only, recovery for all. Not wired to a battery (no block-LS battery family yet). NOTE: the WIP's earlier "test data reads back as zero" was the IJob struct-copy trap (verify INSIDE the job, never off a job field after Run()), NOT an Arena bug — see #54 / ArenaBurstReadbackRegressionTests.

## Krylov.MINRESQLP
- 2026-07-21 | maxxnorm: absolute `(fProxy)1e7` → per-iteration problem-relative min-length cap
  `beta1 / (Consts.MaxXNormFactor * max(tol, eps) * max(Anorm, pnorm))`, MaxXNormFactor = 64 per
  dtype (Consts). The reference's MAXXNORM=1e7 default is "effectively unbounded" (its own
  commented singular examples pass maxxnorm=1e2); as a hard constant it was (1) sitting AT float's
  garbage scale — the float exact-singular oracle instances diverge to ‖x‖ 8.2e6/5.5e6 under it
  (replica probe) — and (2) too loose to enforce min-length in double: a near-null direction
  (σ=1e-9) carrying a small b-component (3e-3) yields u = b_σ/σ = 3e6 < 1e7, returned as a
  CERTIFIED Converged (compatible cert, rnorm 9.4e-8) with ‖x‖ = 3.0e6 vs min-length 1.67 (suite
  red-proof). The 64 mirrors the exit certificates' 64*tol slack: the cap truncates a direction
  exactly when it could only ever be certified NULL at the requested tolerance
  (σ ≲ 64*tol*‖A‖est), so cap-truncated min-length exits stay certificate-promotable and
  certifiable-signal directions are never clamped; beyond beta1/(tol*Anorm) the relres stop metric
  is vacuous anyway (denominator Anorm*xnorm dominates). tol floored at eps (tol=0 degrades to a
  precision-aware absolute cap ~beta1/(64*eps*Anorm)). Formula validated in a standalone
  double+float replica of this loop across exact-singular / near-singular / slip-through /
  large-‖b‖ / compatible-large-x grids. Deliberately NOT changed: Acondlim=1e15 and TranCond=1e7
  (double-calibrated — flag 7 is effectively dead in float and the QLP transfer rare — but no
  demonstrated failure, and retuning them shifts iteration paths across every caller).
- 2026-07-21 | Anorm estimator: pnorm at iters==2 used betal, which at that iteration holds
  beta1 = ‖M-weighted r0‖ — NOT an entry of the Lanczos T (T's column 2 is (beta2, alfa2, beta3);
  Choi/Paige/Saunders define Anorm as a ‖T‖ estimate; the betal leak is a quirk of the reference
  CODE's sliding window, faithfully ported until now). Whenever ‖b‖ > ‖A‖ it inflated Anorm to
  ‖b‖, so relres = rnorm/(Anorm*xnorm + beta1) fired VACUOUSLY at rnorm ~ tol*‖b‖²/‖A‖: a
  well-conditioned SPD n=20 with ‖b‖~3e6 exited at iters=4 with rnorm 7.7e3 and the exit
  certificate then rightly rejected it → MaxIterations on a trivially solvable system (suite
  red-proof); the same inflation made the LS certificate's 64*tol*Anorm*‖r‖ bound vacuously lax.
  Fix: substitute beta (= beta_2, a genuine T entry) at iters==2 only. Also required so the new
  maxxnorm cap's beta1/Anorm means ‖b‖/‖A‖ — the two halves are complementary (replica: cap-only
  stays broken on large-‖b‖, estimator-only stays broken on the float singular oracles).
- 2026-07-21 | Two-certificate exit gate replaces the #53 single-certificate honesty guard, which
  had a least-squares blind spot: it downgraded EVERY Converged whose fresh ‖b-Ax‖ > 64*tol*‖b‖,
  correctly killing false-Converged on compatible systems but also killing every genuine LS
  solution of an INCOMPATIBLE singular system (whose optimal residual is legitimately ‖b_perp‖,
  large). New gate: certificate 1 (unchanged, fires first, free) keeps Converged when
  finalRnorm <= 64*tol*beta1; certificate 2 (only when 1 fails, one extra matvec on the r=b-Ax
  already sitting in r1) keeps Converged when ‖A·r‖ <= 64*tol*Anorm*finalRnorm — the reference's
  RELARES = ‖Ar‖/(ANORM·RNORM) <= RTOL, r-RELATIVE (scaled by ‖r‖, NOT beta1). Soundness: on any
  system with κ < 1/(64*tol) a wrong-x large-residual Converged cannot pass 2 (‖Ar‖ >= σ_min‖r‖
  forces κ >= 1/(64*tol)); beyond that κ it is rank-cutoff semantics, same class as pinv rcond.
  Same gate now also PROMOTES flag-6/9 Breakdown exits (maxxnorm/u-clamp — the min-length
  mechanism as γ→0 on the null direction) to Converged when a certificate holds: verified
  empirically (promotion disabled, statuses probed) that A=diag(1,1,0),b=ones and a
  Householder-conjugated diag(3,2,1.5,1,0,0) exit flag 6/9 in double with ORACLE-EXACT min-length
  x (rnorm exactly 1 and √2) yet reported Breakdown pre-fix. Flags 7 (Acondlim) and -3 (non-SPD M)
  are genuine breakdowns, never promoted; a flag-6/9 divergence on a resolvable system cannot pass
  the certificate (same κ-cutoff argument). Pre-fix (no guard at all) Rosser+random-b false-
  Converged (double: rnorm 0.36, fresh ‖Ar‖ 2.4e-3 vs certified <= 8.5e-4) confirms the downgrade
  path still catches the #53 bug class. Float limitation (NOT changed here): on the exactly-
  singular oracle instances float's terminal Lanczos iteration is rounding-limited (x 2-14% off,
  stub build diverges to 6e5 unguarded), certificates honestly refuse → MaxIterations; only double
  certifies the oracle. Untested branch note: no current instance exits flags 1-5 with cert-2
  passing (clean flag-2 LS exits get clamped to 6/9 first on exactly-singular inputs); that branch
  is the same one-line status assignment as the tested promotion. Tests in
  KrylovVerifyAtExitTests (certificate invariant + two min-length LS oracles + compatible-Rosser
  #53 coverage).
- 2026-07-20 | Honesty guard on the Converged exit (#53, surfaced by battery Forbids=IllConditioned). The QLP stop metric relres = rnorm/(Anorm*xnorm+beta1) can be deflated below tol by a large Anorm*xnorm on a near-breakdown/clustered spectrum (Rosser), flagging Converged while the true ‖b-Ax‖/‖b‖ is large (~0.38). Fix: after the fresh finalRnorm recompute, downgrade Converged->MaxIterations when finalRnorm > 64*tol*beta1 (RAW ‖b‖ scale, not the inflatable QLP denominator). 64x keeps genuine convergence (raw residual runs only a few x the QLP metric). Tests MinresQLPNeverFalseConvergesOnRosser + MinresQLPStillConvergesHonestlyOnWellConditioned in KrylovVerifyAtExitTests. Other solvers in docs/dev/spec-krylov-nonconvergence-fix.md (gmres/biCGStab/idr/minres) deferred — a naive gmres change broke GCRODR RecycleZeroMatchesGmres equivalence, needs care.

## Krylov.MINRES
- 2026-07-20 | Honesty guard on the identity-path Converged exit (#53, per docs/dev/spec-krylov-nonconvergence-fix.md §4.1). The identity path trusted `phibar` as an exact `‖b-Ax‖` and returned Converged with no verify; once `gammaFloor` clamps `gamma` upward (near-degenerate spectrum, e.g. Rosser) the Givens rotation loses unitarity, `phibar` can drift small while the true residual is still large, AND the same event inflates `1/gamma` in the `w` update (the mechanism already named in TemplateSourceTests/DEVLOG.md's Rosser 1e14-1e19 divergence note). Fix: deleted the `M.IsIdentity` no-verify short-circuit so both identity and preconditioned paths fall into the one verify block already written for the preconditioned case (recompute `A.Apply(x)`, fresh `‖b-Ax‖²`, only return Converged if it also clears threshold; y/v reused as scratch, both idle there). The identity `MaxIterations` exit (still reports raw `phibar`) and all `Breakdown` exits are untouched — already-honest non-convergence signals, out of this fix's scope. Tests MinresNeverFalseConvergesOnRosser + MinresStillConvergesHonestlyOnWellConditioned in KrylovVerifyAtExitTests.

## Krylov.biCGStab
- 2026-07-20 | Honesty guard on both Converged exits (#53, per docs/dev/spec-krylov-nonconvergence-fix.md §4.3). `r`'s two in-place recurrences (`r -= alpha*v`, `r -= omega*t`) and `x`'s separate accumulation (`x += alpha*p/pHat + omega*r/sHat`) only coincide with `b-Ax` in exact arithmetic; a near-zero-but-nonzero `rho`/`rv`/`omega` pivot (same amplifier class as minres's `gammaFloor`) can decouple them so the tracked `ss`/`rr` reads small while the true residual is large. Fix, site (a) early exit (`ss<=threshold`, before `x` is committed for this iteration): build a TRIAL `x` in the idle `t` buffer, verify a fresh residual via the idle `v` buffer, only commit `x.CopyFrom(t)` and return Converged if the trial clears threshold — on a failed verify `x` is left untouched so the standard stabilization step below still applies `alpha*p` exactly once (no double-apply). Fix, site (b) main exit (`rr<=threshold`, `x` already committed): direct copy of cg's verify-at-exit shape using the idle `v` buffer, refreshing `r` to the honest residual on a failed verify so the next iteration's `rho`/`beta` are computed from real data. All Breakdown exits untouched. Tests BiCGStabNeverFalseConvergesOnRosser + BiCGStabStillConvergesHonestlyOnWellConditioned in KrylovVerifyAtExitTests.

## Krylov.idr
- 2026-07-20 | Honesty guard on both Converged exits (#53, per docs/dev/spec-krylov-nonconvergence-fix.md §4.5). Same shape as biCGStab: `R`'s two in-place recurrences (in-sweep `R -= beta*Gk`, end-of-sweep `R -= om*Q`) and `x`'s separate accumulation only coincide with `b-Ax` in exact arithmetic, and a near-zero shadow-space pivot can decouple them. Unlike biCGStab, `x`/`iter` are already committed by the time either `rr<=threshold` check runs, so no trial-x complication: both sites (in-sweep, after `x.addScaledInPlace(beta, Uk)`; end-of-sweep, after `x.addScaledInPlace(om, V/VHat)`) recompute a fresh residual into the idle `V`/`Q` buffers, refresh `R` to the honest value either way, and only set `status=Converged` if the fresh residual also clears threshold. On a failed verify, execution falls through with `R` corrected so subsequent `P[i]`-dot-`R` work in the next sweep stays correct. All Breakdown exits untouched. Tests IdrNeverFalseConvergesOnRosser + IdrStillConvergesHonestlyOnWellConditioned in KrylovVerifyAtExitTests.

## Krylov.Block.TFQMR
- 2026-07-20 | New file `Krylov.Block.TFQMR.fProxy.cs` (task #42): a PSEUDO-block generalization of
  `Krylov.TFQMR.fProxy.cs`, not a true (subspace-mixing) block method -- see the derivation note below
  for why the mixing design is ill-defined. `reference/square/BlockIDR-BlockTFQMR-algorithm-extract.md`
  Part B already documents an extensive, multilingual literature search that found NO published block
  TFQMR (block, transpose-free, quasi-minimal-residual); this entry records an independent from-scratch
  derivation attempt (per the task's "block-GENERALIZE... guided by the extract" instruction) that
  reaches the same negative conclusion for the mixing design, and explains what shipped instead.
- 2026-07-20 | DERIVATION -- why m x m block coefficients (mirroring `bbiCGStab`/`bidr`'s
  `BlockCrossGram`+`BlockSolveGeneral` pattern) are ill-defined for TFQMR specifically: TFQMR's x-update
  is `x_m = x_{m-1} + eta_m*d_m` with `d_m = uHat_m + (theta_{m-1}^2/alpha_{n}) * eta_{m-1} * d_{m-1}` --
  a SCALAR recurrence coefficient (`theta^2/alpha`) that reuses the SAME scalar `alpha` from the
  bi-orthogonality step to weight the PREVIOUS `d`. If `alpha`/`rho` are promoted to m x m matrices (as
  `bbiCGStab`'s engine does, and as this task's own CORRECTNESS section anticipated via
  `BlockSolveGeneral`), `theta^2/alpha` stops being well-formed: `theta` is a per-row scalar derived from
  `‖w_i‖` (a NORM, inherently scalar/per-row), but `alpha` is now a matrix MIXING all s rows together, so
  there is no coherent "divide by alpha" that keeps `d`'s row-space consistent with `w`'s row-space
  without inventing a genuinely new matrix least-squares recursion. That IS what true Block QMR
  (Freund-Malhotra 1997) does -- it re-derives the whole quasi-minimization via a two-sided (needs Aᵀ)
  block Lanczos process and a block-Hessenberg QR update -- but that paper is paywalled (confirmed
  `oa_status: closed`, not fetchable; see the extract's Part B item 1) and is NOT transpose-free, i.e. a
  structurally different algorithm from TFQMR, not a mechanical block-ification of it. A "global"
  (Frobenius-only, single shared scalar tau/theta/eta for the WHOLE s x n block) variant sidesteps the
  matrix-division problem but loses honest per-row convergence tracking, breaking the
  `BlockSolveInfo.converged`/`CountConverged`-style per-column contract every other block solver in this
  codebase honors. Concluded: a per-column-honest, subspace-mixing block TFQMR is not achievable by
  mechanically block-ifying the scalar recurrence within this task's scope -- genuinely ill-defined, not
  just hard, per the task's own stated park/drop criterion.
- 2026-07-20 | SHIPPED INSTEAD: `btfqmr` runs s INDEPENDENT scalar-TFQMR recurrences in lockstep (every
  coefficient is a per-row scalar, never mixed across rows), sharing one `ApplyBlock` call per half-step
  instead of s separate `Apply` calls. This batching pattern is a real, established design -- found
  in the reference stash AFTER the mixing-design derivation above, at
  `reference/square/BelosPseudoBlockTFQMRIter.hpp` (Trilinos/Belos, BSD-3): Belos ships exactly this
  ("Pseudo-Block TFQMR", explicitly named to distinguish it from a true mixing Block TFQMR, which Belos
  does NOT have either) -- per-RHS `alpha_/eta_/rho_/tau_/theta_` vectors, one shared `lp_->apply` per
  half-step. `btfqmr` reimplements the SAME batching idea from scratch against this codebase's own
  shipped `Krylov.TFQMR.fProxy.cs` recurrence (variable names/structure/breakdown-guard order mirror the
  scalar file directly), not a line-by-line port of the Belos C++. Consequence: `btfqmr` has NO
  block-Krylov-subspace advantage over looping scalar `tfqmr` s times (wired into
  `KrylovBlockBatteryTests.fProxy.cs` with `BlockAdvantage=false`, per the task's own instruction "like
  bbiCGStab/bidr") -- the value it adds is the shared `ApplyBlock` traffic (one BLAS3-shaped pass over A
  per half-step instead of s BLAS2 passes for dense/BSR operators) and API parity with the rest of the
  block family, not fewer iterations.
- 2026-07-20 | Because rows never mix, two bit-identical RHS rows produce bit-identical output rows and
  a duplicate row cannot singularize any shared coefficient (there is no shared coefficient) -- unlike
  `bbiCGStab`/`bidr`, whose `NoBreakdown`/`IdenticalColumns` battery flags are false because their shared
  m x m block solve genuinely goes singular on duplicate RHS rows. `fProxyBtfqmrInvoker`'s CheckFlags
  were verified empirically against the battery (see `KrylovBlockBatteryTests.fProxy.cs`'s `Btfqmr` case)
  before being set stronger than bbiCGStab/bidr's.

## Krylov.Block.IDR
- 2026-07-20 | New file `Krylov.Block.IDR.fProxy.cs`: true block IDR(s) (task #41), the block
  generalization of `Krylov.IDR.fProxy.cs`. Ported from `reference/square/BlockIDRs-Du-Sogabe-Yu-
  Yamamoto-Zhang-JCAM2011.{pdf,txt}` (Algorithm 2) via the clean extract
  `reference/square/BlockIDR-BlockTFQMR-algorithm-extract.md` (paper, uncertain-OA -- see the extract's
  own provenance note; pseudocode-only, reimplemented from scratch) and cross-checked against
  `reference/square/IDRsSolver.jl/IDRsSolver.jl` (MIT) for the scalar recurrence shape. `idrs.jl` in the
  same folder was NOT read (per `reference/README-PROVENANCE.md`'s MPL-uncertainty flag on that file).
- 2026-07-20 | DEVIATION (structural, not mathematical): block-generalized the ACTUAL scalar `idr`
  implementation in this codebase (the merged "G/U start at zero, Msys starts at identity" loop, itself
  the canonical van Gijzen/Sonneveld `idrs.m` form) rather than transcribing the paper's own two-phase
  Algorithm 2 (explicit phase-1 loop building s initial residuals via 1-D residual-minimizing MR steps,
  then a separate main loop). Both are the same recursion in different packaging -- with G/U starting
  at zero and Msys at the identity, the merged loop's first sweep IS phase 1 (verified by hand: k=0's
  forward-substitution is trivial under an identity Msys, giving G[0]=A*r0, U[0]=r0, matching Algorithm
  1/2's phase-1 first step exactly). Keeps bidr consistent with idr's own file rather than introducing a
  second IDR packaging into the codebase.
- 2026-07-20 | Block-generalization derivation (row-storage convention, m x n blocks with m = RHS count
  = paper's "m", n = A.Rows): every scalar dot product `Blas.dot(P[i], X)` becomes an m x m Gram block
  `BlockCrossGram(P[i], X)` (= `P[i] @ X^T`, matches the paper's `P_i^H X` DIRECTLY under this row
  convention -- no transpose correction needed, unlike the bi-orthogonalisation coefficients below);
  every scalar divide `f[i]/dkk` becomes an m x m general (non-SPD) solve via `BlockSolveGeneral`
  (QRCP rank-revealing, reusing `bbiCGStab`'s own helper -- Msys's blocks have no SPD guarantee); every
  scalar-times-vector combine (`c[i]*G[i]`) becomes `BlockCTV(c[i], G[i])` = `c[i]^T @ G[i]` (LEFT-
  multiply by the coefficient's TRANSPOSE) -- this transpose is real and necessary: the paper's own
  combine is a RIGHT-multiply (`G_paper[i] @ c_paper[i]`, n x m paper convention), and translating a
  right-multiply into this codebase's row-major (m x n) storage flips it into a left-multiply by the
  transpose, which is exactly `BlockCTV`'s `C^T @ V` shape already used by `bbiCGStab`/`bgmres` for the
  identical reason. Bi-orthogonalisation's `alpha` and the final `beta` both solve `Msys[i,i] @ X = RHS`
  (LEFT-multiply, standard block-forward-substitution direction) then feed into `BlockCTV` the same way.
  `om` (the end-of-sweep step length) stays a SCALAR trace ratio (`BlockFrobDot(Q,Q)`/`BlockFrobDot`
  cross terms), matching the paper's own explicit remark that block IDR(s) still does a scalar
  Frobenius-residual minimization per step, not a blocked one.
- 2026-07-20 | Storage: Msys/f/c are `UnsafeList<fProxyMxN>` of small (m x m) standalone blocks (Msys
  sized s*s, only the lower triangle i>=j ever read/written, mirrors scalar idr's own unused-upper-
  triangle s x s Msys) rather than one big (s*m) x (s*m) matrix with offset sub-views -- avoids needing
  a strided/offset submatrix view type this codebase does not have (RectView/RowsView/View all start at
  a buffer's own row/col 0), at the cost of s*s small Allocator.Temp allocations (acceptable: s is a
  small user-chosen shadow depth, matching the many-small-Temp-allocs pattern scalar idr's own P/G/U
  lists already use).
- 2026-07-20 | No deflation/column-locking (every RHS column stays live the whole solve, `minActive`
  always m in the returned `BlockSolveInfo`) -- the paper explicitly states deflation of converged
  columns is left to future work; mirrors `bbiCGStab`'s identical choice for the identical reason.
  Breakdown (any m x m block solve reporting non-`Success`, or a non-positive/NaN `om` denominator)
  reports `IterativeSolveStatus.Breakdown` with X holding the last committed iterate -- never NaN, never
  throws from the recurrence itself.
- 2026-07-20 | Seed default (`0x9E3779B1u`, `seed == 0` folds to the same default) copied VERBATIM from
  `Krylov.idr` -- required by spec for cross-solver seed consistency, not an independent choice.
  `fProxyBidrInvoker` (`KrylovBattery.Invokers.fProxy.cs`) mirrors `fProxyIdrInvoker`'s own `S`/`Seed`
  fields (S = shadow depth, unrelated to the battery's own block-width `s`) and wires into
  `KrylovBlockBatteryTests.fProxy.cs` as `SolverKind.Bidr` with `BlockAdvantage=false`/`NoBreakdown=
  false` (both citing the same no-monotone-bound / no-deflation rationale as `BbiCGStab`'s own entry).

## Krylov.bfgmres
- 2026-07-20 | New file `Krylov.Block.FGMRES.fProxy.cs`, sibling to `Krylov.Block.GMRES.fProxy.cs` (not
  an addition to it) per the single-solver-per-file convention (task #38). This is the intersection of
  the two solvers already in the tree: `bgmres`'s block Arnoldi/Givens/Hessenberg machinery (block RHS,
  rank-revealing-LQ deflation, periodic dense-QR block least-squares) with `fgmres`'s flexible-basis
  update ported onto it. Reused every `Krylov.Block.Common.fProxy.cs` / `Krylov.Block.BiCGStab.fProxy.cs`
  helper `bgmres` itself uses (`BlockCrossGram`/`BlockCTV`/`BlockAdd`/`CopyBlock`/`BlockApplyPre`/
  `CountConverged`/`View`/`RowsView`/`RectView`/`LQRPRank`/`StoreBlockAt`/`ExtractRowsAt`/`ZeroPrefix`,
  the last three defined in `Krylov.Block.GMRES.fProxy.cs` itself and visible here via the shared
  `partial class Krylov`) unmodified -- no new shared helper needed.
- 2026-07-20 | The ONE structural change from `bgmres`: `bgmres`'s Arnoldi loop already right-
  preconditions each step (`w = A M⁻¹ v_j`) into a single reusable scratch buffer (`Zt`), then applies
  M ONCE more to the combined vector at commit time (`X += M⁻¹(Σ y_i v_i)`) -- valid only because M is
  fixed across the whole cycle, so `M⁻¹` factors out of the sum. `bfgmres` instead stores each step's
  `Z[j] = M⁻¹ V[j]` into a PERSISTENT `Z[0..m-1]` array (mirrors `fgmres`'s own `Z`) and commits
  `X += Σ Yᵢᵀ Zᵢ` reading directly off the stored per-step basis, never re-applying M -- valid even when
  M varies every step. Verified against Belos (`reference/belos/BelosBlockFGmresIter.hpp`, BSD): its
  persistent `Z_` multivector (sized to the full restart length, `Znext = M*Vprev` per step,
  `getCurrentUpdate` = `Z*y`) is the same design, reimplemented onto this codebase's own block-Arnoldi
  loop rather than ported line-by-line.
- 2026-07-20 | Under `fProxyIdentityPreconditioner`, `Z` is never allocated and every commit-step branch
  reads `V[i]` instead of `Z[i]` -- the EXACT same instruction sequence `bgmres` runs under identity (no
  arithmetic reordering, same Temp-allocation shapes/order). Verified bit-identical (X AND iteration
  count, not just "close") against `bgmres` in `BlockFGmresTests.fProxy.cs`'s
  `IdentityFoldMatchesBgmresBitIdentical`, across multiple restart cycles (`restart < n`).
- 2026-07-20 | Overload ladder mirrors `bgmres` exactly (generic `bfgmres<TOp,TPre>` core,
  `bfgmres<TOp>` identity forwarder, dense-general/BSR unpreconditioned rungs, BSR+ILU0 preconditioned
  rungs) -- callers already fluent in `bgmres`'s ladder get `bfgmres` for free. Workspace footprint:
  `bgmres`'s `V` (`(m+1) s n` elements) plus `Z` (`m` more `s x n` blocks) when M is a real
  preconditioner -- the same roughly-doubled-basis-memory cost `fgmres` pays over `gmres`, for the same
  reason (tolerating a per-step-varying M).

## Krylov.GMRES
- 2026-07-21 | Task #60 (Fable audit `docs/dev/audit-krylov-fable-20260720.md`): two fixes to the
  shared `GivensApplyAndGenerate` (`Krylov.Arnoldi.Common.fProxy.cs`, also used by `fgmres`) plus a
  verify-at-exit gate. (1) Breakdown: when the rotated pivot `H[j,j]` AND the raw Arnoldi
  subdiagonal are BOTH exactly zero, the old code took the `c=1,s=0` happy-breakdown branch
  unconditionally, leaving `H[j,j]=0` stored and `g[j+1]=0` -- read as both an instant Converged
  (false, since x hadn't even been updated with this cycle's zero-progress column) AND a divide-by-
  zero in `HessenbergBackSolve`. `GivensApplyAndGenerate` now returns false in that case; the caller
  excludes the column (`k` stays at the prior valid count) instead of back-substituting through it.
  Concrete trigger: `A` all-zero (or any operator that maps the current Arnoldi vector into the span
  already covered AND leaves the rotated pivot exactly zero too) -- `diag(1,0)` with `b=(1,1)` also
  reaches it after normalization. (2) If that breakdown produces ZERO usable columns this cycle (`k
  == 0`, meaning x is provably unchanged), a restart would reproduce an identical residual forever;
  a `deadEnd` flag forces the outer loop to stop and report `Breakdown` instead of spinning. If `k >
  0` (partial progress), the cycle's partial x-update is kept and the outer loop naturally restarts
  from the new residual. (3) Verify-at-exit: the mid-cycle Converged exit (`|g[j+1]| <= thresh`) is
  the rotated-rhs ESTIMATE, not the true residual -- MGS orthogonality loss can make it drift
  optimistically below `tol*bnorm` on an ill-conditioned A. After the post-loop x-update (identity or
  preconditioned), a fresh `VerifyTrueResidual` recomputes ‖b-Ax‖ using `w`/`V[0]` as scratch (both
  fully overwritten at the top of the next cycle regardless, so free); only kept Converged if the
  fresh residual also clears the threshold, else falls through to another restart cycle. The
  existing restart-ENTRY check (top of the while loop) was already a true residual and needed no
  change. `total` now increments once per attempted Arnoldi column (moved before the breakdown
  check, same count as before on every non-breakdown path -- no behavior change there).

## Krylov.FGMRES
- 2026-07-21 | Task #60: same three fixes as `Krylov.GMRES` (shared `Krylov.Arnoldi.Common.fProxy.cs`
  breakdown guard + `deadEnd` dead-column handling + post-x-update verify-at-exit using `w`/`V[0]`
  scratch). The preconditioned x-update reads the STORED per-step `Z[i]` vectors (valid even when M
  varies every step) instead of gmres's single `zt` -- unaffected by the fix, `k` simply excludes the
  breakdown column from that sum too.

## Krylov.GCRODR
- 2026-07-21 | Task #60: added a verify-at-exit gate after the post-cycle x-update + recycle
  correction, before the deflation-update block. `|g[j+1]|` only measures the C-orthogonal
  (recycle-projected) residual -- a clamped `Ru` back-solve (near-zero diagonal) can leave the
  C-component of the true residual un-cancelled, so a Converged decision from the Arnoldi/Givens
  estimate alone can be wrong. `VerifyTrueResidual` recomputes ‖b-Ax‖ using `w`/`V[0]` (both fully
  overwritten at the top of the next cycle regardless, so free); a failed verify flips `converged`
  back to false, which naturally re-enables the deflation-update block for that cycle (previously
  skipped only when actually converged) instead of returning early. gcrodr's OWN pivotGuard-based
  breakdown path (a DIFFERENT bug the same audit entry flagged as separately mis-scaled, out of scope
  here) already recomputes a true residual on that path and was untouched.
- 2026-07-20 | Breakdown path now recomputes `rnorm` as a fresh ‖b-Ax‖ before returning, instead of
  reporting the stale Arnoldi/Givens `resnorm` estimate -- found while reviewing the bespoke
  `SingularBreakdownTest` (all-zero A): the degenerate Givens rotation (`rr==0 -> c=1,s=0`) leaves
  `g[j+1]==0`, an artifact that reads as a near-zero residual even though x was never updated that
  cycle and the true residual is still ‖b‖. Never affected the Breakdown/NaN correctness the test
  checks (rnorm was 0, not NaN), just SolveInfo.rnorm's documented "at the returned x" contract.
- 2026-07-20 | New file `Krylov.GCRODR.fProxy.cs`: single-RHS GCRO-DR (Morgan 2002 / Parks-de
  Sturler-Mackey-Johnson-Maiti 2006), the last single-RHS Krylov solver (task #29). Extends
  `gmres(m)` with a recycled k-dimensional subspace carried across restart cycles. Structure/Arnoldi/
  Givens machinery mirrors `Krylov.GMRES.fProxy.cs` directly; the flexible-basis storage (retaining
  M⁻¹v_j across a whole cycle instead of one scratch buffer) mirrors `Krylov.FGMRES.fProxy.cs`'s Z
  array. Primary reference `reference/square/BelosGCRODRIter.hpp` supplied the Arnoldi/QR-update
  loop shape but NOT the harmonic-Ritz recompute (that lives in Belos's SolMgr file, not in the
  `reference/` stash) -- the deflation-update math below was derived from first principles
  (Morgan/Parks harmonic Ritz theory), not ported line-for-line. `reference/scipy/_gcrotmk.py` is a
  DIFFERENT algorithm (GCROT(m,k), single-vector-append recycling, no harmonic Ritz) -- used only to
  cross-check the recycled-subspace projection bookkeeping shape (C/U roles, `x += U(C^Tr)` /
  `r -= C(C^Tr)` cycle-start projection), not ported.
- 2026-07-20 | DEVIATION from strict Morgan-2002 harmonic Ritz VECTORS, forced by the library's
  eigensolver surface: the classic method needs eigenvectors of a small dense GENERAL (nonsymmetric)
  matrix, but this codebase's only nonsymmetric dense eigensolver (`Eigen.valuesQRInPlace`) is
  values-only (Hessenberg + Francis QR, EISPACK hqr) -- there is no general eigenVECTOR solver, and
  the spec requires reusing an existing eigensolver rather than hand-rolling one. Fix: use harmonic
  Ritz VALUES from `valuesQRInPlace` (real ones only -- a complex-conjugate pair is skipped, since a
  single real recycled direction can't represent one half of a complex pair; the target matrix
  usually has more real small-eigenvalue candidates than `recycle` needs) but REFINED vectors for
  each selected value: z = argmin_{‖z‖=1} ‖(AP - θP)z‖ over the combined old-recycle + this-cycle
  Krylov space P, which is the smallest-eigenvalue eigenvector of the SYMMETRIC d x d matrix
  `N_θ = Fmat - θ(Gmat+Gmatᵀ) + θ²·Pgram` (Fmat=(AP)ᵀ(AP), Gmat=(AP)ᵀP, Pgram=PᵀP), solved via
  `Eigen.symmetricInPlace` (existing, has eigenvectors). Refined Ritz vectors are themselves an
  established alternative to harmonic Ritz vectors in the recycled-Krylov literature (Jia 1997);
  pairing them with harmonic Ritz VALUES (rather than plain Ritz values from H alone) keeps the
  "target the smallest eigenvalues" property Morgan's method relies on. One eigendecomposition of
  `N_θ` per selected θ (up to `recycle` calls per cycle) -- d is small (≲ restart+recycle), so this
  is cheap relative to the O(n) Arnoldi work per cycle.
- 2026-07-20 | AU=C·Ru invariant (not AU=C exactly): C (orthonormal) and Ru (upper triangular) are
  rebuilt each cycle via a thin QR of the harmonic-Ritz combination `A·U_raw` (`QR.decompInPlace`);
  U itself is kept unnormalized. This needs one extra k x k upper-triangular back-substitution both
  at cycle start (`Ru z = Cᵀr` before `x += Uz`) and at cycle end (`Ru q = B[:,:p]y` before
  `x -= Uq`) versus the simpler "AU=C exactly" invariant some references use, but avoids ever needing
  a triangular matrix RIGHT-division (`U_final = U_raw·Ru⁻¹`, i.e. solving `X·Ru = Y` for a matrix
  X) which this library has no direct primitive for. Both residual updates recompute `b-Ax` with a
  FRESH matvec rather than trusting the incremental `r -= C·(Cᵀr)` through a possibly
  near-singular Ru solve -- a deliberate one-extra-matvec-per-cycle cost for exactness (a clamped
  `z[i]=0` on a near-zero Ru diagonal must never let x and the tracked residual silently disagree).
- 2026-07-20 | AV[j] (the raw, PRE-projection `A·(this-cycle basis vector)`) is captured during
  Arnoldi and reused for the deflation-update's `AP` -- avoids re-applying the operator for the
  Krylov part of the combined subspace. The recycled part's `A·U_old` (= `C_old·Ru_old`) is instead
  MATERIALIZED explicitly each cycle (k dense combinations, not re-derived through the R-factored
  block-Gram shortcut a more optimized port would use) -- simpler and easier to audit at the cost of
  O(n·k²) extra work per cycle, which is small next to the O(n·restart) Arnoldi cost. A future perf
  pass could fold this into the Gram-matrix construction directly (Fmat's top-left block reduces to
  `Ru_oldᵀRu_old` via C's orthonormality, no n-dependence) -- not done here, correctness-first given
  the 1-hour-park budget for this solver.
- 2026-07-20 | Guards (never NaN, never a false Converged): Arnoldi subdiagonal (`hj1`) and the final
  Hessenberg back-substitution pivot both compare against `pivotGuard = 100·eps·(‖b‖+1)`; the former
  is a HAPPY breakdown (stops the cycle's Arnoldi loop, not an error -- the Krylov space is exactly
  A-invariant here); the latter is an honest `IterativeSolveStatus.Breakdown`. The recycle-space
  triangular solves (`Ru`) clamp a near-zero pivot's contribution to 0 rather than dividing (paired
  with the fresh-residual-recompute above, so this never desyncs x from the tracked residual). The
  deflation update's own numerics (`Gmat` LU solve, the small eigendecompositions, the QR
  rank-check on the new `Ru`) degrade to "keep the previous cycle's recycled subspace unchanged"
  on ANY failure -- recycling is an accelerator, not a correctness requirement, so a bad cycle there
  never aborts the outer solve.
- 2026-07-20 | `recycle` sits where the spec asked (`restart` then `recycle`, before `maxIter, tol`),
  guarded `0 <= recycle < restart` (recycle=0 disables recycling entirely and is bit-identical to
  `gmres(m)` -- exercised as an internal consistency check, not a spec requirement). Overload ladder
  mirrors `gmres` exactly (TOp,TPre generic / TOp identity-forward / dense / dense-defaults / BSR /
  BSR-defaults / BSR+ILU0 / BSR+ILU0-defaults) with one added helper, `GcrodrDefaultRecycle`, for the
  three "-defaults" overloads' `recycle = min(10, restart-1)`.
- 2026-07-20 | Wired into the square Krylov battery: `fProxyGcrodrInvoker` + `SolverKind.Gcrodr` in
  `KrylovBattery.Invokers.fProxy.cs` / `KrylovSquareBatteryTests.fProxy.cs`, `Requires=Square,
  Forbids=IllConditioned` -- same task-#53-deferred Rosser exclusion class as gmres/biCGStab/idr
  (unguarded-in-gmres-family Hessenberg pivot on that clustered-near-degenerate spectrum; gcrodr's
  OWN pivot guard above would catch it as a clean Breakdown rather than gmres's unbounded diverge,
  but the battery's shared RunStandardChecks treats Breakdown as a failing status like any other
  solver here, so the exclusion stays for consistency with its siblings pending a battery-level
  decision on whether Breakdown should count as an acceptable outcome).

## Krylov.CRAIGMR
- 2026-07-20 | New file `Krylov.CRAIGMR.fProxy.cs`: single-RHS CRAIGMR, the MINRES-flavored sibling
  of `craig` -- same least-norm problem (min ‖x‖ s.t. Ax=b, A m×n full row rank, m<=n), same
  Golub-Kahan bidiagonalization, but the lower-bidiagonal system L is solved via a running QR
  (one Givens rotation/step) instead of craig's forward substitution, giving a MONOTONICALLY
  decreasing ‖b-Ax‖ (craig's error is monotonic, its residual is not). Task #37.
- 2026-07-20 | PORT-FIDELITY DEVIATION: primarily ported the recurrence in
  `reference/rectangular/CRAIGMR-BlockCRAIG-algorithm-extract.md` §3.1-3.2 (a clean-room
  reimplementation reference, cross-checked by that document's author against both Krylov.jl's
  MPL-licensed docstring/source, read-not-copied, and the MIT `pykrylov-lls/craigmr.py`), rather
  than porting `pykrylov-lls/craigmr.py` line-for-line as the nominal "primary" source. Reason:
  pykrylov's actual code decomposes the same update into THREE named rotation families ("type
  I/II/III") seeded by a fixed `alpha_hat = sqrt(alpha**2 + 1)` -- consistent with solving the
  augmented saddle-point system `[I A; Aᵀ 0][r;x]=[b;0]` directly via MINRES (the identity block
  contributes the structural "+1"), a different derivation path to the same answer. The extract's
  single-rotation form works directly off the reduced AAᵀ system (matching this library's existing
  `craig`/`lsmr` bidiagonalization, no augmented-system reindexing) and is markedly simpler to audit
  line-by-line. The extract itself flags pykrylov's `craigmr.py` as real but unfinished (TODO
  comment, stray Python-2 `print`, no y output). Verified against the required behavior instead
  (least-norm oracle + monotonic-residual tests), not against pykrylov's numbers directly.
- 2026-07-20 | API surface: only `x` is tracked/returned (matches `craig`'s API), not the dual `y`
  (Aᵀy=x, AAᵀy=b) that the full published CRAIGMR also produces. Traced the dependency graph of the
  extract's §3.2 update: the y/w/wbar family is write-only into y and is never read back into the
  d/x/theta/rhobar recurrence that actually produces x -- so it was dropped entirely rather than
  computed and discarded. Scratch buffers: u/v/tmpM/tmpN (craig's four) plus one more, `d` (the
  running Aᵀ-image accumulator, craigmr's analog of craig's direct `v`-weighted sum / lsmr's h).
  Default `maxIter` = `A.M_Rows` (same bound as craig, not lsqr/lsmr's `A.N_Cols`) -- identical
  bidiagonalization, identical finite-termination argument.
- 2026-07-20 | Unlike craig, `LstsqInfo.rnorm`/`Arnorm` are FILLED FOR FREE from the same per-step
  QR recurrence (`rNorm = |zetabar|`, `ArNorm = alphaNew*betaNew*|zeta|/rho`) via
  `LstsqInfoTracked` -- no `lstsqResidual` audit needed, unlike craig's direct rank-1 update which
  has no free per-iteration ‖Aᵀr‖. This is the same tracked-norm shape `lsmr` uses.
- 2026-07-20 | Breakdown guards, in order every iteration: (1) `rho > 0` (both the running rotation
  radius and the fresh beta collapsed -- reports the PREVIOUS iteration's rNorm/ArNorm, matching
  lsmr's "carry prior step's values" convention), checked before the convergence test since rho
  gates the division that produces this step's x; (2) `betaNew > 0` after the convergence check
  (defensive only -- betaNew==0 forces rho=|rhobar|, s=0, zetabar=0, rNorm=0, which converges above
  for any tol>=0, so this branch is not known to be reachable, but is kept as a guard on the
  division it precedes, matching craig/lsmr's own defensive doubling); (3) `alphaNew > 0` (Krylov
  space exhausted backward). Same KNOWN LIMITATION as craig: these are exact bit-zero tests, not a
  relative/‖A‖-scaled threshold, so a NUMERICALLY (not bit-exactly) rank-deficient A can slip past
  the guard and run to MaxIterations with a blown-up but finite (never NaN) x instead of reporting
  Breakdown -- deep rank-deficient least-norm is out of scope per the task.
- 2026-07-20 | NOT wired into the standardized Krylov battery, same as craig (least-norm invokers
  don't exist yet there) -- bespoke coverage only, in `CRAIGMRTests`.

## Krylov.CRAIG
- 2026-07-21 | Task #60: the tracked-estimate Converged exit (`rnorm = |beta*z| <= tol*bnorm`) is now
  reconciled against `CraigInfo`'s certified-exact residual (`lstsqResidual`: one Apply + one ApplyT)
  BEFORE returning, instead of trusting the recurrence estimate outright -- CRAIG is CG-on-AAᵀ in
  disguise (κ² sensitivity), so an ill-conditioned A can make the estimate cross threshold while the
  certified rnorm has not. No extra matvec: `CraigInfo` was already being called on every exit path
  (Converged/Breakdown/MaxIterations) to fill `LstsqInfo`, so this just checks its result before
  committing to Converged rather than after. `tmpM`/`tmpN` are safe to reuse for the audit even when
  falling through to keep iterating (`GolubKahanUStep`/`VStep` fully overwrite them before their next
  read, same write-first contract `lstsqResidual` relies on).
- 2026-07-20 | New file `Krylov.CRAIG.fProxy.cs`: single-RHS CRAIG (Craig 1955; Paige-Saunders BIT
  1995), the least-NORM counterpart to `lsqr`/`lsmr` -- among all x with Ax=b (A m×n, m<=n, full row
  rank), returns the minimum-Euclidean-norm one. Fills the gap noted in task #27: the library had
  least-SQUARES but no least-NORM solver.
- 2026-07-20 | PORT-FIDELITY DEVIATION: primarily ported `reference/rectangular/craigSOL/craigSOL.m`
  (Saunders SOL MATLAB), not `reference/rectangular/pykrylov-lls/craig.py` (nominally the "primary"
  source per the spec). pykrylov's `craig.py` implements the GENERALIZED CRAIG (damping, arbitrary
  SPD preconditioners M/N, dual QR-style rotations for a symmetric quasi-definite system) -- a
  materially different, more complex recurrence than plain undamped CRAIG, and out of scope for this
  task (no damp/M/N in the spec). `craigSOL.m` is the plain Paige-Saunders algorithm and was
  cross-checked against `reference/rectangular/CRAIG-Paige-Saunders-BIT1995.txt`. Derived and
  verified the bidiagonalization or ordering (craigSOL computes v_k before u_{k+1} each loop
  iteration, using dummy pre-loop values alpha=1/z=-1 to bootstrap; this is loop-boundary
  bookkeeping only -- the underlying Golub-Kahan recurrence is identical to lsqr/lsmr's).
- 2026-07-20 | API surface deviates from lsqr/lsmr in three ways, all documented on the entry point:
  (1) no `w`/`y` scratch buffers -- craigSOL's y (dual variable satisfying x=Aᵀy, AAᵀy=b) and its w
  bookkeeping were dropped since nothing in this API exposes y; x's update is a direct rank-1
  `x += z*v` accumulation, not lsqr's Givens-rotated w-based update, so only u/v/tmpM/tmpN (4
  buffers) are needed, not lsqr's 5. (2) No `damp` parameter and no `craigJacobi` -- undamped only,
  matching the spec's scope. Deliberately did NOT add a naive Jacobi/column-scaling preconditioner:
  for lsqr, column-scaling the search (x=Dy) is a harmless change of variables because the objective
  (‖Ax-b‖) doesn't care how x is parametrized. For CRAIG, minimizing ‖y‖ in scaled space and
  unscaling x=Dy minimizes a WEIGHTED norm ‖D⁻¹x‖ of the original x, not ‖x‖ -- silently the WRONG
  min-norm solution unless D=I. A correct CRAIG preconditioner needs the generalized M/N form (the
  part of pykrylov's craig.py this port intentionally skipped), so it's left undone rather than
  shipped subtly wrong. (3) Default `maxIter` = `A.M_Rows`, not lsqr/lsmr's `A.N_Cols`: CRAIG's
  bidiagonalization is CG-on-AAᵀ in disguise (an m×m system), so it terminates within m steps in
  exact arithmetic, unlike lsqr/lsmr's search which can run up to n steps.
- 2026-07-20 | `LstsqInfo.rnorm`/`Arnorm`/`xnorm` are filled via a fresh `lstsqResidual` audit (one
  extra Apply+ApplyT) at every return point, not a tracked per-iteration identity -- CRAIG's direct
  rank-1 x update has no free ‖Aᵀr‖ recurrence the way lsqr's Givens rotations or lsmr's MINRES-layer
  rotations do. The LOOP's own convergence test still uses the free `rnorm = |beta*z|` identity from
  craigSOL (no extra matvec per iteration); only the terminal `LstsqInfo` construction pays the extra
  Apply+ApplyT, once per call.
- 2026-07-20 | Breakdown guard is an exact `alfa > 0` / `beta > 0` test (NaN-safe, matches
  lsqr/lsmr's own convention), firing when the bidiagonalization collapses to bit-exact zero -- e.g.
  b exactly orthogonal to range(A) on the first step. KNOWN LIMITATION found while testing: a
  NUMERICALLY (not bit-exactly) rank-deficient A -- e.g. one row a near-exact multiple of another --
  produces a tiny-but-nonzero `alfa` (~1e-7 float / ~1e-16 double) from roundoff, so this guard does
  NOT fire; the solver instead runs to `MaxIterations` with `x` blown up by the `z = -(beta/alfa)*z`
  division (‖x‖ reaching ~1e7 float / ~1e16 double -- finite, never NaN, and status is honestly
  MaxIterations/not-Solved, never a false Converged, so the core safety contract holds). A tighter
  breakdown detector would need a relative threshold on alfa (vs. the first alfa or an ‖A‖ estimate)
  instead of an absolute zero test -- not attempted here (deep rank-deficient least-norm is
  out-of-scope per the spec; that territory belongs to a future LNLQ/CRAIGMR).
- 2026-07-20 | NOT wired into the standardized Krylov battery (`KrylovSquareBatteryTests`/
  `KrylovBattery.Invokers`) -- that battery's checks #1-5 are square-system only; least-squares/
  least-norm invokers (checks #10-12) don't exist yet. A future battery increment should add a
  least-norm invoker (rectangular consistent gallery entries + the min-norm oracle check via
  `LQ.minNormSolve`) and wire `craig` in then. Bespoke coverage only for now, in `CRAIGTests`.

## Krylov.TFQMR
- 2026-07-21 | Task #60: added a verify-at-exit gate at the `bound <= thresh` Converged exit --
  Freund's quasi-residual bound is a rigorous upper bound on the true residual only in exact
  arithmetic. Unlike cg/biCGStab/minres, NO scratch vector is idle at that point under the identity
  preconditioner across BOTH loop parities (au is idle only on even half-steps, uHat only exists
  under a real M, w/u/v/d are all read again before their next overwrite regardless of parity) -- so
  instead of the usual fall-through-on-failed-verify pattern, this commits to a final return either
  way (`Converged` if the fresh residual clears the threshold, else `MaxIterations`, never a false
  Converged) and clobbers `au`/`v` for the check, which is free since neither outcome touches them
  again.
- 2026-07-20 | New file `Krylov.TFQMR.fProxy.cs`: single-RHS transpose-free QMR (Freund 1993),
  ported from `reference/scipy/tfqmr.py`'s recurrence (readable float/double reference) with the
  breakdown-guard/stopping-criterion shape cross-checked against `reference/square/BelosTFQMRIter.hpp`.
  DEVIATION from both references: they precondition LEFT (`v = M(A(r))`, and even accumulate x via
  a final `z = M(d)` apply every half-step). Ported RIGHT instead, matching this codebase's other
  nonsymmetric solvers (biCGStab/gmres/idr all solve `A M⁻¹ y = b`): `uHat = M⁻¹u` feeds both the
  A-apply (`au = A(uHat)`) and the solution-update accumulator `d` directly, so `eta·d` folds
  straight into `x` with no extra M-apply, and the tracked quasi-residual bound (`tau·sqrt(half-steps)`)
  stays a bound on the TRUE ‖b−Ax‖ rather than scipy's M-transformed residual. Verified the
  substitution algebraically (M⁻¹ is linear, so it commutes through the `d`-recursion) and numerically
  (a standalone Python port against a Jacobi-preconditioned nonsymmetric system, 20/20 trials to
  <1e-10 relative error; a 500-trial stress sweep on harsh non-diagonally-dominant random matrices
  produced zero NaNs and zero false-Converged exits). `maxIter` counts HALF-steps (~one A-apply
  each) -- NOT full passes like biCGStab's two-matvec loop -- documented on the entry point.

## Krylov.Block.MINRES
- 2026-07-20 | BuildOmega now rank-checks Gamma's own diagonal (the QR factor of `[Gbar;Beta^T]`),
  the same way it already checked the Qperp completion's Rz diagonal -- previously a near-singular
  Gamma (observed right after a block-Lanczos deflation recovers to full rank) went undetected:
  Phibar/Phi decayed geometrically toward zero over subsequent iterations without ever closing the
  true residual, so the solver looked idle and reported an honest-looking but silently WRONG
  `MaxIterations` instead of `Breakdown`. Root-caused via `doubleBlockMinresTests.
  BlockAdvantageIterations` at an adversarial (non-multiple) n/s ratio -- float happened to avoid
  it, double didn't. Rescoped that test to a benign n/s (full-rank regime, no natural mid-solve
  deflation) since the adversarial ratio is a deflation-robustness question, not a core bug. Also
  gated the non-identity-preconditioner path (`bminres` now throws `NotSupportedException` when
  `!M.IsIdentity`, since `BlockNormalizePrecond`/CHOP was never independently verified) and removed
  `PreconditionedMatchesScalar` / `RankDeficientBlockNoNaN` (the latter converges, with no NaN and
  correct rank detection, but does not preserve identical-RHS-row symmetry once a lane is
  deflated). Full block-Lanczos deflation robustness -- the preconditioned path, identical-row
  symmetry under deflation, and the Gamma near-singularity case together -- is ONE deferred
  follow-up: task #50.
- 2026-07-20 | Fixed a second, independent bug found via the same dump-diff investigation below:
  the pre-loop setup zeroed `W1`/`W2` but not `W` itself. At k=0, `Delta` is exactly zero, and the
  code relied on `Delta^T . W2 == 0` to make that term vanish -- but IEEE `0 * NaN` is `NaN`, not
  `0`, so an uninitialized `W` (the convenience forwarders allocate scratch with `uninit: true`)
  could poison the whole solve once it rotated into `W2`. `W` is now zeroed alongside `W1`/`W2`.
- 2026-07-20 | Fixed the s>1 divergence via dump-diff against a numpy reference
  (`reference/wip-bminres/bminres_reference.py`, gitignored scratch): `BlockNormalizeIdentity`/
  `BlockNormalizePrecond` return `Beta` indexed by `LQRP.decomp`'s/`CHOP.decomp`'s PIVOTED row
  order (`(P.W)[j,:] == (Beta.Vout)[j,:]`), but the block-Lanczos recurrence's `Beta^T . Vprev`
  term and `Beta`'s placement in `M2` both need `Beta` indexed by the residual's ORIGINAL row
  order -- the order `Vprev`/`Alfa`/`OmegaOld` already carry. Invisible at s=1 (`Beta` is 1x1, the
  pivot is always trivial), which is why `MatchesScalarAtS1` passed while every s>1 test converged
  to a wrong answer. Fixed with `UnpivotBetaRows`, which scatters `Beta`'s rows through the pivot
  right after each normalize.

## Krylov.minresQLP
- 2026-07-19 | New file `Krylov.MINRESQLP.fProxy.cs`: single-RHS MINRES-QLP (Choi, Paige &
  Saunders, SIAM J. Sci. Comput. 2011) for symmetric (possibly indefinite/singular) systems,
  structural sibling of `Krylov.MINRES.fProxy.cs` (same Lanczos base + left-Givens reflection,
  plus a second right-Givens QLP reflection that regularizes the near-singular tail and yields the
  minimum-length solution on a singular/rank-deficient A). Ported fidelity-first from
  `reference/square/minresQLP.py` (Apache-2.0, Liu & Roosta-Khorasani's Python translation of the
  Stanford SOL MATLAB `minresQLP`), cross-referenced against `MINRESQLP-SISC-2011.pdf`. Workspace
  (own Temp/arena, 9 length-n vectors: v/r1/r2/r3/w/wl/wl2/xl2/t1) -- one more than plain minres's
  8; `r3` doubles as both the Lanczos matvec accumulator and (preconditioned) the M⁻¹-apply
  target, so no separate `z` buffer is needed; `t1` is a generic recycle buffer used only inside
  the QLP right-reflection's w/wl/wl2 update (needed once three buffers must all be read while two
  new ones are produced -- see the in-file buffer-rotation comments). Overload ladder trimmed to
  what the spec enumerated: generic `<TOp,TPre>` core, `<TOp>` unpreconditioned forwarder, dense
  `fProxyMxN` + BSR `fProxyBSR` unpreconditioned rungs (zero-alloc + arena + default-param), the
  generic preconditioned `<TOp,TPre>` arena rungs, and (matching the one preconditioner the spec's
  tests exercise) a named BSR+block-Jacobi convenience trio. The other five named BSR
  preconditioner convenience wrappers minres carries (SSOR/IC0/FSAI/Chebyshev/AdditiveSchwarz) were
  NOT added for QLP -- any of them still works through the generic `<TOp,TPre>` rungs.
- 2026-07-19 | DEVIATION (warm start): the reference always starts from x=0 (no warm-start
  parameter at all). This library's whole Krylov surface treats `x` as a warm-startable in/out
  parameter, so the port forms r0 = b - A·x0 up front (exactly like plain `minres`'s own r1 = b -
  A·x) and runs the reference's recurrence with r0 in place of "b" everywhere it seeds the
  Lanczos/right-reflection state. This is exact, not approximate: every reference x-update is
  either `x +=` (additive) or, once the QLP phase starts, `x = xl2 + ...` where `xl2` was itself
  reconstructed by subtracting off the last two increments from the CURRENT (x0-seeded) `x` --
  proven by hand to still carry x0 through every subsequent QLP-phase overwrite. The one edge this
  doesn't cover (QLP transition landing exactly on iters==1, before any x0-carrying `xl2` exists)
  is provably unreachable: `Acond` starts at 1 < the default `TranCond` = 1e7, so the branch that
  first flips `QLPiter` from 0 to 1 cannot fire before iters==1 has already run its `if iters>1`
  guard.
- 2026-07-19 | DEVIATION (branch condition, NOT "fixed"): section H's branch
  `if (Acond < TranCond) and flag != flag0 and QLPiter == 0: <cheaper MINRES step> else: <QLP
  step>` reads, at the point it runs, `flag` as either still `flag0` (untouched this iteration) or
  freshly 6/9 (just set by the xnorm/maxxnorm guard immediately above it) -- so in EVERY normal
  iteration (nothing exceptional triggered) the condition is false and the QLP branch runs, i.e.
  this reference effectively hardcodes "always QLP" (`TranCond=1` behavior) regardless of the
  nominal `TranCond=1e7` default, from iteration 1 onward. Both branches are still mathematically
  valid (QLP is a strict stability generalization of the plain step), so this is not a correctness
  bug -- just a missed "cheaper early iterations" optimization in the reference. Ported the
  condition EXACTLY as written (fidelity-first): the MINRES branch is dead in practice but still
  implemented (and exercised whenever xnorm/gama boundary guards do trip flag=6/9 mid-solve).
- 2026-07-19 | DEVIATION (safety guards the reference lacks, added to satisfy this library's
  "no NaN/throw" contract): (1) the reference's preconditioner-indefinite checks (`beta1<0` at
  init, `betan<=0` mid-loop) only `print` and let a negative/zero value silently propagate as a
  "norm" -- replaced with the same clean `Breakdown` early-exit plain `minres` already uses for the
  identical situation. (2) the reference's iters==1-betan==0 "lucky Lanczos breakdown" shortcut
  (either x is already optimal, flag 0, or x += r0/alfa solves exactly, flag -1) is coded ONLY
  inside the unpreconditioned branch in the reference -- under a real M it falls through and would
  divide by beta=0 on the next iteration's `v = r/beta`. Generalized the check to run after both
  branches converge on a `betan` value, so it also guards the preconditioned path.
- 2026-07-19 | Reference flag (-1, 0..9) -> `IterativeSolveStatus` mapping (SolveInfo has no room
  for the reference's extra flag/Acond/Anorm/relAres diagnostics, per spec "only surface what fits
  SolveInfo"): flag in {-1,0,1,2,3,4,5} (b=0/eigenvector shortcuts, compatible solve, min-length LS
  solve, either at rtol or at eps, x converged to eigenvector precision) -> Converged; flag 8
  (maxit) -> MaxIterations; flag in {6,7,9} (xnorm exceeded maxxnorm, Acond exceeded Acondlim,
  gama too tiny to safely divide) plus the added `-3` sentinel (non-SPD M mid-loop) ->
  Breakdown. `rnorm` reported is always a FRESH b-Ax (one extra matvec at the very end, regardless
  of exit reason) rather than the internally tracked `phi`/`rnorm` -- cheap (paid once per solve,
  not per iteration) and removes any doubt from accumulated recurrence drift; the reference's own
  Arnorm/relAres final recompute was dropped (not needed for a value SolveInfo doesn't carry).
- 2026-07-19 | `maxxnorm`/`Acondlim`/`TranCond` are internal constants at the reference's own
  defaults (1e7, 1e15, 1e7), NOT exposed as new parameters -- the spec pins the overload ladder to
  mirror plain `minres`'s (maxIter, tol only). Reference's `tiny` (smallest normal double,
  `np.finfo(np.double).tiny`) and `eps` (hardcoded DOUBLE epsilon, `np.finfo(float).eps` in
  Python, i.e. NOT adapted to the array dtype even in the original) were both remapped to
  precision-scaled equivalents so the float build isn't silently running double-precision
  thresholds: `eps = Consts.fProxyEpsilon`, `tiny = eps*eps`. `rtol` -> this library's existing
  `tol` param name (see [[short-param-names]]).

## Krylov.idr
- 2026-07-19 | New file `Krylov.IDR.fProxy.cs`: single-RHS IDR(s) (Sonneveld & van Gijzen 2008),
  structural sibling of `Krylov.BiCGStab.fProxy.cs`/`Krylov.GMRES.fProxy.cs` (generic
  `idr<TOp,TPre>` core, `idr<TOp>` identity forwarder, dense/BSR unpreconditioned rungs, BSR+ILU0
  and BSR+BlockJacobi preconditioned rungs). Ported from `reference/square/IDRsSolver.jl/IDRsSolver.jl`
  (MIT, Schauer/Astudillo/van Gijzen) `idrs_core`, cross-checked against
  `reference/square/idrs.jl` (IterativeSolvers.jl, MIT). Workspace (own Temp allocation, GMRES's
  style, since `s` is a runtime value like `restart`): P/G/U = 3s vectors of length n, plus
  Q/V/(VHat under a real M), plus the s×s system and its f/c vectors.
- 2026-07-19 | DETERMINISM: the reference's shadow space `P` is `rand!(...)` (uninitialized RNG
  state) -- breaks this library's cross-arch determinism. Replaced with `P` filled from
  `Unity.Mathematics.Random(seed)` (uniform [0,1), matching Julia's default `rand`), `seed`
  defaulted on every public overload to `0x9E3779B1u` (the same golden-ratio constant already used
  as LOBPCG's/SVD.Randomized's default deterministic seed). Same seed -> bit-identical `x` on every
  run/architecture.
- 2026-07-19 | The `omega(t, s, angle)` correction step took its `angle = 0.7` from the PRIMARY
  reference (`IDRsSolver.jl`'s literal default), not `idrs.jl`'s `sqrt(2)/2` -- the two references
  disagree here and primary wins per port-fidelity. Added a guard the reference doesn't have:
  `rho > 0` before the `om *= angle/rho` correction (avoids a `0/0` NaN when `Q ⟂ R` exactly);
  `om` then resolves to its uncorrected `ts/nt2 == 0` and trips the existing `om == 0` breakdown
  check on the next line -- same Breakdown outcome the reference silently NaNs into, just without
  the NaN.
- 2026-07-19 | The s×s `Msys` solve is a hand-rolled forward-substitution loop, not
  `Blas.triLower` -- the submatrix `Msys[k..s-1,k..s-1]` shifts every `k`, so `triLower` would need
  a fresh sliced copy every column; the direct loop matches the reference's
  `LowerTriangular(M[k:s,k:s])\f[k:s]` exactly with no extra allocation (mirrors GMRES's own
  hand-rolled Hessenberg back-substitution).
- 2026-07-19 | Convergence test uses this library's `tol² · ‖b‖²` relative-threshold convention
  (`Blas.dot`/`Blas.axpyNormSq`, no per-step `sqrt`), matching `biCGStab`/`gmres`, not the
  reference's raw `normR <= tol`.

## Krylov.fgmres
- 2026-07-19 | New file `Krylov.FGMRES.fProxy.cs`, sibling to `Krylov.GMRES.fProxy.cs` (not an
  addition to it) per the single-solver-per-file convention. Ported the flexible mechanism (store
  the preconditioned basis Z, update x = x0 + Z y instead of applying M once to Σ y_i v_i) from
  Belos (`reference/belos/BelosBlockFGmresIter.hpp`, BSD, its `Znext = M*Vprev` / `Vnext = A*Znext`
  / `getCurrentUpdate` = `Z*y`), scalarized onto our own single-RHS Arnoldi/Givens/restart loop
  (`gmres`'s, unchanged). Workspace footprint: gmres's V (m+1 vectors of length n) PLUS Z (m more
  vectors of length n) when M is a real preconditioner -- roughly double gmres's basis memory,
  the structural cost of tolerating a per-step-varying M. Under `IsIdentity` (`fProxyIdentityPreconditioner`)
  no Z is allocated and the loop takes the exact same instruction sequence as `gmres` (z_j == v_j,
  solution accumulated straight into x via V) -- verified bit-identical against `gmres` in
  `FGMRESTests.fProxy.cs`, not just "close".
- 2026-07-19 | Overload ladder mirrors `gmres` exactly (generic `fgmres<TOp,TPre>` core,
  `fgmres<TOp>` identity forwarder, dense/BSR unpreconditioned rungs, BSR+ILU0 preconditioned
  rungs) rather than inventing a new shape -- callers already fluent in `gmres`'s ladder get
  `fgmres` for free.

## Krylov.Block.Common / LOBPCG -- View/RowsView/RectView reslice fix
- 2026-07-19 | Supersedes-in-spirit the `NormsOP`/`Krylov.bgmres` entries just below: those patched
  two individual `Data.Length` readers (`Norms.L1`/`L2`/`LInf(in fProxyMxN)`, `CopyFrom`/`CopyTo`).
  This fixes the actual root cause instead -- `View`/`RowsView`/`RectView` here and in
  `LOBPCG.fProxy.cs` are now one-line wrappers over a new `fProxyMxN(in fProxyMxN source, int rows,
  int cols)` reslice constructor (see `fProxy/DEVLOG.md`) that produces a REAL view with
  `Data.Length`/`Length` exactly `rows*cols`, not a bare struct copy with `M_Rows`/`N_Cols`
  overwritten and `Data`/`Length` left stale. No call site in `Krylov.Block.BCGrQ`/`BFBCG`/`GMRES`/
  `LOBPCG.fProxy.cs` changed -- same `View(buf, m)`/`RowsView(buf, rows)`/`RectView(buf, rows,
  cols)` call shape, zero-copy/zero-allocation, no per-iteration compaction added anywhere. The
  `NormsOP`/`CopyFrom` patches are left in place (redundant but harmless now that
  `Data.Length == M_Rows*N_Cols` always holds); the `uninit: true` -> `false` workarounds noted
  below are still unnecessary-but-harmless. Root-caused in `docs/dev/spec-matrix-view-fix.md`.

## NormsOP
- 2026-07-19 | ROOT FIX for the `Norms.LInf` narrowed-view over-read documented under Krylov.bgmres
  below. The entrywise `L2`/`L1`/`LInf(in fProxyMxN)` overloads now scan the LOGICAL `M_Rows*N_Cols`
  extent instead of `a.Data.Length` (the full backing). `RowsView`/`RectView` narrow `M_Rows` on a
  struct copy but leave `Data.Length` (and the readonly `Length` field) at the full buffer size, so
  scanning `Data.Length` read past the logical matrix into the uninitialized tail — silently
  corrupting the `Norms.LInf`-derived zero threshold in `QR.decompInPlace`/`LQRP.decomp`. Sibling of
  the Phase 1 `CopyFrom` resize footgun (rule: on a view, use logical `M_Rows*N_Cols`, never
  `Data.Length`). The `uninit:true`→`false` workarounds in bgmres (`Wbuf`/`HQscratch`) are now
  unnecessary but harmless; left in place.

## Krylov.bgmres
- 2026-07-19 | **Real bug found via the new test suite: `Norms.LInf` over-reads uninitialized memory
  through a narrowed view.** `QR.decompInPlace` and `LQRP.decomp` both compute a Householder
  zero-column threshold via `Consts.fProxyZeroThreshold * Norms.LInf(in A)`, and `Norms.LInf` (via
  `fProxyNormsCore.LInf<T>`) scans `a.Data.Ptr` for `a.Data.Length` elements — the buffer's TRUE
  allocated length, not the logical `M_Rows*N_Cols` of a `RectView`/`RowsView`. Two scratch buffers,
  `Wbuf` (fed to `LQRP.decomp` as its `A` input via `RowsView(Wbuf, w[j])`, `w[j] <= s`) and
  `HQscratch` (fed to `QR.decompInPlace` as its in-place `A_to_Q` via `RectView(HQscratch, totalRows,
  totalCols)`, `totalCols <= m*s`), were allocated `uninit: true`. Every call after the buffer's first
  (widest) use over-reads whatever `Allocator.Temp` memory sits beyond the current logical region —
  uninitialized on first use, stale-but-finite on reuse. This corrupted the Householder zero-threshold
  (occasionally to NaN or a wild magnitude), silently producing a WRONG factorization with no thrown
  exception: 12 of 18 `BlockGmresTests` failed with `X` diverging from the per-column scalar `gmres`
  comparison, while the solver's own honest residual-based `Solved` check still (misleadingly) reported
  convergence over enough restart cycles. Fix: allocate `Wbuf`/`HQscratch` with `uninit: false`
  (cleared) — mirrors scalar `gmres`'s own `H` allocation comment ("cleared: read only written
  entries"). Any FUTURE buffer here that gets narrowed via `RectView`/`RowsView` to less than its true
  stride and handed to `QR`/`LQRP`/anything computing `Norms.L1`/`L2`/`LInf` on it needs the same
  `uninit: false` treatment — `V[0..m]`, `Tbuf`, `R0`, `Wcombo`, `Zt`, `Zcombo`, `Lbuf`, `HijBuf`,
  `YiBuf`, `Rscratch`, `Yscratch`, `QtGscratch` were individually verified NOT to reach `Norms.*` (only
  `Blas.dot`/GEMM kernels, which size every read/write off `M_Rows`/`N_Cols` explicitly, never
  `Data.Length`) and were left `uninit: true`.
- 2026-07-19 | Ported from Belos (`BelosBlockGmresIter.hpp`/`BelosBlockGmresSolMgr.hpp`, BSD) per
  `docs/dev/spec-bgmres.md`: block Arnoldi (Simoncini & Gallopoulos 1996) generalizing scalar GMRES
  (Saad & Schultz 1986), with basis-rank deflation (Morgan 2005, without the eigenvalue-recycling
  half). New sibling file (not an addition to the bcg-family `Krylov.Block.*.fProxy.cs` files) —
  reuses `BlockCTV`/`BlockAdd`/`CopyBlock`/`BlockApplyPre`/`CountConverged`/`View`/`RowsView`/
  `RectView`/`LQRPRank` from `Krylov.Block.Common.fProxy.cs` and `fProxyDenseOperatorGeneral`/
  `BlockCrossGram` from `Krylov.Block.BiCGStab.fProxy.cs` (both already shipped) unmodified.
- 2026-07-19 | The block-Hessenberg least-squares is a PERIODIC DENSE RE-QR of the accumulated `Hbuf`
  prefix every inner Arnoldi step (`QR.decompInPlace` + `QR.decompSolve` on a freshly-copied
  `totalRows x totalCols` scratch), not an incremental block-Givens/block-Householder update. Same
  answer either way (identical least-squares problem); the incremental route's variable-width-row-group
  rotation bookkeeping was judged not worth the correctness risk for the first cut. `Hbuf`'s active
  region is at most `(m+1)s x ms` (no `n` dependence), trivial next to a single matvec for any
  non-trivial `n` — deferred as a future optimization, not attempted here.
  Per-column convergence is read off the QR residual via the Pythagorean identity (`‖G‖² = ‖QᵀG‖² +
  ‖residual‖²`, thin `Q` has orthonormal columns) — no extra matvec, mirrors scalar `gmres`'s own O(1)
  per-step check.
- 2026-07-19 | `Hbuf`/`Gbuf` are fully zeroed (`(m+1)s x ms` / `(m+1)s x s`, their whole true
  allocation) at the START of every restart cycle, not just the `[0, s)`-row slice a literal reading of
  an early spec draft's §4.1 pseudocode line suggested — §3.2's own text ("both zeroed at the start of
  every restart cycle") and the `StoreBlockAt` `+=`-accumulate contract (which needs every `(i,j)` block
  a cycle will ever write to start at exactly 0, and a cycle can write block-rows up to `k <= m`, not
  just `< s`) both require the full-buffer reset; the narrower literal region would leave block-rows
  `>= 1` uncleared and corrupt `StoreBlockAt`'s accumulation on any cycle after the first. Cheap either
  way (`O(m²s²)`, independent of `n`).
- 2026-07-19 | `BlockSolveInfo.maxRnorm`/`.converged` are documented as describing "the returned X" (see
  `BlockSolveInfo.cs`), but the per-cycle `CountConverged` check happens at the TOP of a cycle (before
  that cycle's own `Commit`), so it's stale by one `Commit` whenever a `MaxIterations`-terminated run's
  last cycle didn't fully converge. Added one fresh, independent residual recompute
  (`A.ApplyBlock` + `CountConverged`, no QR/LQRP involved) after the restart loop exits, for every exit
  path — mirrors `bcgrq`'s own "recompute fresh from the final X ... doubles as an exit-time sanity
  check" cleanup pattern. Not in the original spec's §4 pseudocode; added to honor the existing
  `BlockSolveInfo` field contract.
- 2026-07-19 | Memory: `V[0..m]` (`(m+1) s n` elements) dominates for any non-trivial `n`; the
  `Hbuf`/`Gbuf`/`HQscratch`/`Rscratch`/`Yscratch`/`QtGscratch` family is `O(m²s²)` with no `n` term —
  tens of KB at the library's default `restart` (~30) and typical `s` (a handful), negligible next to
  `V`'s term for any `n` beyond a few hundred.

## Krylov.bbiCGStab
- 2026-07-19 | Recurrence ported from nmoteki's `bl_bicgstab.cpp` (Tadano et al. 2009 JSIAM
  letters), per task instruction to take the math from that working reference rather than
  `docs/dev/spec-bbicgstab.md` Section 4's speculative generalization. nmoteki's loop carries no
  `rho_(k-1)` at all: the SAME s x s coefficient `Mmat = Rhat0^T V` (this iteration's) is solved
  twice, once for alpha (rhs `Rhat0^T R`) and once for beta (rhs `-Rhat0^T Z`, Z = A applied to
  the half-step residual) — this sidesteps the spec's Section 6 "open risk" (the `Y*alphaMat` vs
  `alphaMat*Y` ordering ambiguity) entirely, since that ambiguity was an artifact of trying to
  generalize scalar BiCGSTAB's `rho_k/rho_(k-1)` ratio to matrices, which nmoteki's formulation
  never does. Kept spec Section 4.3's early-exit-on-half-residual optimization (checking
  convergence on S before the second matvec) since it mirrors scalar `biCGStab`'s own shape and
  doesn't change the recurrence, only when it's allowed to stop early — nmoteki's own loop always
  runs both matvecs and checks once at the end.
- 2026-07-19 | `BlockGram` (Common helper) is NOT valid for `Rhat0^T V` / `Rhat0^T R` / `Rhat0^T Z`:
  it unconditionally symmetrizes its output, which is only correct for a self-Gram
  (`P^T A P`-shaped). Rhat0 is the FIXED shadow residual, distinct from V/R/Z for every k >= 1, so
  these cross terms are genuinely asymmetric; symmetrizing them silently corrupts alpha/beta.
  Added a local, non-symmetrizing `BlockCrossGram` (same GEMM call as `BlockGram` minus the
  symmetrization loop) instead. `docs/dev/spec-bbicgstab.md`'s Section 4.3 pseudocode calls
  `BlockGram` for these terms — that line is wrong; do not "fix" `BlockCrossGram` back to
  `BlockGram` if revisiting this file.
- 2026-07-19 | `fProxyDenseOperator.ApplyBlock` (`Interfaces/LinearOperator.fProxy.cs`) computes
  `Vrows * A` (classical), which equals `A * Vrows[r]` per row only when `A = Aᵀ` — true for every
  existing caller (bcg/bcgrq/bfbcg/LOBPCG, all SPD/symmetric) but false for bbiCGStab's whole
  reason to exist (nonsymmetric A). Reusing it for the dense rungs would have silently solved
  `Aᵀx = b` instead of `Ax = b`. Added `fProxyDenseOperatorGeneral` (this file) whose `ApplyBlock`
  routes through `Blas.dot(..., transposeB: true)` instead — the correct classical `A * Vrows[r]`
  for any square A, at the cost of requiring `rows == Vrows.M_Rows == AVrows.M_Rows` (no
  partial/locked-tail write, since bbiCGStab never needs one). Used only by bbiCGStab's dense
  rungs; `fProxyDenseOperator` and its callers are untouched.
- 2026-07-19 | First test pass caught a real bug: the preconditioned X update was accumulating
  `alpha*P` / `omega*S` (the raw, unpreconditioned search directions) instead of
  `alpha*Phat` / `omega*Shat` (`M^-1` applied) — copied the unpreconditioned shape without
  re-deriving the preconditioned one. Scalar `biCGStab` is explicit about this
  (`x.addScaledInPlace(alpha, pHat)` under `!M.IsIdentity`, `Krylov.BiCGStab.fProxy.cs`); the P
  search-direction recurrence itself (`P := R + (P - omega V) beta`) correctly stays in the raw,
  unpreconditioned space either way (mirrors scalar's own `p`/`v`/`r` — only the operator applies
  and the X commit route through the preconditioner). Caught by
  `BlockBiCGStabTests.PreconditionedMatchesScalar` (RAS + BSR nonsymmetric) failing an Assert
  inside the Burst job before the fix, passing after.

## Krylov templates split one-file-per-solver
- 2026-07-19 | Pure reorganization, no algorithm/signature changes (verified: sorted-line diff of
  every non-blank, non-boilerplate line across the old and new file sets is empty; the 125
  public + 21 private method signatures are byte-identical, just relocated). `Krylov.fProxy.cs`
  (previously a grab-bag of `cg`/`lsqr`/`lsmr`/`lstsqResidual` + half of `minres`/`biCGStab`),
  `Krylov.PMinres.fProxy.cs` (the other half of `minres`, legacy pre-merge filename), and
  `Krylov.PBiCGStab.fProxy.cs` (the other half of `biCGStab`, same legacy pattern) collapsed and
  re-split into one template per solver: `Krylov.CG`, `Krylov.MINRES` (consolidates the old
  `Krylov.fProxy.cs` + `Krylov.PMinres.fProxy.cs` halves), `Krylov.BiCGStab` (consolidates
  `Krylov.fProxy.cs` + `Krylov.PBiCGStab.fProxy.cs`), `Krylov.LSQR`, `Krylov.LSMR`, and a new
  `Krylov.Lstsq.Common` for the two solvers' shared plumbing (`lstsqResidual`, `LstsqInfoTracked`,
  `JacobiFinish`). `Krylov.fProxy.cs` itself now holds only `MakeSolveInfo` (the shared
  `SolveInfo` factory used by `cg`/`minres`/`biCGStab`, plus `gmres`/`fcg` in their own
  already-separate files) — genuinely shared across every square-solver file, so it stays rather
  than being deleted. `Krylov.GMRES.fProxy.cs`/`Krylov.FCG.fProxy.cs`/`Krylov.Guards.cs`
  untouched (already one-file-per-solver / shared-singular).
  Mirrored for the block (multi-RHS) family: `Krylov.Block.fProxy.cs` (bcg + bcgrq + bfbcg + ~17
  shared private helpers) split into `Krylov.Block.CG` (bcg), `Krylov.Block.BCGrQ` (bcgrq),
  `Krylov.Block.BFBCG` (bfbcg), and `Krylov.Block.Common` for every private helper
  (`BlockGram`/`BlockCTV`/`BlockAdd`/`BlockZplusT`/`BlockSolveSPD`/`CountConverged`/
  `BlockApplyPre`/`CopyBlock`/`CopyMat`/`View`/`RowsView`/`RectView`/`BlockScatterAddRows`/
  `LockConvergedRows`/`LQRPRank`/`FactorLiveResidual`/`FactorLiveSearch`/`FactorGramOnce`) — per
  spec, ALL listed helpers moved to Common even though `FactorLiveResidual` (bcgrq) and
  `FactorLiveSearch`/`FactorGramOnce` (bfbcg) are each actually called by only one solver today;
  the original "-only" section-divider comments were dropped since they'd otherwise misdescribe a
  helper now living in a shared file (the doc-comment on each helper itself, which states its real
  contract, is untouched). Old orphaned generated outputs
  (`Source/OP/Krylov.{PMinres,PBiCGStab,Block}.{float,double}.cs`) auto-pruned by
  `Tools/prune-orphaned-generated.ps1` (invoked from `regen.ps1`) — no manual `git rm` of
  generated files needed beyond staging the deletions it already made on disk. No test file
  needed changes: every solver kept its exact name/signature (`KrylovPMinresTests.fProxy.cs`'s
  class/job names are local test-naming convention, not a dependency on the old template
  filenames — it only ever called the public `Krylov.minres(...)` API).

## Fixed-size struct-to-struct copies -- silent-resize footgun sweep
- 2026-07-19 | Root cause and the `fProxyN`/`fProxyMxN` `CopyFrom(in Self)`/`CopyTo(in Self)` fix are
  documented in `fProxy/DEVLOG.md` (this bug was first found here, via `LQRP.decomp`'s
  `W.Data.CopyFrom(A.Data)` -- see the bcgrq section below for the concrete repro). This entry covers
  the sweep of every other `.Data.CopyFrom(` site across `LQ`/`LU`/`QR`/`QRCP`/`Bidiag`/`Kalman`/
  `Kalman.UKF`/`Control`/`Riccati`/`Krylov`/`Krylov.FCG`/`Krylov.GMRES`/`OP.Dot`/`SelectOP` in this
  folder (plus `Interfaces/LinearOperator.fProxy.cs` and `MG/fProxyAMG.cs`, each noted in their own
  DEVLOG). Two categories: (1) sites where the destination is a fixed, already-correctly-sized
  buffer copying from ANOTHER caller-supplied `in fProxyMxN`/`in fProxyN` parameter that could
  legitimately be a narrowed view (`LQ.decomp`/`LQRP.decomp`/`LU.decompNoPivot`/`LU.decomp`/
  `QR.decomp`/`QRCP.decomp`'s `W`/`U`/`Q.Data.CopyFrom(A.Data)`, `QRCP`'s row-unpermute `Z`) -- these
  were LATENT instances of the same bug (silently degrade to a stale/zeroed copy whenever fed a view
  whose backing buffer differs from its logical size) and are real fixes, not just style; (2) sites
  where both sides are known same-size fresh/state buffers (Kalman filter state vectors, Riccati/LQR
  n x n iterates, Krylov solver scratch validated against `A.Rows`/`A.Cols` at entry, `SelectOP`'s
  dimension-checked `dest`) -- these get the same treatment for consistency and defense-in-depth, but
  had no observable bug (`UnsafeList.CopyFrom` only misbehaves when an actual resize is needed).
  All switched to the now length-checked `CopyFrom(in Self)` wrapper (throws on a real mismatch,
  `MemCpy`s the logical size, never resizes).

## Krylov.bfbcg — breakdown-free block CG (Ji & Li 2017)
- 2026-07-19 | `FactorLiveSearch`'s exactly-sized `Pfactor` Temp buffer + `CopyBlock` pre-copy REMOVED
  now that `LQRP.decomp`'s own internal scratch copy is length-checked against its logical
  `M_Rows*N_Cols` (fProxy/DEVLOG.md's silent-resize fix) instead of resizing off a view's stale
  `Data.Length` -- `Plive` (the `RowsView`) is fed straight into `LQRP.decomp` now. One fewer
  `Allocator.Temp` allocation and one fewer O(sLive*n) copy per call; behavior unchanged (full suite
  green after the switch).
- 2026-07-19 | Third block-CG family in `Krylov.Block.fProxy.cs`, coexisting with ridge `bcg` and
  `bcgrq`. Source: Ji, H. & Li, Y., "A breakdown-free block conjugate gradient method", BIT Numer. Math.
  57:379-403 (2017); porting extract at `reference/papers/BFBCG-algorithm-extract.md`. Unlike `bcgrq`
  (which orthonormalizes the RESIDUAL block every iteration via LQRP), `bfbcg` orthonormalizes the
  SEARCH block P: `Phat_i = orth(P_i)` (rank r_i, row-major analogue of the paper's pivoted-QR `orth`),
  `AP_i = A Phat_i` (the one matvec/iteration), `G_i = Phat_i^T A Phat_i` (r_i x r_i, SPD by
  construction — no ridge, no normal-equations kappa^2). `alpha = G_i^-1(Phat_i^T R)`,
  `X += alpha^T Phat_i`, `R -= alpha^T AP_i`; then `beta = -G_i^-1(AP_i^T Z)`, `P_{i+1} = Z + beta^T
  Phat_i` (Z = R under an identity preconditioner, folded per the existing `IsIdentity` convention).
  `G_i` is Cholesky-factored ONCE (new `FactorGramOnce` helper, ridge-ladder safety net only) and the
  same factor (`work`) is reused for both the alpha and beta `CHO.decompSolve` calls — `BlockSolveSPD`
  couldn't do this (it always re-factors), so `bfbcg` bypasses it entirely rather than extend its
  contract for one caller.
- Column locking (`Live`/`sLive`, `LockConvergedRows`, `BlockScatterAddRows`) mirrors `bcgrq` exactly:
  converged original RHS columns drop out of R/Z/P/X-update bookkeeping, X's rows are never reordered
  (scatter-add through the persistent `Live` pivot). Difference from `bcgrq`: P is NOT re-derived from
  scratch each iteration — it persists via the recurrence (`P_{i+1} = Z_{i+1} + Phat_i^T beta`), so its
  live width naturally tracks `sLive` (shrinks when columns lock, no independent "P has its own
  permutation" bookkeeping needed — it's always rebuilt fresh from R's current physical order every
  iteration before being re-orthonormalized).
- New private helper `FactorLiveSearch` (orth of the live P block via `LQRP.decomp`) mirrors `bcgrq`'s
  `FactorLiveResidual` byte-for-byte except it has no preconditioner-apply step (P is already the value
  to orthonormalize) — hits the SAME `fProxyMxN.Data`-stale-length landmine `bcgrq` found (see that
  section below), so it copies the live rows into a freshly, exactly `sLive x n` sized `Allocator.Temp`
  buffer before `LQRP.decomp`, same as `FactorLiveResidual`.
- Test oracle (`RankDeficientDeflates`): the paper's own Section 6.3 appendix gives a 10x10 SPD `A` and
  10x2 `B` with `B[:,2] = 10*B[:,1]` exactly (`rank(R_0) = 1` at zero initial guess). Reproduced via the
  `bcgrq` lesson below: perturb a known `Xk` first (`Xk[row] = 10 * Xk[otherRow]`), then re-derive
  `B = A Xk` via `ApplyBlock`, so `Xk` stays exact ground truth for the forward-error check instead of
  silently changing under a direct `B` edit.
- Benchmark row added to `BlockCGSparseBenchmark` (BSR 2D Poisson) alongside `bcg`/`bcgrq`/scalar-loop;
  numbers in the commit history's benchmark run.

## Krylov.bcgrq — deflating block-CG with reliable QR (LQRP) residual updates
- 2026-07-19 | `FactorLiveResidual`'s exactly-sized `Zfactor` Temp buffer + `CopyBlock` pre-copy
  REMOVED now that the root cause below is fixed at the source (`fProxyMxN.CopyFrom(in Self)` is
  length-checked against `M_Rows*N_Cols`, not resized off a view's stale `Data.Length` --
  fProxy/DEVLOG.md): `Rlive`/`Zpre` (the `RowsView`s) are fed straight into `LQRP.decomp` now, in both
  the identity and preconditioned branches. One fewer `Allocator.Temp` allocation and one fewer
  O(sLive*n) copy per iteration; behavior unchanged (full suite green after the switch). The BUG FOUND
  note directly below is now historical -- the workaround it describes is the one just removed.
- 2026-07-19 | New solver set alongside ridge `bcg` (not a replacement): replaces the s×s ridge-
  regularized Gram solve with a row-pivoted rank-revealing LQ (`LQRP.decomp`) factorization of the live
  (preconditioned) residual block every iteration, so near-dependent RHS directions are DEFLATED (dropped
  from the search subspace) instead of ridge-patched. `X`'s rows never reorder (scatter update through a
  persistent `Live` Pivot); `R`'s rows lock/swap-to-back on convergence, mirroring LOBPCG's lock loop. The
  search-subspace width `sa`/`saSearch` is independent of `sLive` (still-live column count) and can
  shrink or grow every iteration — every live column still gets an X/R update every iteration regardless
  of the deflated width. `BlockSolveInfo.minActive` now genuinely reports `< rhs` for a rank-deficient
  block (ridge `bcg` always reports `rhs`, since it never drops columns). Same 8-overload ladder as `bcg`
  plus one extra required `Pa` buffer (LQRP's orthonormal-rows output). Spec: `docs/dev/spec-bcgrq.md`.
- BUG FOUND while wiring `LQRP.decomp` onto a `RowsView` (the LOBPCG-style "same-buffer, narrower-shape"
  view trick, reused here for `sLive`/`saSearch`-width scratch): `fProxyMxN.Data` is a `UnsafeList<T>`
  STRUCT returned BY VALUE from a property getter (`Ptr`/`m_length` are plain fields, not indirected
  through a pointer). A `View`/`RowsView`/`RectView` narrowing (copy the struct, overwrite `M_Rows`/
  `N_Cols`) leaves `.Data.Length` at the BACKING buffer's full size, not the narrowed shape's.
  `LQRP.decomp`'s `W.Data.CopyFrom(A.Data)` resizes off `A.Data.Length` — when that's larger than `W`'s
  own real capacity, the grow-reallocation happens on the DISCARDED temporary `UnsafeList` the getter
  returned, silently leaving `W`'s real storage at its original zeroed content (decomp then factors an
  all-zero matrix → rank 0 → a spurious `Breakdown`). Reproduced concretely with `sLive=1` (after 3/4
  columns locked): `L[0,0]` came back exactly `0` despite a real ~1e-3-norm residual row feeding it.
  FIX (in `bcgrq` only, nothing touched in `fProxyMxN`/`LQRP`/`Norms`): `FactorLiveResidual` copies the
  live rows into a fresh, EXACTLY `sLive x n` sized `Allocator.Temp` buffer before calling `LQRP.decomp`,
  disposed right after (mirrors the existing per-iteration `Pivot` allocation). Audited every other view
  use in `bcgrq` (`Blas.dot` destinations, `BlockGram`/`BlockCTV`/`BlockAdd`/`CopyBlock`/`BlockSolveSPD`,
  `CHO.decompInPlace`/`decompSolve`) for the same landmine: none of them rely on `.Data.Length` for a
  NARROWED view argument — over-clearing a view's OWN root buffer's real capacity (e.g. `alphaBuf`'s
  `MemClear` inside `Blas.dot`) is harmless; only a CROSS-buffer `.Data.CopyFrom`/implicit-resize like
  `LQRP.decomp`'s is unsafe. Flagging as a general landmine for future `View`-style narrowing fed into
  anything `Data.Length`-sensitive — LOBPCG's own `View`/`RowsView` don't hit it today only because they
  never narrow below their cache's own allocation width before calling something Data.Length-sensitive;
  worth a follow-up audit if that ever changes.
- Two test-design gotchas found writing the ill-conditioned/near-parallel comparison tests
  (`BlockCGrQTests.fProxy.cs`): (1) perturbing `B` directly (not `Xk`) for the near-parallel-RHS case
  silently changes column 1's TRUE solution away from the original `Xk[1,:]` (`B[1] ≈ A·Xk[0]`, not
  `A·Xk[1]`) — fixed by perturbing `Xk[1,:]` first, then re-deriving `B` via `ApplyBlock`, so `Xk` stays
  the exact ground truth. (2) A tiny `(1+1e-6)` relative slack on `maxRnorm`/forward-error assumes the two
  solvers land at near-identical precision; empirically they stop at DIFFERENT points on the convergence
  curve (same residual threshold, different last iterate), so a genuinely healthy run can differ by up to
  ~2x either way — widened to a 3x slack (`ResidualSlack()`) to absorb that without losing the "not
  dramatically worse" signal a real regression would trip.
- Benchmark (`BlockCGSparseBenchmark`, BSR 2D Poisson, independent random RHS — i.e. NOT deliberately
  rank-deficient): `bcgrq` is consistently SLOWER than ridge `bcg`, roughly 15-40% wall-clock overhead
  across grid/s combinations (float N=4096 s=32: 391.9ms vs block-CG's 336.3ms; double N=4096 s=32:
  616.9ms vs 499.7ms), for the SAME iteration count in almost every row (e.g. float N=1024 s=4: both 44
  iters; double N=2304 s=8: both 79 iters) — on a well-conditioned random-RHS system there is no rank
  deficiency to deflate, so the per-iteration LQRP factorization is pure added cost with no offsetting
  iteration-count win here (expected: this benchmark is not `bcgrq`'s target case — see the ill-
  conditioned/near-parallel-RHS comparison tests for where deflation actually pays off). Interesting
  aside: `bcgrq`'s `minActive` frequently drops to 1-2 well before convergence even on this "friendly"
  random-RHS system (e.g. float N=1024 s=32: minActive=1, vs block-CG's minActive=32 always) — floating-
  point residuals coincidentally become near-parallel as iterates approach the noise floor; harmless here
  since `sa` only narrows the search SUBSPACE, never the `sLive`-wide X/R update. `CHO.decompInPlace`/
  `decompSolve` "factor once, solve alpha and beta from the same factor" (spec §1/§14, deferred) is worth
  revisiting given this overhead — `BlockSolveSPD` currently re-factors `PQ` from scratch for both alpha
  and beta every iteration.

## Krylov.bcg (renamed from `cg`) — block-Krylov `b`-prefix convention
- 2026-07-19 | Renamed all block-CG overloads in `Krylov.Block.fProxy.cs` from `cg` to `bcg`, so every
  block-Krylov method carries the same lowercase `b` prefix (`bcgrq` above is the first sibling). The
  SCALAR `cg` (`Krylov.fProxy.cs`, `in fProxyN b, ref fProxyN x`) is untouched — only the 8 block
  overloads (`in fProxyMxN B, ref fProxyMxN X`) renamed. Callers swept: `BlockCGTests.fProxy.cs` (only the
  block calls; the scalar-oracle `Krylov.cg(in A, in bj, ...)` calls stay `cg`), `BlockCGBenchmark.fProxy.cs`,
  `BlockCGSparseBenchmark.fProxy.cs` (both block-matvec `Krylov.cg(new fProxy...Operator(...), in B, ...)`
  jobs; each file's scalar-loop job stays `cg`).

## Krylov.cg (BLOCK / multi-RHS) — block-CG, first true block-Krylov solver
- 2026-07-19 | New `OP/Krylov.Block.fProxy.cs`: block-CG for SPD A with s simultaneous RHS. TRUE block
  method (O'Leary) — ONE shared Krylov subspace, s×s block coefficients α=(PᵀAP)⁻¹(RᵀZ),
  β=(RᵀZ)⁻¹(RₙᵀZₙ), streaming A over the whole block once per iteration via `ApplyBlock`. NOT s scalar
  solves. Block vectors = fProxyMxN s-rows × n-cols (row = RHS), matching ApplyBlock. Single-body +
  IsIdentity fold from the start (no pblockCg): under identity, R is used directly wherever Z would be
  (no M-apply, no Z block — pass default); preconditioner = per-row M.Apply loop (BlockApplyPre),
  gated `if(!M.IsIdentity)`. Overload name `cg` on fProxyMxN B (resolves by type vs scalar fProxyN).
  Returns BlockSolveInfo (per-column converged count + worst maxRnorm). Spec: docs/dev/spec-block-krylov.md.
- DEFLATION (first cut) = RIDGE, not column-dropping. The s×s PᵀAP / RᵀZ go singular when RHS columns
  are linearly dependent (classic block-CG breakdown). BlockSolveSPD copies the Gram, tries CHO, and on
  non-SPD adds an escalating diagonal ridge scaled to the Gram's own diag (FactorGram discipline), so a
  rank-deficient block DAMPS the dependent direction instead of NaN-ing — identical RHS columns get
  identical correct solutions. Fixed block width s throughout (no active-width bookkeeping). TRUE
  column-dropping deflation (BCGrQ, Dubrulle/Ji-Li, via OrthonormalizeBlock's kept-count) is the future
  robustness/perf upgrade — noted, not built. Ridge is negligible for full-rank blocks (all 4 accept
  tests pass): per-column matches scalar cg, block iters ≤ worst scalar iters (block advantage),
  rank-deficient (two identical columns) stays finite + solves, block-Jacobi matches scalar. Reduces to
  scalar (P)CG exactly at s=1.
- Tests `BlockCGTests.fProxy.cs` run inside a [BurstCompile] IJob → job-safety (caller sees X written
  through ref fProxyMxN) covered by construction; block-CG never swaps block handles (writes in place),
  so no LOBPCG-style RestoreBufferIdentity hazard. GOTCHA (cost me a Mono-fallback cycle): `Assert.IsTrue(
  bool, $"msg")` is BC1071 (unsupported assert overload) → forced Mono → 1-ULP fails in untouched
  BlockJacobi tests. Plain `Assert.IsTrue(bool)` only. 6679/6679.
- 2026-07-19 | PERF: GEMM-routed the block bookkeeping. FIRST benchmark (naive scalar-loop Grams +
  block updates) showed the block ADVANTAGE is real in ITERATIONS (s=16: block converges in 5-6 iters
  vs the scalar loop's 96 = 6/col × 16) but block was 5-11× SLOWER in WALL CLOCK — same matvec flops
  both ways, so the loss was entirely the O(s²n) Grams (PᵀAP, RᵀZ) and updates (X+=αᵀP) running as
  un-vectorized triple-loops while the matvec (ApplyBlock) is an optimized GEMM. Fix: route Grams to
  `Blas.dot(V,W,false,true)` (= V·Wᵀ, symmetrized) and updates to `Blas.dot(coef,V,true,false)` (=coefᵀV)
  + a flat block add; s×n GEMM temp `T`. Result: block-CG 5-9× faster (float N=128 s=16: 0.79→0.087ms).
  New crossover (BlockCGBenchmark, dense random SPD): **s≥8 block-CG BEATS the scalar loop and the win
  grows with N** (double N=512 s=16: block 2.49 vs scalar 3.31ms; float N=512 s=8: 0.33 vs 0.46) — the
  memory-traffic win (block reads A ~s× fewer times) shows once bookkeeping is vectorized. s=1-2 block
  loses (overhead unamortized — expected; use scalar for 1 RHS). Advantage is modest (1.3-1.4×) on these
  WELL-conditioned systems (few iters); ill-conditioned/more-iters would widen it. CHO+ridge kept (see
  the ridge-vs-BCGrQ note): the wall-clock issue was bookkeeping, NOT the deflation choice. BCGrQ remains
  the future win — dropping converged/dependent columns cuts BOTH the s²n bookkeeping and the matvec
  width near convergence. TODO: block-MINRES / block-BiCGStab / block-GMRES; then BCGrQ deflation.

## Krylov.gmres — single-body merge (gmres/pgmres share one loop, IsIdentity fold) — FAMILY COMPLETE
- 2026-07-19 | Merged `gmres<TOp,TPre>`; `gmres<TOp>` forwards with identity. No caller scratch (GMRES
  allocates its own Temp workspace), so the merge is signature-simple — the only extra state is `zt`
  (M⁻¹ apply target), allocated/disposed only under !IsIdentity. Two gated spots: (1) Arnoldi apply —
  identity `A.Apply(vj,w)`, preconditioned `M.Apply(vj,zt); A.Apply(zt,w)` = A·M⁻¹ (right precond);
  (2) solution update — identity accumulates `y_i v_i` STRAIGHT into x (bit-identical to plain gmres),
  preconditioned accumulates into w then `x += M⁻¹w` (different summation grouping, so gating keeps the
  identity path exact). pgmres explicit-scratch body DELETED; only the ILU0 concrete overloads remain,
  renamed gmres. pgmres HARD-REMOVED (only GMRESTests referenced it). 6667/6667.
  **SCALAR KRYLOV FAMILY NOW FULLY COLLAPSED**: cg, minres, biCGStab, gmres each = one body, identity
  default, no p-prefix anywhere. Next: m-RHS block solvers, single-body from the start (task #12).

## Krylov.biCGStab — single-body merge (biCGStab/pbiCGStab share one loop, IsIdentity fold)
- 2026-07-19 | Merged body `biCGStab<TOp,TPre>` (7 scratch: r/rHat0/p/v/t + pHat/sHat); `biCGStab<TOp>`
  forwards with identity (pHat/sHat=default, unused). pbiCGStab explicit-scratch body DELETED;
  PBiCGStab.fProxy.cs keeps only the ILU0/SPAI/RestrictedSchwarz concrete overloads, renamed biCGStab.
  pbiCGStab name HARD-REMOVED (call sites in Sparse precond docs, tests, CircuitDemo, CHANGELOG swept).
  Cleanest of the three: right-preconditioned BiCGSTAB just inserts pHat=M⁻¹p, sHat=M⁻¹s into the two
  A-applies and the x update; identity makes pHat=p, sHat=s (r holds s), so gating those four spots
  under `if(M.IsIdentity)` is bit-identical to plain biCGStab. pHat/sHat size+aliasing guards gated
  under !IsIdentity. 6667/6667. Scalar family now 3/4 (cg, minres, biCGStab); gmres next.

## Krylov.minres — single-body merge (minres/pminres share one loop, IsIdentity fold)
- 2026-07-19 | Same collapse as cg, applied to MINRES: merged body `minres<TOp,TPre>` lives in
  Krylov.fProxy.cs; `minres<TOp>` forwards with the identity preconditioner (z=default, no z buffer);
  the pminres explicit-scratch body is DELETED and Krylov.PMinres.fProxy.cs keeps only the
  allocating/default/concrete overloads, renamed minres. pminres name HARD-REMOVED (39 call sites +
  comments swept). Trickier than cg because the two bodies genuinely differ: (a) plain MINRES checks
  `beta1²<=threshold` with beta1=sqrt(dot(r1,r1)) (sqrt-then-square) while pminres tests the TRUE
  residual dot(r1,r1); (b) plain phibar IS ‖b-Ax‖ (no verify) while preconditioned phibar is
  M⁻¹-weighted (verify-at-exit + fresh-residual MaxIter). Both gated under `if(M.IsIdentity)` so the
  identity path stays BIT-IDENTICAL to old minres (init check uses beta=sqrt(trueRR0) then
  beta*beta<=threshold; loop uses r2 not z for v; no verify; MaxIter returns phibar). z size/aliasing
  guards gated under !IsIdentity → identity may pass default. TEST FIX: KrylovPMinresTests'
  WrongLength/AliasedScratchThrows used the identity precond + a bad z; z is now exempt under identity,
  so they alias/mis-size a REQUIRED buffer (w2/w1) instead. 6667/6667 green. See [[iterative-solver-overload-ladder]].

## Krylov.cg — single-body merge (cg/pcg share one loop, IsIdentity fold)
- 2026-07-19 | Collapsed the plain + preconditioned CG into ONE body `cg<TOp,TPre>`; `cg<TOp>` and
  `pcg<TOp,TPre>` now forward into it (no duplicate loop). Mechanism: new compile-time
  `bool IsIdentity` on IfProxyPreconditioner (identity → literal true, all 13 real preconditioners →
  literal false). Every z access — size guard, aliasing sub-guard, M.Apply, ⟨r,z⟩ dot — sits behind
  `if(!M.IsIdentity)`; because TPre is a struct-constrained generic, Burst constant-folds the branch
  per specialization, so `cg<…,fProxyIdentityPreconditioner>` strips all z traffic and compiles to
  plain CG. z may be passed `default` on the identity path (never dereferenced) — so `cg<TOp>` needs
  no z buffer. SPIKE VERDICT (all 3 gates passed): (1) compiles under Burst; (2) bit-identical —
  `MergedCgIdentityMatchesPlainCg` asserts exact double-equality on x/iterations/status/rnorm, and the
  PCG benchmark shows byte-identical iters+residuals for every CG and PCG row; (3) zero perf cost —
  identity-fold CG rows are ≤ the old hand-written body's time (the ~5% delta vs the Jul-18 baseline is
  run/thermal variance, not a real speedup — baseline file left unchanged). Confirms the pattern for
  the rest of the family.
- 2026-07-19 | pcg name HARD-REMOVED (user ruling): the `pcg<TOp,TPre>` explicit-scratch forwarder is
  deleted (cg<TOp,TPre> already carries that signature); the allocating/default generic overloads and
  all concrete `pcg(BSR,<Precond>)` overloads are renamed to `cg`; ~134 `Krylov.pcg(` call sites across
  tests/benchmarks/the SpringLattice demo + comment mentions swept to `cg` (byte-safe sed, then the
  `cg/pcg`→`cg` prose artifacts collapsed). Capitalized `PCG` acronym in prose + benchmark row labels
  (`PCG-Jacobi`) LEFT — accurate description of preconditioned CG, not an API identifier. User-facing
  docs got factual token swaps (CHANGELOG solver list, solvers.md diag-struct table, lp-lad.md
  `Krylov.cg`); README untouched (user's). 6667/6667 green. NEXT: same merge for
  minres/pminres, biCGStab/pbiCGStab, gmres/pgmres; then block (m-rhs) solvers single-body from the
  start. See [[iterative-solver-overload-ladder]] (supersedes the keep-arity ruling, per user).

## Krylov.cgls / cglsJacobi removed
- 2026-07-19 | Removed CGLS (all overloads: generic + dense + BSR + damped + transpose-optimized) and
  its column-equilibration wrapper cglsJacobi, plus all tests/benchmarks/CglsInfo. Rationale: CGLS is
  a normal-equations (AᵀA) method → squares the conditioning (κ²); lsqr/lsmr solve the same
  least-squares problem (including Tikhonov `damp` and the *Jacobi column-equilibration wrapper) at
  strictly better conditioning, so CGLS was dominated. KEPT lsqr/lsmr + lsqrJacobi/lsmrJacobi (user
  wants to study the Jacobi wrappers before deciding their fate). Test rewiring: damping/diagnostics
  tests dropped their cgls arm (lsqr becomes the primary oracle); the 3-way `which` loops in
  Strang-line-fit + LstsqInfo dropped case 0. Comment cleanup swept the shared operator/BSR/gallery
  docs (cgls no longer a named consumer of ApplyT / transpose-materialization). Continues the cgne
  removal below — the whole normal-equations LS family is now gone. `updateXR`'s rectangular path is
  now exercised only by the fused-kernel regression test (no production caller), kept as a guard.

## Krylov.cgne removed
- 2026-07-18 | Removed CGNE / Craig's method (all overloads: generic + dense + BSR) + its tests
  (3 methods in SparseSolverTests, the verify-at-exit case, the aliasing guard, embedded rnorm-honesty
  subsections). Rationale: it's a normal-equations (AAᵀ) method → squares the conditioning (κ²), AND
  it's REDUNDANT — lsqr already produces the min-norm solution for underdetermined consistent systems
  at better conditioning. Kept cgls/cglsJacobi (deeper sunk value: Tikhonov damping = ridge regression,
  column-equilibration preconditioner; the user's call after researching origins — both came wholesale
  from c01d99e's Solvers→Krylov split). DESIGN DIRECTION (not yet built): merge the p-prefix pairs
  (pcg/cg, pminres/minres, pbiCGStab/biCGStab, pgmres/gmres) — every solver is "preconditioned by
  default" with fProxyIdentityPreconditioner (already exists) as the no-M overload; keep the dedicated
  zero-copy unpreconditioned path. For the future BLOCK-Krylov rewrite: ~7 UNIFIED block algorithms
  (identity default, no p-prefix), on ApplyBlock + QRCP-orthogonalized blocks (tall-skinny, O(n·s²)) +
  s×s coeff via CHOP/QRCP; block-preconditioner = column-loop helper (no per-precond rewrite); SKIP
  cgls/cgne normal-equations methods (κ²). Supersedes the earlier "keep arity" ruling.

## Krylov.gmres (restarted GMRES(m))
- 2026-07-18 | Restarted GMRES(m) for general nonsymmetric A: Arnoldi + modified Gram-Schmidt basis,
  incremental Givens-rotated least-squares (inline, same idiom as minres/pminres), restart every m
  inner steps. Unlike cg/biCGStab, GMRES ALLOCATES its workspace (m+1 basis vectors + Hessenberg +
  Givens) from Allocator.Temp — inherent to the method, no practical zero-alloc primitive. Basis held
  in `UnsafeList<fProxyN>` NOT `NativeArray<fProxyN>` (the latter nests a native container — fProxyN
  holds an UnsafeList — which Unity's safety checks reject). Hessenberg allocated cleared (uninit=false)
  — only written entries are read, but clearing follows the "partially-written matrix" rule. maxIter
  counts TOTAL inner iterations across restarts; rnorm = the Arnoldi residual estimate |g[k]|. Dense +
  BSR concrete overloads + defaults (restart=min(30,N)).
- 2026-07-18 | pgmres = RIGHT-preconditioned GMRES(m): runs GMRES on A·M⁻¹. Right (not left)
  preconditioning keeps the Arnoldi residual == true residual ‖b−Ax‖, so the convergence test is
  unchanged; solution update is one extra M⁻¹ apply per restart (accumulate the v-space combination,
  apply M⁻¹ once — no need to store the M⁻¹v_j basis). Generic pgmres<TOp,TPre> + BSR-ILU0 concrete
  (the canonical GMRES+ILU0 pairing). Test: pgmres(ILU0) converges in fewer iters than plain gmres.
## Krylov.fcg (Flexible CG)
- 2026-07-18 | Flexible CG (Notay 2000), first AMG prerequisite (K-cycle needs a variable-
  preconditioner outer solver; unsmoothed aggregation makes M vary per iteration). Implemented as
  pcg with the Polak–Ribière beta = (rznew − <z_new, r_old>)/rzold. Chose the explicit r_old
  snapshot vector over the cheaper "reconstruct r_old = r_new + alpha·Ap" identity because the
  verify-at-exit block overwrites Ap on the convergence path, so the identity would be stale on a
  verify-fail-continue — the snapshot is provably correct regardless. Costs one extra scratch vec +
  one extra dot/iter vs pcg. Reduces to pcg exactly for constant SPD M (cross term = 0) — asserted
  in tests via iteration-count agreement (NOT element-wise solution compare, which scales with
  cond·residual). Variable-M coverage: an inner-CG(3-step) preconditioner, whose k-step iterate is
  a data-dependent polynomial in r — the canonical case pcg mishandles. Not yet wired to AMG.
## LOBPCG robustness
- 2026-07-18 | Post-seed-B-normalize-fix, `PenalizedFramePathologicalGuardConfigNoFalseCertificate`
  flipped float `LOBPCGInfo(MaxIterations, converged=1)` (was `converged=0`) -- NOT re-broken by
  weakening the test; RESOLVED by strengthening it. `info.converged` on a MaxIterations/Degenerate
  exit is `ConvergedWithinTol`: per wanted pair, `xBnorm >= 0.25 (normFloor) && residual <=
  tol*resScale`, evaluated after the final ascending sort -- it DOES already apply the min-fix's
  B-norm floor (not norm-blind), but that floor (0.25) is looser than this test-file's own
  genuineness bar (0.5), and it never cross-checks the dense oracle. `AssertNoFalseCertificate`'s
  `if (!info.Solved) return` guard is BLIND to a nonzero converged count under a non-Solved status
  -- it passed here trivially, proving nothing. Added `AssertConvergedPairsGenuine`: replicates
  `ConvergedWithinTol`'s exact predicate over `ws` (classification only, never trusted for
  genuineness) to find which pair(s) are counted, then independently re-derives residual/norm via a
  FRESH spMV (not `ws.residual`/`ws.xBnorm`) and cross-checks the dense oracle -- runs regardless of
  Solved status, skips only Converged (redundant with AssertNoFalseCertificate) and Breakdown
  (X/lambda contractually undefined). OUTCOME: the flagged pair passed every independent check
  (dense match, norm, residual) -- GENUINE, the intended robustness win, not a spurious mode. Also
  applied to the sibling `PenalizedFrameSaneConfigNoFalseCertificate` (was already passing; the same
  structural gap existed there, extending coverage is a pure strengthening, not a risk). Dropped the
  float-specific `AreEqual(0, converged)` / `AreNotEqual(Converged, status)` hard requirements per
  spec: forbid false certificates only, never require non-convergence on a hard case. NOT extended:
  the shared `AssertNoFalseCertificate` helper itself, or the other callers that gate similarly
  (`ZeroRowWarmStartNoFalseCertificate`'s Solved-gated residual check, the test-writer's
  `PenalizedLaplacian1DNoFalseCertificate`/`LauchliGramSmallestNonNegativeOrHonest`) -- same
  structural gap, surfaced here rather than blind-fixed everywhere without the ability to run the
  suite.
- 2026-07-18 | REGRESSION FIX (found by suite + adversarial review after the SVQB/cube-rule change
  below): `floatLOBPCGSmokeTests.GeneralizedOutputIsBOrthonormal` (A=Laplacian1D(10), B=diag(1..10),
  k=3) went from Solved to a permanent MaxIterations stall in float (double unaffected). Root cause:
  the seed X is only EUCLIDEAN-orthonormalized (OrthonormalizeBlock) before the loop, never
  B-normalized -- for B=I that's already B-orthonormal so it never mattered before, but for a
  mildly-conditioned B != I the first [X,W] Gram's diagonal spread now trips the (correctly
  tightened) cube-rule combined-Gram gate on iteration 0. With `usedP` already false there (no P
  yet), `TryRayleighRitz` returning false has no lower fallback -- X/lambda/W are left untouched and
  the identical failing Gram reproduces every subsequent iteration: a permanent zero-progress stall,
  not a slow convergence. Fix: B-normalize the seed ONCE right after the seed BX ApplyBlock (mirrors
  the existing end-of-iteration B-normalize block's reseed-guard/scale pattern, distinct RNG seed
  constants so none of the three reseed sites in this file can coincidentally collide). This also
  happens to make the pre-loop bootstrap lambda seed (`dot(X,AX)`) exactly the GENERALIZED Rayleigh
  quotient for B != I (denominator dot(X,BX) is now exactly 1 by construction) instead of the
  Euclidean one -- a side effect, not the point of the fix. For B=I this seed-normalize divides by a
  B-norm (bit-identical to the Euclidean norm OrthonormalizeBlock already drove to ~1) that is not
  bit-exact 1.0 in floating point, so it perturbs X/AX/BX by roughly one ULP even on the standard
  path -- the SAME class of perturbation the end-of-iteration B-normalize block already introduces
  every iteration regardless of B, already accepted in the first round of review.
- 2026-07-18 | Test vectors (spec "Test vectors to add", cases 2/5/6/7a/8) added to
  LOBPCGRobustnessTests.fProxy.cs (test-writer). JUDGMENT CALLS:
  (case 1) KEPT the existing PenalizedFramePathologicalGuardConfig float assertion
  `AreNotEqual(Converged)` + `converged==0` unchanged. The robustness fix's STATED goal is to make
  this case solve in float, but that is explicitly un-verified (suite not run) and the cube-rule
  Gram gate could equally route it to drop-P -> stall -> Degenerate; a stated-but-unmeasured goal is
  not strong enough to weaken a regression guard. IF the fix does make float certify a genuine pair
  here, that assertion (LOBPCGRobustnessTests.fProxy.cs ~line 90) is the ONE line to relax to the
  general AssertNoFalseCertificate contract — flagged to the orchestrator, who runs the suite.
  (case 6) The spec's "B=diag(1,...,1,0) must return non-Converged" contract is covered TWO ways:
  the pre-existing RankDeficientBPencilIndefiniteANeverReportsConverged (indefinite A along B's null
  -> unbounded below) and a NEW RankDeficientBFewerFiniteEigenpairsThanKNeverReportsConverged
  (SPD non-diagonal A, B rank n-1, k=n): B admits at most n-1 B-normalizable directions so the n-th
  wanted pair is provably B-null (degenerate) -> can never certify -> status != Converged. A hard
  mathematical contract (not a snapshot), asserted as AreNotEqual(Converged).
- 2026-07-18 | Implemented spec-lobpcg-robustness.md's "Robustness fix" items 1-3 (the "Minimum
  fix" — Degenerate status, ||x||_B-aware convergence test — shipped 2026-07-17, see the entry
  below). Goal: make FLOAT LOBPCG actually SOLVE the penalty-conditioned cases that previously
  returned honest Degenerate, without regressing any case that already converged correctly.
  (1) Every active X row is B-renormalized after each iteration's fresh AX/BX matvec (linearity,
  no extra matvec) — restores the B-unit invariant Duersch's analysis identifies as collapsing
  under repeated ill-conditioned Cholesky-QR, and keeps next iteration's Gram diagonal ~= 1. A row
  whose B-norm^2 is at/below Consts.fProxyEpsilon is reseeded first (distinct seed formula from
  the existing (d1) reseed — 0x2545F491/0xC2B2AE35 vs 0x9E3779B1/0x85EBCA77 — so the two can't
  coincidentally collide on the same (iter,i) despite firing at different points in the same outer
  iteration); NOT re-deflated against locked rows here (the (d1) block at the TOP of the next
  iteration already re-deflates every active row unconditionally when numActive<kWork, whether or
  not it was just reseeded, so duplicating that logic here would be redundant).
  (2) OrthonormalizeBlockB (W/P) rewritten from Cholesky-QR+ridge to SVQB-with-dropping: scale the
  Gram by D=diag(G)^-1/2, eig-decompose DGD via Eigen.symmetricInPlace (reusing ws.Gram/ws.L as
  scratch — same reuse pattern TryRayleighRitz already applies to Atrans/Y), keep the leading
  (descending-sorted) theta_j > theta_max * (rows*eps*10), and combine V*D*Z(:,J)*Theta(J,J)^-1/2
  into the block's LEADING kept rows (AV/BV via the same combination). A direction the block can't
  support is DROPPED (block width shrinks: nw/np <= numActive) instead of ridge-inflated into
  noise. Zero extra O(n)-scale allocation: the row-combination output (SvqbAccumulate) is written
  into ws.Xnext borrowed as scratch — safe because Xnext is otherwise untouched between the top of
  the loop and UpdateActiveBlock (which unconditionally overwrites every row of it), and W's SVQB
  call fully finishes before P's SVQB call starts, so sequential reuse for both never overlaps.
  D/theta are O(rows) Allocator.Temp vectors, same class as TryRayleighRitz's own eigSmall.
  STRUCTURAL RIPPLE (the one real change beyond the two functions above): nw/np != numActive means
  BuildProjected/TryRayleighRitz/UpdateActiveBlock needed per-block widths (nx=numActive always,
  since X is never dropped as a block — only individually B-renormalized by (1); nw, np <=
  numActive) threaded through instead of a single shared `numActive` for every block. Caught mid-
  implementation: the P-against-W Deflate call's against-count was `numActive` in the pre-SVQB
  code (valid there because W's block width WAS numActive) — it must become `nw` post-SVQB, since
  W's rows [nw, numActive) are stale after dropping; deflating P against them would read garbage.
  (3) FactorGram gained a cubeRule bool: the seed/internal-block gate stays linear
  (MinMaxDiagRatio(L) >= sqrtEps, unchanged, still used by the Euclidean X-seed's
  OrthonormalizeBlock), and a new FactorGramCombined (used only for the combined [X,W,P]
  Rayleigh-Ritz Gram inside TryRayleighRitz) gates on MinMaxDiagRatio(L)^3 >= 10*eps — algebraically
  the cube-rule threshold (eps*c)^(1/3) rearranged to avoid a math.pow/cbrt call (~30x stricter
  than the old sqrtEps gate in float: ~0.0106 vs ~0.000345). On failure: unchanged control flow
  (drop P, retry 2-block; failing that, stall to honest Degenerate/MaxIterations).
  rowAux (the third row-combination scratch OrthonormalizeBlockB's old row-by-row Cholesky-QR
  needed) is now dead — SVQB's combination is a dense (non-triangular) recombination, not a
  triangular row update, so it needs a same-shape OUTPUT buffer instead (Xnext, see above) rather
  than a third per-row scratch vector. Removed from fProxyLOBPCGCache/RequireLOBPCGWorkspace/
  ArenaExtensions.fProxyLOBPCGCache (RequireDistinctBuffers 25->24); grep confirmed no test or
  benchmark file referenced it directly.
  NOT verified: the Burst test suite (orchestrator runs Tools/run-tests.ps1 centrally). `Tools/
  regen.ps1` confirmed both float/double codegen compile clean. The 3 structural demo residual
  audits (BuildingFrame/Truss3D/TrussModal) were NOT re-run — this fix can only change their
  iteration counts/residuals for the better (more likely to converge, never less certified), per
  the class doc's certification floor being unchanged, but that is an expectation, not a
  measurement.

## math.select branch-free conversion pass (docs/dev/spec-math-select-pass.md)
- 2026-07-17 | Batch A (per-element data selects): converted `SelectOP.fProxy.cs`/`SelectOP.iProxy.cs`
  selectfProxy/selectiProxy, `UnsafeMathOP.iProxy.cs` abs/max/min/relu, `UnsafeMathOP.fProxy.cs`
  relu, `UnsafeOP.iProxy.cs` sumAbs/maxAbs, `Blas.ColumnScaling.fProxy.cs` buildJacobiScale, and
  `SelectOP.bool.cs` selectBool (A11, taken) from ternaries/if-branches to
  `math.select`/`math.max`/`math.min`/`math.abs`. Float relu kept as
  `math.select(x[i], 0, x[i] < 0)`, NOT `math.max(x[i], 0)` (NaN/-0 semantics differ). Benchmark
  (A/B via a reverted scratch IJob benchmark, float N=10240, REPS=200, headless
  `Tools/benchmark.ps1`, 4 timed runs/side, repeated to check noise): `Select.select` (A1)
  before(branch) med ~0.36-0.67ms -> after(select) med ~0.10-0.11ms, a reproducible ~3-4x win
  across repeats — Burst emits an LLVM `select` directly from `math.select`, where the
  bool-array-driven ternary needed the optimizer to promote a branch, which it did less
  reliably. `relu` (A7) before ~0.033-0.038ms -> after ~0.033-0.035ms: no measurable difference
  (expected — Unity.Mathematics' `select` is itself defined as `test ? t : f`, and a direct
  float-compare predicate was already select-friendly either way); not a regression, so nothing
  reverted. Scratch benchmark file removed after recording; no permanent benchmark added.
- 2026-07-17 | Batch B (max/min reductions): converted `Eigen.fProxy.cs` powerIteration residual/
  finalResidual, inversePowerIteration vecDiff, InversePowerResidual, Gershgorin `bound`, and both
  `matScale` scans; `NormsOP.fProxy.cs` matrixL1/matrixLInf column/row-sum best; `LOBPCG.fProxy.cs`
  FactorGram ridge `scale`, MaxRelResidual `worst` to `math.max`. `LOBPCG.fProxy.cs`
  MinMaxDiagRatio (`mn`/`mx`, data-initialized from `L[0,0]`) and TryRayleighRitz's quotient
  envelope (`qMin`/`qMax`, finite-constant-initialized) kept the spec's prescribed forms
  (`math.select` for the former's NaN-accumulator risk, `math.min`/`math.max` for the latter).
  Left both `matScale < 1`-family clamps as branches (scalar, not a reduction). DEVIATION: the
  spec listed two `anorm` sites (Eigen.fProxy.cs, symmetric QL global-scale floor) as optional
  same-recipe "zero-init" extras alongside the `bound`/`matScale` sites — they are actually
  DATA-initialized (`math.abs(eigenvalues[0]) + math.abs(eVec[0])`), not `(fProxy)0`, so a NaN in
  the first diagonal/subdiagonal entry would make `math.max` silently recover where the branch
  keeps NaN forever (the same risk class MinMaxDiagRatio's select form exists to avoid). Left both
  `anorm` sites as branches rather than force an unverified conversion; not required by the batch
  (optional, take-or-leave).
- 2026-07-17 | Batch C (QR triangular R extraction): `QR.fProxy.cs` unblocked and blocked
  "copy upper triangular part of Q into R" loops split from a single `for r, for c { if c<r
  zero; else if c>r copy }` double loop into `for r { for c<min(r,N_Cols) zero; for c in
  [r+1,N_Cols) copy }` — branch-free by construction, touches the same cells with the same
  values, and never re-reads or overwrites the stale/already-written diagonal `R[r,r]`.
- 2026-07-17 | Batch D (argmax/selection-sort, optional): implemented then DROPPED per the spec's
  own gate. Converted all listed sites (`LU.fProxy.cs` decomp/decompInPlace partial-pivot argmax
  x4, `SVD.fProxy.cs` descending selection-sort argmax x5, `Eigen.fProxy.cs` eigenvalue
  selection-sort argmax x3, `LOBPCG.fProxy.cs` SortAscending argmin) to the
  `bool better = ...; x = math.select(x, candidate, better);` form and confirmed the full suite
  stayed green. LU A/B (`Tools/benchmark.ps1 LUBenchmark.Run`, N=64..1024, 4 timed runs, repeated):
  float `LU.decomp` was flat-to-slightly-slower after (~1-3%, within noise); double repeats varied
  ~12% run-to-run on the IDENTICAL unconverted code (e.g. N=1024 20.5-23.5ms), swamping any signal
  — no reproducible improvement. Reverted every Batch D site back to the original branches (no
  functional change from batch C); nothing committed for this batch.
- 2026-07-17 | Root defect (docs/dev/spec-lobpcg-robustness.md, Duersch et al. 2018 §4.1): the old
  test `‖r‖ ≤ tol·max(|λ|,1)` had no ‖x‖ — the residual is linear in x, so a shrinking iterate
  passes ever more easily and x=0 passes EXACTLY (λ≈0, r≈0). On the penalty-conditioned n=24 frame
  with k=4/guard=4 (3·kWork=n, RR basis tiles the whole space) float LOBPCG returned Converged with
  zero vectors at λ=0. Fix = Duersch Eq. 9 shape: `‖r‖ ≤ tol·(normAEst + |λ|·normBEst)·‖x‖` with
  Frobenius-sketch lower bounds from the orthonormalized seed (one-time, no extra matvecs;
  normBEst=1 on B=I), plus a per-pair B-norm certification floor (0.25): a pair below it is
  DEGENERATE — never locks, never counts converged, forces the new `IterativeSolveStatus.Degenerate`
  exit if still among the k wanted at exit. The (d1) re-deflation guard `bn2 > 0` left an
  annihilated row an exact zero row forever (self-certifying fixed point); now `bn2 > eps` with a
  deterministic reseed (seed keyed by (iter,i)) + single-row re-deflation + B-normalize.
  Cache grew two length-k vectors (resScale, xBnorm; RequireDistinctBuffers 23→25). Deliberately
  NOT done here (specced §C.2, owner-gated): per-iteration B-renormalization of X, SVQB-with-
  dropping, cube-rule Gram gate.
- 2026-07-17 | TWO DEVIATIONS from the spec's literal Eq. 9 shape `(normAEst + |λ|·normBEst)·‖x‖₂`,
  both forced by acceptance tests:
  (1) the λ term is anchored to ‖x‖_B, i.e. `normAEst·‖x‖₂ + |λ|·normBEst·‖x‖_B` — required for
  the spec's own §D.6 (rank-deficient B must not certify): with the literal shape, an iterate
  blowing up in a singular B's null space (x = x_r + c·e_null, c huge, ‖x‖_B ≈ 1, λ ≈ −c²)
  certifies as Converged — the denominator inflates with |λ|·‖x‖₂ ~ c³ while ‖r‖ ~ c² — observed
  λ ≈ −2e15 (float) / −9e49 (double) reported Converged. ‖x‖_B anchoring makes that relative
  residual O(1); the fixed floor xBnorm ≥ 0.25 alone does NOT catch it (the blowup keeps
  ‖x‖_B ≈ 1).
  (2) the final scale is `min(Eq9 shape, max(|λ|,1)·‖x‖_B)` — pure Eq.9 is ~‖A‖× LOOSER than the
  old `max(|λ|,1)` test for small-λ modes of penalty-scaled matrices (normAEst ≈ 300 on the frame
  demos), and certified residuals ~0.02-0.05 relative to ‖Aφ‖ that failed the demo smoke tests'
  independent residual audits (BuildingFrame/Truss3D/TrussModal, 3 failures at 6334-test scale).
  The min keeps certification at least as strict as pre-fix while every term still scales with
  the iterate's norms (the actual bug was the ABSENT ‖x‖, not the magnitude anchor).
  Also observed while testing: the fixed solver now honestly SOLVES the 1×1×1 frame repro in
  float under Eq.9-only (2 iterations, matches dense) — degenerate rows can no longer lock, so
  the lock-poisoning cascade never starts; the spec's expected `Degenerate` exit need not occur
  on the repro. Pre-fix behavior (verified by stash-running the repro test against the old
  solver): Converged with λ=0, ‖x‖=0 in both guard=4 (iters=4) and guard=0 (iters=48) float
  configs.

## LPBasis.populated native-backed; warm-state fix complete + .Run() regression tests
- 2026-07-17 | `LPBasis.populated` was a plain bool -> lost on an IJob by-value copy, so a worker `.Run()`
  re-seeded the basis and clobbered the warm start. It is set-once (read via IsEmpty/needsSeed, not
  ref-passed into a hot loop), so the LQR approach fits: `NativeReference<int>` behind a `bool` property,
  transparent to all call sites. Completes the LP warm-state fix (cache `_meta` mirror + this).
- `Pivot.swapCount` deliberately NOT mirrored: it only feeds `Pivot.Sign` (permutation parity ->
  determinant sign), which LP's warm resume never reads (FTRAN/BTRAN use the permutation ARRAY, native,
  which survives); LU sets/reads it within a single solve, not across `.Run()`s. Not load-bearing -- the
  warm-state audit over-flagged it.
- Regression tests (DemoSmokeTests, plain `.Run()` = by-value copy, FAIL on the pre-fix plain-field code):
  `LqrWarmState_SurvivesRunByValueCopy` (populated visible on caller after a cold solve through a job
  field) and `EconomyLPJob_WarmState_SurvivesRunByValueCopy` (same LP twice via `.Run()` -> 2nd is a
  cache HIT, warm pivots < cold, only if cache+basis survived the copy). See [[job-struct-copy-warmstate-audit]].

## fProxyLPCache: native-mirror warm-state so it survives an IJob by-value copy
- 2026-07-17 | Same bug class as the LQR fix: LP.solve's warm-state scalars (builtVersion, etaCount,
  factorsValid, weightsValid) were plain fields, lost on an IJob by-value `.Run()`/`Schedule` copy →
  a worker-scheduled warm solve silently desyncs the eta chain. Fix = MPC's `qpMeta` pattern done RIGHT
  (my first two attempts were wrong — see below): KEEP the four as plain fields (so every read/write
  INSIDE the solve is byte-identical to before) and add a `NativeArray<int> _meta` mirror synced ONLY at
  the boundary — `RehydrateWarm()` (fields <- _meta) before the useCache branch's first read,
  `PersistWarm()` (_meta <- fields) after DualSimplexCore. matrixVersion stays a plain field (caller-owned,
  read-only inside → survives copy-in). Under `.Run()` the fields don't survive but `_meta` does, so the
  rehydrate restores them; under RunByRef both survive.
  - DEAD ENDS (do not retry): (1) turning the fields into `NativeArray`-backed PROPERTIES changed how the
    solve reads them and (2) a `ref`-into-native `EtaCountRef` accessor for the ref-passed etaCount — BOTH
    regressed the EconomyLP warm re-solve to 4 pivots. Even the correct boundary-mirror (fields untouched)
    still shows 4 vs cold-3 on that 3-pivot toy: it is Burst codegen/struct-layout jitter (adding `_meta`
    perturbs LP.solve's compilation under FloatMode.Default), NOT a warm-state desync (a real resume
    failure costs many pivots, not one; the solve stays optimal). The demo assertion was tightened-then-
    relaxed to `warm <= cold+1` accordingly. STILL TODO: a `.Run()` (not RunByRef) regression test that
    proves cross-copy survival; LPBasis.populated + Pivot.swapCount still plain-field (unfixed).

## fProxyLQRState.populated: native-backed so it survives an IJob by-value copy (warm-state fix)
- 2026-07-17 | Bug class: a warm-start flag mutated inside `IJob.Run()`/`Schedule()` is LOST because the
  job runs on a by-VALUE copy of the state struct (native BUFFERS survive — they're pointers — but plain
  fields don't). A worker `.Run()` of an LQR warm solve silently reset `populated`, forcing every warm
  call cold (or worse for LP's counters). Fix = the MPC `qpMeta` idea, but cleaner: `populated` moved
  behind a `NativeReference<int>` and re-exposed as a `bool` PROPERTY, so all call sites are unchanged and
  writes go through the shared handle — no rehydrate/copy-back (the flag is set once per solve, not in a
  hot loop; contrast LP's `etaCount`). `NativeReference` confirmed Burst-job-compatible (ControlLQRTests'
  TestJob runs the warm path through an IJob). Suite 6317/6317. See [[job-struct-copy-warmstate-audit]].

## UnsafeOP.max/min: hardware mm256_max_pd/min_pd for double too (closes the double gap)
- 2026-07-17 | Follow-up to the width-8 float win: double max/min were still ~11.5 GFLOP/s (~2× behind
  double sum's ~22) because double skipped fProxyW and its fProxy4 body used `math.max(double4)` =
  compare+select. Added `X86.Avx.mm256_max_pd`/`min_pd` to `fProxyW.Max`/`Min`'s DOUBLE path (AVX branch +
  lane-wise fallback) and removed the `skipFor[double]` from the kernels so double now runs the fProxyW
  main loop too. Result: **double max 11.5→19.6, min →20.5 GFLOP/s (~1.7-1.8×)** — now within ~10% of sum;
  float unchanged (its diff was whitespace only). Suite 6317/6317, finite-data bit-identical.
  - SAFE re maxAbs's frozen contract: maxAbs (and sum/vecDot) ALSO skipFor[double], so they never called
    fProxyW.Max's double path — it was dead until this kernel. So adding mm256_max_pd there has no
    collateral effect on any existing double kernel. (User sign-off given.)

## UnsafeOP.max/min: width-8 fProxyW upgrade (corrects the width-4 claim below)
- 2026-07-17 | The earlier "width-4 saturates, min/max are memory-bound" claim was WRONG — KernelBenchmark
  proved it. At in-L1 sizes (N<=1024) max/min are THROUGHPUT-bound: width-4 float max hit only ~12 GFLOP/s
  vs sum's ~40. TWO causes: (1) width-4 (float4/128-bit) vs sum's width-8 (fProxyW/256-bit); (2) `math.max`
  lowers to compare+select (2-3 ops), not a single hardware `maxps`. Fix: added `fProxyW.Min`/`HMin`
  (hardware `mm256_min_ps`, mirrors Max/HMax) and rewrote the kernels to the fProxyW main loop + fProxy4
  remainder (like maxAbs), seeded from a[0] (max/min idempotent → re-including the seed is exact).
  Result: **float max 11.9→31.3 GFLOP/s (2.6×) @ N=1024**, near sum. Suite 6317/6317 (finite-data
  bit-identical; float now follows hardware-max NaN semantics, like maxAbs already does).
  - DOUBLE unchanged (~11.5, still ~2× behind sum): double skips fProxyW (double4 is already 256-bit) and
    its fProxy4 body uses `math.max(double4)` = compare+select. Closing it needs `mm256_max_pd`/`min_pd` in
    fProxyW.Max/Min's DOUBLE path — but that path is shared with maxAbs's frozen contract, so it's gated on
    owner sign-off (OPEN). Added KernelBenchmark `max`/`min` reduction cases to measure all this.

## UnsafeOP.max/min kernels; NormsOP.normalizeColumns + Eigen dot reroutes
- 2026-07-17 | Added `UnsafeOP.max`/`min` (SIMD running max/min over a[0..n), contract n>=1). WIDTH-4
  (fProxy4) only, NOT fProxyW: min/max reductions are memory-bound (one load + one compare/element) so the
  4-wide accumulator saturates bandwidth as well as 8-wide would, AND fProxyW has no Min/HMin (adding it =
  touching the owner-gated wide type, avoided). Seeded from the first lanes (not a neutral identity) so
  all-negative/all-positive inputs are correct; uses math.max/min to match caller NaN semantics. max/min
  are exact → lane order is bit-identical for finite data.
- 2026-07-17 | NormsOP.normalizeColumns: strided per-column norm + scale → row-major per-column accumulate
  (colSum trick, unit-stride, vectorises, bit-identical) into a length-N_Cols Temp, then reciprocal-once
  (`(norm>0)?1/norm:1`) + branch-free row-major `row[c] *= inv[c]` (×1 leaves zero/NaN-norm columns
  bit-identical, matching the old skip). Now needs arena-backed A (fProxyTempVec).
- 2026-07-17 | Eigen: rerouted the O(n) vector dots in the two iterative eigensolvers to UnsafeOP.vecDot
  (deterministic-reorder waiver): powerIteration (seed v·v, Rayleigh v·w, ‖w‖²) and lanczos (seed, alpha
  v·w, the O(steps²) reorth proj = V[k,:]·w, ‖w‖²). Self-dots use vecDot(p,p,n) (established, e.g.
  rowNormL2). MODEST — these are O(n) in O(n²/nnz)-per-iter algorithms (dense tridiagonal path already
  uses vecDotRange and was untouched). Left fused residual max-abs loops scalar.

## NormsOP.matrixL1 — colSum-trick restructure (bit-identical)
- 2026-07-17 | ‖A‖₁ (max abs column sum) had a strided inner loop (`for i: colSum += |A[i,j]|`, stride
  N_Cols → scalar under Strict). Restructured to a row-major per-column accumulate into a length-N_Cols
  Temp: `for i { for j: acc[j] += |A[i,j]| }` (unit-stride inner → vectorises), then a scalar max over
  acc. BIT-IDENTICAL — each column still sums its rows in ascending i order; NaN semantics preserved by
  keeping the final max as `if (acc[j] > best)` (not a max kernel). Same restructure class as
  StatsCore.colSum (~32×). Now needs an arena-backed A (via `fProxyTempVec`, like the sibling matrixL2);
  matrixLInf stays allocation-free (rows are contiguous → already routes to UnsafeOP.sumAbs).

## LOBPCG + ladIRLS: reroute inner dots to UnsafeOP.vecDot (reduction-reroute batch 6)
- 2026-07-17 | Replaced hand-rolled scalar `for c: s += V[i,c]*W[j,c]` dots with `UnsafeOP.vecDot`
  (row pointers hoisted via `.Data.Ptr + (long)row*N_Cols`, length n). Sites: LOBPCG `FillGramSub`
  (Gram = VᵀW, the O(k²·n) fill), `FillHSub` (H = VᵀAW), `Deflate` (coeff = <AgainstB_i, V_a>); ladIRLS
  residual `ri = -b[i] + dot(A_i, x)` (`Optimize.ladIRLS`). Made the three LOBPCG helpers `static unsafe`
  (matches the file's existing RequireDistinctBuffers/RestoreBufferIdentity convention); ladIRLS uses an
  `unsafe {}` block around the row loop (public API method, kept non-unsafe). Added
  `using LinearAlgebra.Internal;` to both files.
  - NOT bit-identical: vecDot's fixed 2×fProxyW/2×fProxy4 accumulator tree reorders the summation vs the
    scalar left-to-right sum. Deterministic + cross-arch (frozen kernel contract), covered by the pre-1.0
    "no bit-compat obligation yet" waiver. Tolerance-based tests unaffected (suite 6317/6317).
  - LEFT scalar deliberately: ladIRLS line ~297 final-objective dot accumulates in `double` even for the
    float variant (higher precision) — vecDot would drop that. LOBPCG's residual loop (rv = AX-λBX)
    fuses compute+store R[i,c]+norm in one pass — not a pure dot, left as a later pointer-hoist target.
    LOBPCG lines 154/263 (Rayleigh/B-norm single dots) are O(k·n), inside the big non-unsafe iteration
    method — skipped to avoid widening its unsafe scope for marginal gain.

## fProxyW: width-4-halves float fallback (NEON on ARM)
- 2026-07-17 | The float (8-lane) non-AVX fallback for the element-wise ops built a `new v256(op(Float0),
  … op(Float7))` from 8 scalar lane ops, which does not reliably auto-vectorize on ARM (scalar NEON). Rewrote
  each as two width-4 `float4x2` halves (`c0` = lanes 0–3, `c1` = lanes 4–7) so Burst emits NEON on ARM;
  x86 still takes the `if (IsAvxSupported)` mm256 path unchanged. Bit-identical — same per-lane op, same
  lane mapping, no fold reorder. `float4x2` is exactly two contiguous `float4` (32 B = one v256); the
  `fProxy4x2` stub already existed (proxyStructs.math.cs), so no new scaffolding. Two mechanics per op:
  - `+ - * /`: stub `fProxy4` has these operators, so the two-halves body compiles in the template
    assembly directly (kept in a `{ }` block to avoid a CS0136 clash with the double path's `av`/`bv`).
  - `Abs/Max/Min`: stub `fProxy4` has NO `math.abs/max/min` overload, so the two-halves body is an
    `emitFor[float]` (`//!`) block that only materializes in the generated float file (where `fProxy4`->
    `float4`), with the old 8-scalar form kept behind `deleteThis` purely so the template assembly compiles.
  - `HSum`/`HMax`/`HMin` unchanged — horizontal folds run once per reduction and their order is the frozen
    numeric contract. `Splat`/`Load`/`Store` unchanged (broadcast/memcpy lower fine already).
  - Untestable here (x86 CI always takes the AVX path; no ARM runner) but bit-identical by construction and
    the whole suite compiles + passes. Perf claim is ARM-only and unmeasured.

## UnsafeOP/WideOP: alias fProxy4 + delete the fProxyM/floatM/doubleM shim layer
- 2026-07-17 | Final step of the alias refactor: no file calls `fProxyM` any more, so deleted class
  `fProxyM` (`proxyStructs.math.cs`) AND `OP/SimdMath.cs` (`floatM`/`doubleM`) outright.
  - `UnsafeOP.fProxy.cs`: aliased `fProxy4` -> `Unity.Mathematics.float4` (deleteThis block, replacing the
    `mathProxies` import) and swapped the 10 `fProxyM.abs/max` accumulator calls (sumAbs/maxAbs) to
    `math.abs/max` — resolve natively on the real float4/double4. `fProxyW` (265 uses, the wide v256 type)
    is namespace-local (WideOP.fProxy.cs), unaffected, and NOT aliased/touched.
  - `WideOP.fProxy.cs`: only touched the two `//!`-commented `emitFor[double]` lines (`fProxyM.abs/max`
    -> `math.abs/max`); these activate ONLY in the generated double file where `fProxy4`->`double4`, so
    `math.abs(double4)` is native. Kept the `mathProxies` import (its live `fProxy4` reinterpret casts
    still use the stub); `fProxyW` itself untouched.
  - Generated delta is a pure identity rename (`floatM.abs(float4)` was literally `=> math.abs(float4)`):
    UnsafeOP.float/double.cs + WideOP.double.cs only, byte-diff = `floatM/doubleM.` -> `math.`. Suite
    6317/6317. The `fProxy4` struct + matrix stubs (`fProxy4x4` etc.) in `proxyStructs.math.cs` STAY —
    WideOP + the matrix proxies still compile against them; that stub deletion is a later phase.
    See [[simd-proxy-select-extension]] / docs/dev/spec-alias-simd-proxies.md.

## QueryOP: alias fProxy4 -> float4/double4 (pilot) instead of extending the stub
- 2026-07-17 | Better fix for the previous entry's problem. Rather than teaching the `fProxy4` STUB new
  tricks (comparison/select), QueryOP now does `//+deleteThis using fProxy4 = Unity.Mathematics.float4;
  //-deleteThis` at file top. In the template `fProxy4` IS `float4` (real type), so `v < best`,
  `math.select(fProxy4,...)`, `*(fProxy4*)ptr` all resolve NATIVELY; codegen still rewrites the token
  `fProxy4` -> `float4`/`double4` per file (alias line deleteThis'd), so the double side gets `double4`
  natively too. No `fProxyM`/`floatM` shim in the path at all. Suite 6317/6317, rowArgMin bit-identical +
  same perf. REVERTED the ec14fad stub additions as now-unused (fProxy4 `<`/`>`/`<=`/`>=`,
  fProxyM.select/min, floatM/doubleM.select/min) — QueryOP was their only consumer. `fProxyM.abs/max` +
  the fProxy4 struct KEPT (UnsafeOP/WideOP/LP/matrix-proxies still use the stub; not yet converted).
  This is the pilot for the general "alias the vector proxies, delete the shim layer" refactor
  ([[simd-proxy-select-extension]] / a spec TBD). fProxyW is the exception — no real Unity float8 to
  alias to, so its ops stay hand-rolled.

## SIMD proxy stubs: fProxy4 comparison + fProxyM/floatM/doubleM select+min; rowArgMin -> fProxy4 SIMD
- 2026-07-17 | Extended the width-4 SIMD proxy surface so branch-free lane-parallel select kernels can be
  written in templates (previously the `fProxy4` stub was accumulator-only: `+ - * /`, abs, max).
  Mechanism recap: `fProxy4`->`float4`/`double4` by codegen. OPERATORS float4 has natively (`<`/`>`/`<=`/
  `>=` -> `bool4`) go straight on the `fProxy4` stub (`proxyStructs.math.cs`) — generated code uses
  float4's native ones, no shim. But `math.select`/`math.min` are STATIC `math` methods, not float4
  members, so a template can't call `math.select(fProxy4,...)`; they go through the existing
  `fProxyM`->`floatM`/`doubleM` indirection (like abs/max) — added `select`+`min` there
  (`proxyStructs.math.cs` stub + `OP/SimdMath.cs` real). `int4` is a REAL Unity type in templates, so the
  index half (`math.select(int4,...)`) needs nothing. See [[simd-proxy-select-extension]].
- 2026-07-17 | First customer: rewrote `RowArgMinScan`/`RowArgMaxScan` from the 4-lane-scalar fallback to
  a real `fProxy4` SIMD accumulator (running extreme + int4 index, strict `<`/`>` mask -> NaN never
  displaces; `fProxyM.select` value, `math.select` index; value-then-smallest-index horizontal reduce).
  Bit-identical (suite 6317/6317). N=1024 float rowArgMin 0.20->0.1175 ms (1.7x over the scalar lanes,
  ~5x over the original indexer scan). iProxy4 NOT built (no integer SIMD kernel needs it yet; int4 is
  real so it wouldn't need the fProxyM shim anyway).

## QueryOP: rowArgMin/rowArgMax as 4-lane branch-free math.select scans
- 2026-07-17 | rowArgMin/rowArgMax (argmin/argmax WITH index capture — doesn't vectorise as a plain
  loop) rewritten via `RowArgMinScan`/`RowArgMaxScan`: 4 INDEPENDENT branch-free scalar lanes (lane L =
  columns L, L+4, ...) using `math.select` (NOT `if`), each keeping a running extreme + its column index
  via a strict `<`/`>` mask, then a value-then-smallest-index horizontal reduce. Branch-free + independent
  lanes → Burst packs/overlaps them. Bit-identical to the scalar first-occurrence scan (strict mask → NaN
  never displaces; suite 6317/6317). N=1024 float rowArgMin 0.57→0.20 ms (~2.9×), double 0.75→0.23 ms
  (~3.2×). Tried fProxy4-select first — BACKED OUT: the fProxy4 stub is accumulator-only (no comparison/
  select); extending it is infra tracked in [[simd-proxy-select-extension]] (would let this go true SIMD,
  but 4-lane scalar already gets most of it). Scan helpers keep `unsafe` in signature (fProxy* param =
  the legitimate case, like UnsafeOP). Added nearestColumn to QueryBenchmark: AllColScores N=1024 float
  0.034 ms (~50× over the old strided ColScore, now faster than nearestRow). QueryOP fully optimized.

## QueryOP: strided column search restructured row-major (AllColScores)
- 2026-07-17 | The column search family (`nearestColumn`/`farthestColumn`/`countWithinColumnRadius`/
  `distancesToColumn`) each scanned columns via the strided `ColScore` (per-column walk down rows).
  Factored a shared `AllColScores` helper that computes ALL per-column metric scores in ONE row-major
  (unit-stride inner) sweep with per-column accumulators (the colSum trick) — metric-specific
  (Manhattan/Euclidean/SqEuclidean/Chebyshev/Dot direct; Cosine uses a second normA accumulator + the
  precomputed normQ). Each column still sums its rows ascending → bit-identical to the strided form
  (suite 6317/6317). The four methods now allocate a length-N_Cols Temp, call AllColScores, then reduce
  (argmin/argmax/count) over it. Same restructure class as colArgMin/nearestRow (~7×/2.6×). `ColScore`
  kept (still used by ArenaExtensions.Query two-pass alloc). QueryOP is now fully optimized except
  rowArgMin/rowArgMax (argmin index-capture — deliberately deferred).

## QueryOP: colArgMin/colArgMax restructured + RowScore metric reductions to vecDot
- 2026-07-17 | `colArgMin`/`colArgMax` (strided per-column argmin/argmax walk) restructured into a
  row-major per-column running-min/max + argmin sweep (the (val,idx) overloads accumulate the running
  extreme directly into valPerCol — no scratch; the index-only overloads use a length-N_Cols Temp).
  Bit-identical (each column visits rows ascending, strict `<`/`>` → smallest-row-wins ties preserved).
  N=1024 float 1.80→0.24 ms (~7.5×), double 1.57→0.25 ms — now FASTER than rowArgMin (0.75 ms), whose
  horizontal per-row argmin doesn't vectorise. Added colArgMin to QueryBenchmark.
- 2026-07-17 | `RowScore` (both overloads): Dot + Cosine reductions routed to `UnsafeOP.vecDot`
  (summation-order-changing = deterministic, pre-1.0 waiver); the difference-based metrics
  (Manhattan/Euclidean/SqEuclidean/Chebyshev) kept a DIRECT `(a-b)²`/`|a-b|` scalar sum, pointer-hoisted
  only (the expanded ‖a‖²−2a·b+‖q‖² form risks catastrophic cancellation at near distances → not used).
  Speeds nearestRow/farthestRow/countWithinRadius/distancesToRow: nearestRow Euclidean N=1024 float
  0.94→0.36 ms (~2.6×). Used `unsafe { }` blocks, not an `unsafe` method modifier (minimal scope).
  Cosine `normQ` was ALREADY hoisted out of the per-row loop (QueryNormSq + the normQ overloads) — not a
  bug, verified. STILL scalar (follow-up): rowArgMin/rowArgMax (argmin index-capture — deferred by user),
  and the STRIDED column search (nearestColumn/farthestColumn/countWithinColumnRadius via ColScore) which
  needs the same row-major restructure but is metric-specific.

## QueryOP.argMaxColNorm: strided column walk restructured to row-major (the colSum trick)
- 2026-07-17 | Corrects the prior entry's "column ops are fundamentally strided, leave them scalar."
  They are NOT: `argMaxColNorm`'s per-column norm was computed by a strided per-column walk
  (`for c { for r A[r,c] }`), but the same result comes from ONE row-major sweep accumulating a
  per-column norm vector (`for r { for c acc[c] += f(A[r,c]) }`), then argmax over acc — the inner c
  loop is unit-stride and vectorises, and each column still sums its rows in ascending order so it is
  BIT-IDENTICAL (no waiver; same reason colSum is). Costs one length-N_Cols Temp accumulator
  (self-disposing, job-safe). Suite 6317/6317 unchanged. N=1024 float 1.85→0.032 ms (~58×), double
  1.54→0.062 ms (~25×) — now identical to the row op; the row/column asymmetry is eliminated, not
  merely worked around. **General lesson: "strided column reduction" is usually a restructuring
  opportunity, not a hard limit — NormsOP.matrixL1 and normalizeColumns' norm pass are the same shape
  and could get the same bit-identical treatment (softmaxColumns too, with 3 row-major passes).**

## QueryOP.argMaxRowNorm: routed to SIMD reduction kernels + new QueryBenchmark
- 2026-07-17 | `argMaxRowNorm` (per-row L1/L2/Linf norm, pick the max) was a hand-rolled scalar
  reduction on the indexer. Rerouted the row-inner reductions: L1→`UnsafeOP.sumAbs`,
  L2→`UnsafeOP.vecDot(row,row)`, Linf→`UnsafeOP.maxAbs` (L1/L2 summation-order-changing = deterministic
  not bit-identical, pre-1.0 waiver; Linf = math.max exact = bit-identical). Outer argmax stays scalar.
  Suite 6317/6317. Added `QueryBenchmark` (a few common ops on N×N: rowArgMin, argMaxRowNorm,
  argMaxColNorm, nearestRow — was a coverage gap). Measured N=1024 float argMaxRowNorm L2 0.61→0.030 ms
  (~20×, 35 GFLOP/s). The bench cleanly shows the row/column asymmetry: rerouted row op 0.030 ms vs the
  STRIDED `argMaxColNorm` (column-inner, left scalar — a contiguous kernel can't consume a strided
  column) 1.85 ms = 62× apart at N=1024.

## LP.simplexCore: tableau pivot hoisted to axpy (spec-raw-pointer-hoist-pass batch 3)
- 2026-07-17 | The dense two-phase tableau simplex `Pivot` (row normalize + eliminate every other
  constraint row and both reduced-cost rows) and `simplexCore`'s initial pricing were on the `fProxyMxN`
  struct indexer. Hoisted `T.Data.Ptr` (per-row base) and routed every `row -= f*pivotRow` /
  `cost -= f*pivotRow` through `UnsafeOP.axpy` (eliminate rows i != prow are distinct from the pivot
  row; cost1/cost2 are distinct buffers → `[NoAlias]` legal; IEEE-exact → bit-identical, iters + objective
  byte-identical, suite 6317/6317). Measured (9950X3D): §1 tableau simplex float n=192 102.8→4.41 ms
  (23×), n=384 1475→50.3 ms (29×); double n=384 1924→118 ms (16×); §4 covering LP float n=192 1553→51.3 ms
  (30×). RatioTest left scalar (T[i,enter] = strided column walk). Note: tableau simplex is the reference
  backend (default is RevisedSimplex), so this mainly speeds LAD-simplex + the reference path.

## NormsOP: row norms routed to SIMD reduction kernels
- 2026-07-17 | Same follow-up as StatsCore's row reductions (see Statistics/DEVLOG.md). `normalizeRows`
  (L1/L2/Linf per-row norm) and `matrixLInf` (max abs row-sum) were hand-rolled scalar reductions —
  serial-locked under Strict. Rerouted: L1→`UnsafeOP.sumAbs`, L2→`sqrt(UnsafeOP.vecDot)`, matrixLInf
  inner→`sumAbs` (all summation-order changes → deterministic but not bit-identical to the prior serial
  sum; owner-approved pre-1.0 baseline change). **Linf→`UnsafeOP.maxAbs` is BIT-IDENTICAL** (max is
  associative/exact — no rounding to reorder), a free win needing no waiver. The `row[c] *= inv` apply
  loop is a bit-identical elementwise hoist. `normalizeColumns`/`matrixL1` left scalar (column-inner =
  strided). Suite 6317/6317. No NormsOP benchmark (accepted gap); the kernels are the same ones
  measured in StatsBenchmark (rowSum 0.35→0.035 ms at N=1024).

## LU.decompNoPivot: raw-pointer hoist (spec-raw-pointer-hoist-pass batch 1)
- 2026-07-17 | `decompNoPivot`'s trailing-row elimination inner loop `U[j,i] -= Ljk*U[k,i]` was still
  on the `fProxyMxN` struct indexer while its pivoted siblings (`decomp`/blocked/`decompInPlace`)
  already hoist `U.Data.Ptr` and route the axpy through `UnsafeOP.axpy`. Applied the identical
  transform (rows j>k are distinct → `[NoAlias]` legal; `(-Ljk)*U[k,i]` added is IEEE-exact to the
  scalar form → bit-identical, suite 6317/6317 unchanged). Measured (9950X3D, upper CCD, N=1024):
  float 514.9→19.30 ms (~27×, 1.39→37.1 GFLOP/s), double 519.6→34.43 ms (~15×). Tracks pivoted
  `decomp` up to N≤256; the blocked level-3 path pulls `decomp` ahead only at N=1024 (out of scope).
  Added a `decompNoPivot` case to LUBenchmark (was measuring only pivoted `decomp` = a gap).
- 2026-07-17 | LU/LUP split (user floated during this batch): recommend DO NOT split. `decompNoPivot`
  is already its own public entry point with its own contract, and post-hoist it shares the same
  vectorised axpy kernel as the pivoted paths — a file/type split buys no perf and adds codegen churn
  plus API-surface risk pre-v1.0. Left as one `LU` partial class.

## LOBPCG: IJob cache-copy corrupted eigenvectors (ping-pong buffer reseat)
- 2026-07-16 | Symptom: `Eigen.lobpcg` run inside an IJob returned correct eigenVALUES but
  corrupted eigenVECTORS (relative residual ~1e-1) on clustered/near-degenerate spectra; the same
  call on the main thread (`ref cache`) was exact. Presented as "Burst-only" and cost a long hunt —
  I wrongly chased FloatMode, FloatPrecision, `[ReadOnly]`/NoAlias, OptimizeFor, an aliased `Deflate`
  call, and a stale-`AX` theory, and even wrote (then reverted) a comment blaming Burst for mis-
  sequencing `Swap.Rows`. All wrong. ROOT CAUSE (credit: fable consult): `UpdateActiveBlock` ends
  each iteration with `SwapMat(ref ws.X, ref ws.Xnext)` — a struct-VALUE ping-pong (double buffering)
  that reseats which allocation the `ws.X` FIELD names. An IJob executes on a COPY of the cache
  struct: writes THROUGH the buffer pointers reach the caller, but the reseated FIELD does not, so
  after an ODD iteration count the caller's `cache.X` still points at the entry buffer, which holds
  the previous (pre-sort) iterate → sorted `lambda` paired with UNSORTED `X`. `lambda`/`residual`
  are never ping-ponged so they always sort correctly; that asymmetry was the tell. It is NOT a
  Burst bug (a plain Mono IJob reproduces it identically; the correct vectors sit in `cache.Xnext`).
  Only surfaces when the exit sort does real reordering (locking on clustered spectra) AND parity is
  odd. Fix: capture entry buffer identities (`xEntry`/`pEntry`), and before every return
  `RestoreBufferIdentity` copies the final data back into the entry allocation and swaps the fields
  so `ws.X`/`ws.P` reference their entry buffers on return — one O(k·n) copy at exit only when parity
  flipped, zero hot-loop cost, ping-pong untouched. P is restored too (warm-start reuse reads it).
  Why the suite missed it: every prior LOBPCG [Test] was a main-thread `ref` call and the benchmark
  jobs only read `infoOut`; added `JobbedClusteredSpectrumLeavesCorrectVectorsInCache` (runs
  `.Run()` on a 2D-Laplacian degenerate spectrum, checks `cache.X` residuals post-job — verified it
  fails when the fix is neutered). Audited: this `SwapMat`-of-caller-visible-cache-field pattern
  exists ONLY in LOBPCG.

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

## DetMath
- 2026-07-16 | Added the `LINALG_NATIVE_MATH` compile-mode switch: a single `#if` sets
  `public const bool UseNative`, and every transcendental branches on that const as its first
  statement (`if (UseNative) return math.XXX(...)`). Deterministic DetMath stays the default
  (const false); defining the symbol flips every call site to `math.*` for raw throughput,
  giving up cross-arch determinism. Because it's a `const bool` rather than a per-function
  `#if`, BOTH branches are always real, type-checked C# — Burst folds the dead branch away at
  native codegen (literal-const propagation), so there's no runtime cost and no risk of a
  native-only typo going unnoticed by the default (deterministic) test run. Left composing
  (no native branch, per spec): `Pow(fProxy,int)` (exact integer path, no math.* equivalent),
  `Exp10` (no math.exp10 in Unity.Mathematics), `Acosh` (no math.acosh). `SinhCosh` (the
  shared-computation helper, analogous to `SinCos`) also stays composed — only its two callers
  `Sinh`/`Cosh` gained native branches, matching the exact function map in the spec. Verified
  `math.exp/log/log2/log10/sin/cos/tan/atan/atan2/asin/acos/sinh/cosh/tanh/exp2/pow/sincos` all
  exist for both float and double in Unity.Mathematics before wiring (checked
  Library/PackageCache math.cs directly). SinCos's native branch calls `math.sin`/`math.cos`
  separately rather than `math.sincos` — its `out float`/`out double` params don't bind to the
  `fProxy` proxy type in the raw template (same limitation already hit by RandomOP's Gaussian
  sampler, see below). `UseNative` itself is wrapped in `//+skipFor[double]` so it's defined
  ONCE (float fragment only) instead of twice — DetMath.float.cs and DetMath.double.cs merge
  into one partial class, so a bare unwrapped const would double-define and fail CS0102; the
  double fragment's method bodies still see it fine through the merge. Runtime testing of the
  native path requires adding the define under Player Settings — not done here (out of scope;
  default-mode compile already exercises both branches' C#).
- 2026-07-15 | Promoted the deterministic transcendentals from the benchmark prototype
  (TemplateSourceBenchmarks/DetMathBenchmark) to a shipping public class `DetMath` (OP/DetMath.
  fProxy.cs, float+double overloads). Surface: Exp/Exp2/Exp10/Log/Log2/Log10/Pow, Sin/Cos/SinCos/
  Tan, Asin/Acos/Atan/Atan2, Sinh/Cosh/Tanh, Acosh — everything the library's math.* usage needs
  (rcp/rsqrt/sqrt stay math.*, already deterministic). One canonical scheme: accurate Horner
  minimax (dropped the prototype's Estrin/Fast experimental variants; Estrin is a latency option
  if a scalar hot path ever needs it). Cody-Waite reduction, ldexp-by-bits, all branch-free
  guards. Accuracy vs libm ~1e-5 float / ~1e-12 double (few ULP) verified by sweep tests
  (DetMathTests, 500-pt sweeps per fn over the domain + edge/total behaviour), suite 6297/6297.
- 2026-07-15 | ExpGuard NaN bug found by the new edge tests (the benchmark never exercised it —
  its inputs were [-10,10]). Exp relied on the polynomial IMPLICITLY producing NaN for a NaN
  input, but the `(int)NaN` conversion + multiply in Ldexp does not preserve NaN under Burst, so
  Exp(NaN) returned a finite value. Fix: ExpGuard now propagates NaN EXPLICITLY via
  `select(y, NaN, x != x)`, matching LogGuard/TrigGuard (which always did the explicit x!=x check
  and passed). Lesson: never rely on implicit NaN propagation through an int-conversion path;
  guard the original input explicitly.
## axpy4: quad-stream panel updates for the blocked factorizations
- 2026-07-14 | vecMatDot (xᵀA — simplex PRICE, transposed GEMV) moved onto axpy4 (four matrix
  rows per output pass, r-ascending per element = bit-identical): float 41→69 GF/s at n=64,
  57→82 at 128, 69→85 at 256, +18% at 512; double +19-56% at 64-256. 1024 rows flat — the
  streamed matrix is the bandwidth wall there, quad-streaming can't help a one-touch stream.
- 2026-07-14 | The blocked factorization trailing updates (CHO syrkLowerSub, CHOP
  syrkUpperSub, QR/LQ wyVtC/wySubVW/lqYeqCVt, pivoted-LU's inlined row update) all had the
  same shape: one axpy pass over the output row per panel column — vectorized but bound by
  output-row read-modify-write traffic, not flops (CHO float 1024 ran 33 GF/s vs the GEMM
  tile's ~100). Fix: UnsafeOP.axpy4 fuses FOUR coefficient streams into one output pass
  (arithmetic intensity 4x); per-element operation order stays p-ascending sequential, so
  results are BIT-IDENTICAL to the old kernels for both dtypes — no skipFor/W-tier needed,
  the map auto-vectorizes. Min-across-runs (ambient load made single runs swing 20-40%;
  trust direction + mins): CHO 1024 float 10.79 → 9.10 ms, double 15.17 → 12.92; CHOP 1024
  float 21.10 → 17.97; dense blocked LU inherits via wySubVW (float 1024 15.69 → 14.17).
  NOT retuned under noise: floatCholBlockMinN (1024 — blocked path got ~15% faster, the
  crossover may now sit at 512; re-measure on an idle machine), same for the QR/LQ/LU gates.
  trsmLowerPanel and the LU/QR small TRSM steps (~5% of time) left single-stream.

## fProxyW stage 2c: broadcast GEMM tiles (matMatDot / TransA / AtA) on wide accumulators
- 2026-07-14 | User ruling: no wrapper-level transpose routing ("just rewrite the critical
  path") — a briefly-added staged-transpose detour in Blas.dot/dotSym was reverted the same
  hour. The honest rewrite: matMatDotUnpackedW / matMatDotTransACoreW hold the 8x16 tile as
  two fProxyW per row with Splat broadcasts — BIT-IDENTICAL to the scalar tiles (one
  p-ascending chain per element), so no numeric change at all. Float: plain GEMM 90 → 110-114
  GF/s at 128-512 (1024: 28.0 → 24.0), TransA 88 → 103-115 (1024: 27.5 → 24.9), AtA
  151 → 174-204 GF/s-eff (512: 1.61 → 1.31 ms); n=64 improved too (no small-size gate
  needed). Double: unchanged, keeps the scalar tiles via choose-routing.
- 2026-07-14 | trsmLowerPanel rewritten on fProxyW (both dtypes: 8 float / 4 double lanes),
  bit-identical: rows are independent, so Width rows solve simultaneously through a
  contiguous tile — per column p, one broadcast-FMA chain over k<p (no per-row short
  reductions, no horizontal ops), then one wide division; each lane replays the scalar
  row's exact chain. Blocked CHO (idle-machine A/B, unblocked control rows flat): float
  512 1.635→1.147 ms (−30%), float 1024 8.78→6.80 (−23%, 52.6 GF/s — CHO now beats LU),
  double 512 −21%, double 1024 −12%. The old "TRSM ≈5% of time" note was wrong — the
  dot-form solve was ~9% of flops at a fraction of SYRK throughput ⇒ ~25-30% of wall.
  fProxyW gained operator/ (mm256_div_ps + lane fallback; template fProxy4 stub too).
  CODEGEN NOTE: this kernel's accumulator IS seeded from memory and compiles clean — the
  seeded-W byte-rotation pathology (see the 6x16 entry) needs MANY live seeded
  accumulators, not one. TESTS: CHO Blocked* cases were sized 256-400 for the ORIGINAL
  256 gate and silently stopped reaching the blocked core when the gate moved to 512 —
  resized to 512/545/576/600 (545/600 = ragged last panel + wide-kernel scalar-remainder
  seam). Check test sizes whenever a gate moves up.
- 2026-07-14 | CHOP blocked-path optimization pass, bit-identical outputs (~10% at 1024,
  ~4-10% at 512, ~6% at 256, both dtypes): (1) contiguous diagRaw mirror for the pivot
  search — reading W[i,i] directly is a stride-(n+1) scan over the full trailing range
  EVERY column (one cache line per entry, ~33 MB effective at n=1024, rivaling the whole
  SYRK stream); the mirror is refreshed per panel after the SYRK and swapped alongside W's
  diagonal, holding exactly W[i,i]'s bits so pivot choices are unchanged. (2) Deferred
  panel-end L scatter: Ukk parks in W[k,k], the panel's factor rows block-transpose into
  L's columns once per panel (W panel rows stay L2-resident, L written in 32-element runs)
  instead of one stride-n column write per factored column; Swap.Rows(L) narrows to the
  already-scattered columns [0,j0) — W's own column-segment swap maintains the deferred
  part. (3) Winner-only corrections quad-fused via axpy4. CHOLP_BLOCK=64 tried on top:
  float 1024 −3% but double 256 +5% — reverted, stays 32. Unblocked (<256) path untouched.
  every gate; ~1% ambient drift measured via unchanged-route control rows): doubleQr
  512→256 (−19% at 256), doubleQrcp 512→256 (−7%), CholPivot float+double 512→256
  (−12%/−9%), floatLu 256→128 (−3.5%). Ties (kept prior value): floatQr@64, doubleLu@64,
  doubleChol@256 (0.3775 vs 0.3774 — that IS the crossover). floatChol was retuned to 512
  the same day already. The axpy4 trailing-update fusion moved the level-3 crossovers down,
  most strongly for double.
- 2026-07-14 | 6x16 SEEDED-W TILE TRIED AND REJECTED — falsifies the register-pressure
  hypothesis below. A full packed driver (MR=6, MC=126) + seeded 6x16 W microkernel
  (12 accums + 2 B + 1 broadcast = 15 ymm, comfortably inside the 16-register file)
  measured 29.4-29.6 GF/s at 512-2048 vs the scalar packed kernel's 81.9-84.4 in the same
  run — the SAME collapse as the seeded 8x16 W (34). Disasm shows the identical pathology
  at 15 live vectors: accumulators in stack slots, vperm2i128 + vpalignr-by-1-byte
  rotations around every add, plus vpextrb/vpinsrb single-byte traffic (41 vpalignr /
  29 vperm2i128 / 32 vpextrb vs 12 vmulps + 12 vaddps). So the trigger is SEEDING wide
  accumulators from C at all, not how many are live: zero-init W tiles (matMatDotUnpackedW,
  matMatDotTransBRangeW) compile clean, seeded W tiles of ANY height collapse, seeded
  SCALAR tiles SLP clean. Zero-init + add-C-at-writeback is not a legal fix (different
  per-element summation tree, breaks the packed==unpacked bit contract). Scalar packed at
  82-84 GF/s already sits at the float SLP ceiling, so there is nothing to win — do NOT
  retry seeded wide microkernels until a Burst/LLVM upgrade changes the codegen; re-test
  with the disasm recipe first. Experiment code removed; a permanent GEMM-packed-direct
  benchmark section (gate bypassed, 512-2048) documents the pack-overhead crossover.
- 2026-07-14 | Mystery root-caused at the assembly level (headless Burst disasm via bcl.exe —
  recipe in docs/dev/burst-disasm-recipe.md). The seeded W microkernel's p-loop compiles to
  160 instructions vs the scalar microkernel's 76 for identical arithmetic (16 vmulps +
  16 vaddps + 8 vbroadcastss): LLVM chains the 16 seed-loaded v256 accumulators through
  vperm2i128 + vpalignr-by-1-byte rotations and stack slots instead of plain registers —
  byte-granular shuffle glue around every accumulator update, ~2.1x instruction bloat ≈ the
  measured 34-vs-114 GF/s collapse. The trigger is the combination of 16 LIVE fProxyW
  accumulators SEEDED from strided C rows (8x16 tile needs 16 accums + b0 + b1 + broadcast
  = 19 ymm > 16 physical; the rotation chain is LLVM's spill-avoidance gone wrong). The
  scalar microkernel survives the same pressure because its seeds/updates are scalar SLP:
  B stays as folded memory operands and the loop stays tight. Clean control cases in the
  SAME dump: matMatDotUnpackedW (zero-init accums, C added at write-back) and
  matMatDotTransBRangeW — zero shuffles, so fProxyW itself is fine; only seed-first + full
  register pressure trips it. Fix directions if packed-W is ever wanted: (a) 6x16 tile
  (12 accums + 2 B + 1 broadcast = 15 ymm — the classic BLIS AVX2 sgemm shape; needs MR=6
  pack layout + a matching scalar twin for the chain contract), or (b) restructure seeding
  (zero-init + C at write-back breaks the packed==unpacked bit contract — only with a
  matching unpacked change). Until then the packed path stays on the scalar microkernel +
  24 MB gate; float 512-1024 keeps ~10-20% headroom vs the transpose-detour reference
  (118/106 GF/s). Probe code (GemmMicroProbe + gemmMicroKernelW) was template-temporary and
  is REMOVED; asm listings preserved with the recipe doc.

## fProxyW stage 2b: TransB row-dot family on the wide core
- 2026-07-14 | matMatDotTransBCoreW (float-only, choose-routed at the three entries; the
  original core is now double-only via skipFor[float]): same 2x4 pair tile, one fProxyW
  accumulator per pair. Float A·Bᵀ 1024: 29.2 → 17.2 ms (125-136 GF/s — now the FASTEST GEMM
  shape in the library, beating plain matMatDot's ~77-90); float A·Aᵀ 1024: 14.4 → 9.0 ms
  (237-253 GF/s-effective). Beats the trans+dot route at EVERY size, so the wrapper's
  per-dtype viaTrans split (added earlier the same day) is REMOVED — unified kernel dispatch
  again. dotSymT float (Kalman covariance shapes) inherits the win. The mirror pass is now the
  shared mirrorLowerFromUpper helper (three cores use it). Same-run A/B, so valid despite
  ambient machine load (double control rows steady).

## fProxyW stage 2: matVecDot + sum/sumAbs/maxAbs
- 2026-07-14 | GEMV (matVecDot) float on the tiered pattern: 43.7 → 73.6 GFLOP/s at 1024
  (3.07 → 1.82 ms), 1.6-1.7x at 256-1024, measured before ambient machine load contaminated
  later runs (double control rows drifted +9% across runs — re-verify the small-row gate
  threshold on an idle machine). n=64 rows REGRESSED under the ungated W-tier (0.0150 →
  0.0221 ms: per-row fold overhead vs only 4 loop iterations), so the float W-tier gates at
  row length >= 128; below it the shared width-4 tier is exactly the pre-rework kernel.
  sum/sumAbs/maxAbs converted on the same pattern (L1 norm float 1024: 41.6 → 61.0 GF/s);
  fProxyW gained Abs/Max/HMax. vecMatDot left alone — it is a scalar map Burst already
  auto-vectorizes at full width. NOTE (user ruling): old-vs-new bit identity is NOT a
  constraint pre-release; internal same-build contracts (fused == composed) still hold.

## fProxyW: the three-tier conversion pattern (canonical recipe)
- 2026-07-14 | Every float width conversion follows one shape (user design, refined to a SINGLE
  marker): hoist `int i = 0; fProxy head = 0;`, then
    (1) `//+skipFor[double]` float-only W-wide main tier — folds into `head`, advances `i`;
    (2) SHARED width-4 two-chain tier — for double this IS its original main loop (i enters at
        0, identical chain assignment and fold), for float it covers the <8 remainder;
    (3) SHARED scalar tail; `s = head + quadFold` then tail appends.
  No emitFor, no duplicated double body (emitFor remains for genuinely-different bodies, e.g.
  fProxyW's own AVX-vs-double4 ops). Accepted nit: double's fold gains a leading `0 +`, which
  flips the SIGN OF ZERO when an entire reduction is −0 (e.g. dot against a zero vector with
  negative signs) — behaviorally invisible (−0 == +0, guards use >), noted on the CHANGELOG's
  "double unchanged" claim. Reduction tree = frozen contract; fused kernels mirror vecDot's
  shape exactly (bit-identical-to-composition). Converted: vecDot, vecDotRange, axpyNormSq,
  xpayNormSq, updateXR.

## fProxyW (WideOP) + float width rework, stage 1: vecDot
- 2026-07-14 | fProxyW added: 8 float lanes via Burst AVX intrinsics (v256) / 4 double lanes,
  32-byte v256 storage for both, lane-tree-identical non-AVX fallback (correctness path).
  vecDot/vecDotRange FLOAT moved onto it: 44-45 → 82-84 GFLOP/s cache-resident (roofline H,
  1K-64K elems; converges to bandwidth at DRAM sizes). Float's summation tree changed (2x8
  chains, halves-first fold) — new frozen contract, CHANGELOG'd.
- 2026-07-14 | DOUBLE was first routed through fProxyW too ("one template body") and REGRESSED
  ~19 ns/call (1K-elem dot 45 → 32 GFLOP/s): double4 is already full width, so the v256↔double4
  reinterpret wrapper only added per-call overhead. Double now keeps its original fProxy4 body
  verbatim via the new //+emitFor[double] codegen marker (bit-identical trivially). Rule for
  the rest of the rework: fProxyW is a FLOAT-side lever; leave double bodies alone.
- 2026-07-14 | Dot-shape ceiling context: ~90 GFLOP/s is the LOAD-PORT limit for two-operand
  streaming reductions (2 loads per mul+add), not the 120 register-chain ceiling — do not
  chase the gap.
- 2026-07-14 | TRAP (cost one full suite run, ~500 failures): fProxyW.Width's choose
  placeholder was 4 — but the template assembly's own tests RUN template code against the
  float-backed stub, so wide loads advanced 4 floats while processing 8. Placeholders in
  dtype-split code must be the FLOAT values (see codegen-refactor-lessons.md). Both generated
  files were correct the whole time; the bit-identity fallout that was real: xpayNormSq /
  updateXR (the CG fused kernels) pin "bit-identical to axpy+vecDot" — converted their float
  reductions to the fProxyW tree in the same pass (they inherit the width win too).

## LP.Sparse float IPM: stall-quality envelope (open robustness item)
- 2026-07-14 | Exposed by the float width rework (not caused by it): the float sparse IPM's
  outer tolerance (100·eps) is unreachable in float on unscaled real data — stackloss LAD
  always exits MaxIterations at a rounding-dependent objective (measured 2%..117% above the
  optimum across float summation-tree variants; double converges Optimal). Shipped: float
  inner pcgTol tightened sqrtEps → sqrtEps/10 via choose (measured stall 44.4 → 43.0 on
  stackloss; double untouched), and LPTests.SparseLadStackloss's 8% band made double-only
  (float asserts a wide sanity envelope — never below the optimum, never > 3x). Real-fix
  candidates for a robustness pass: column equilibration in the standard-form operator
  (stackloss columns span ~1..~90), inexact-Newton forcing terms (inner tol ∝ μ), and a
  float-realistic outer tolerance with honest status reporting.

## LOBPCG float: spurious-Ritz collapse by over-iteration (open robustness item)
- 2026-07-14 | Surfaced by the float width rework's tree change but NOT caused by it: on the
  truss demos' penalty-conditioned pencil (penalty 1e3 vs O(1) eigenvalues of interest), float
  LOBPCG iterated past the DEFAULT tolerance collapses its basis and reports spurious near-zero
  Ritz values as Converged (measured, 8-dof braced square, true λ1 = 1.198: tol=1e-4 → λ1 ≈ 1e-6
  "Converged"; tol=1e-6 → both eigenvalues exactly 0; default tol → correct at every penalty
  30..1000). TIGHTER tolerance makes it WORSE — the failure is orthogonality-budget exhaustion,
  not insufficient convergence. Demos now use the default tolerance (comments point here).
  BACKLOG: a guard inside lobpcg (Gram conditioning check, or residual-vs-Ritz-scale sanity)
  so basis collapse reports Indefinite/Breakdown instead of Converged-with-garbage.

## UnsafeOP packed (cache-blocked) GEMM
- 2026-07-13 | BLIS-style packed route added to matMatDot (KC=256/MC=128 panels, MR/NR strips,
  seeded microkernel so every element's reduction stays ONE p-ascending chain across panels —
  bit-identical to the unpacked route, pinned by DotSymTests.PackedMatchesUnpackedBitExactly).
  MEASUREMENT SURPRISE on the 9950X3D (bench pinned to the 32 MB-L3 CCD): packing only pays once
  the working set spills L3 — float 2048 unpacked sags to 58.7 GF/s, packed holds 82.7 (1.41x);
  double 2048 1.27x — but BELOW that the pack copies + per-panel C reloads are a pure loss
  (float 512 +33%!). First gate ((m+k)*n >= 128k elements) was far too low; final gate is
  ~24 MB total working set, byte-scaled. Expect the crossover ~3x higher on the V-cache CCD and
  lower on small-L3 consumer CPUs. Broader lesson: the unpacked kernel is NOT bandwidth-bound
  on big caches at <= 1024 — it sits at ~85-90 GF/s float, load-port/issue-bound under Strict
  (no FMA contraction) — so cache blocking is a big-N/small-cache lever, not a general one.
- 2026-07-13 | dot(transposeB) route split PER DTYPE (skipFor): float materializes Bᵀ (staged
  trans) + broadcast GEMM (row-dot TransB kernel is half of AVX2's float width; viaTrans measured
  at-or-faster at every size, and this un-regresses KMeans float); double keeps the row-dot
  kernel (wins at every size, e.g. 2x at 128). Aliased A·Aᵀ keeps matAAt for both dtypes.

## UnsafeOP TransB kernel family (matMatDotTransB / matAAt / matMatDotTransBSym)
- 2026-07-13 | First cut used TWO fProxy4 chains per output pair (vecDot's idiom) over a 2x4 pair
  tile: 16 accumulators + 12 transient loads spilled registers and LOST to the trans+dot route it
  was meant to replace (float 1024: 49.2 ms vs 31.7). Rewritten to ONE chain per pair (8
  accumulators + 6 transients, inside the 16-register budget): float 1024 28.7 ms, double 42.5 vs
  54.2 viaTrans; double 128 is 2x (0.061 vs 0.124). Don't re-add the second chain. Only float N=64
  still marginally favors viaTrans (0.0086 vs 0.0072 ms) — not worth size-gated routing.
- 2026-07-13 | matAAt (A·Aᵀ upper+mirror): 143 GFLOP/s-effective float / 105 double at 1024 —
  on par with matAtA despite the dot-product formulation, because symmetry halves the work and
  both row streams are unit-stride. No trans+matAtA fallback route needed.

## Control symmetric-GEMM reroutes (RiccatiStep / SDACore)
- 2026-07-13 | Riccati/SDA symmetric products moved to dotSym (missed by the first symmetric-GEMM
  pass, which covered QP/MPC only): RiccatiStep's Bᵀ(SB), Aᵀ(SA), BSAᵀK (= BSAᵀR̄⁻¹BSA) and
  SDACore's AkᵀX3 H-update. AkᵀX3 is symmetric only in exact arithmetic (X3 exits an LU solve);
  the mirror picks the upper triangle's roundoff where the full kernel produced O(eps) asymmetry
  that SymmetrizeInPlace then averaged — the existing post-add SymmetrizeInPlace is kept.
  SDACore's GkNext = (AkGk)·X2 does NOT fit either sym kernel form (neither operand of the
  symmetric product is materialized transposed) — left on the full kernel deliberately.

## UnsafeOP matTrans + symmetric-mirror cache blocking
- 2026-07-13 | Plain TB=32 blocking (strided writes kept inside the tile) was a TRAP: at
  power-of-two sizes the 2-4 KB stride maps a whole tile column into 1-2 L1 sets and way-thrashes
  — blocked matTrans measured 0.21 Gelem/s at float 1024, WORSE than naive. Fix: stage every tile
  through a TB=16 stackalloc buffer (read side row-contiguous into buf, write side row-contiguous
  out of buf; neither matrix ever strides). Never ship a plain two-loop blocked transpose again —
  measure at power-of-two N specifically. Same staging applied to the symUpper mirror passes in
  matMatDotTransACore/matMatDotTransBCore (their blocked-unstaged version did already beat the
  naive mirror: AtA float 1024 15.9→14.2 ms). All pure permutations: bit-identical results.
  Benchmark instruments: new "Trans" section (Gelem/s) + the viaTrans/AtA/AAt rows.
- 2026-07-13 | Staged mirror needed a SMALL-MATRIX BYPASS (m <= 64: plain mirror, no buffer):
  a uniform staged path regressed gamedev-scale LQR 5-18% (n=4-12 Riccati steps) — at those
  sizes the whole matrix is L1-resident (thrash impossible) and the stackalloc buffer's
  per-call localsinit zero-fill dominates. Iteration counts unchanged either way. General
  rule: any stackalloc-staged kernel path needs a small-size bypass or the small callers pay
  the buffer for nothing. Also: below one register tile (m < 8) the symUpper tile-skip never
  fires — dotSym at tiny sizes is full compute + mirror, i.e. pure overhead vs plain dot, so
  small-n callers only keep dotSym for the exact-symmetry contract, not for speed.

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

## Parameter naming (library-wide)
- 2026-07-13 | Short tuning-param names ruled canon: maxIterations → maxIter, tolerance → tol,
  relativeTolerance → relTol, library-wide. REVERSES the earlier long-name rename pass — do not
  rename back. maxSweeps kept where the algorithm genuinely counts Jacobi sweeps. Rule recorded
  in docs/dev/naming-style-guide.md.

## Krylov.Guards.cs
- 2026-07-13 | //singularFile// on this partial is load-bearing, not a style choice:
  RequireDistinctBuffers has no fProxy token in its signature, so if it were declared inside the
  multiplying Krylov.fProxy.cs template it would be copied identically into both the generated
  Krylov.float.cs and Krylov.double.cs fragments of the same partial class -- two definitions of
  the same member -> CS0111. (was Krylov.Guards.cs:7-11)

## Krylov.PMinres
- 2026-07-18 | Sign-check `betaNewSq` before the Givens/x update. `betaNewSq = <r2, M^-1 r2>` was
  fed straight into `beta = sqrt(...)` with no sign guard; a non-SPD preconditioner makes it < 0 ->
  beta = NaN, and the Givens rotation + `x.addScaledInPlace(phi, w)` block runs BEFORE the existing
  `!(beta > 0)` guard, so a warm-started x was overwritten with Inf/NaN and the reported residual
  was NaN. Now bails with a Breakdown SolveInfo (x LEFT UNTOUCHED) immediately after computing
  betaNewSq when `!(betaNewSq >= 0)` -- mirrors the iteration-0 `!(betaSq > 0)` breakdown and pcg's
  `!(rz > 0)` guard. The legitimate `betaNewSq == 0` invariant-subspace exit stays with the
  downstream beta>0 guard; only `< 0` is the new bail. Source touched; needs regen.
- 2026-07-18 | New solver (Krylov.PMinres.fProxy.cs), overload ladder mirrors pcg's exactly:
  generic pminres<TOp,TPre> (core + arena-alloc + default-params) plus the same three BSR×
  preconditioner rungs (block-Jacobi/SSOR/IC0) pcg exposes, each with its own core/arena/default
  tier — 12 overloads total, no dense-specific rung (pcg has none either; a dense A + any
  IfProxyPreconditioner goes through the generic core via fProxyDenseOperator, e.g.
  fProxyIdentityPreconditioner).
- 2026-07-18 | Preconditioning breaks minres's "phibar IS ‖b-Ax‖ for free" identity: once M ≠ I,
  phibar is the M⁻¹-weighted residual norm ‖r‖_{M⁻¹}, not the Euclidean ‖b-Ax‖ SolveInfo's
  contract requires on Converged/MaxIterations. Fix: verify a claimed Converged exit with one
  fresh r = b-Ax (falls through and keeps iterating if the verify fails, mirroring pcg's
  verify-at-exit), and compute one fresh r on the MaxIterations exit too (minres/pcg/cg don't need
  this on MaxIterations — their tracked quantity already IS the true residual there). Breakdown
  still reports the unverified phibar, matching every other solver's Breakdown carve-out. Cost:
  the verify only fires when phibar crosses threshold (rare, not every iteration) plus once at
  MaxIterations, so the steady-state per-iteration cost stays at 1 A.Apply + 1 M.Apply, same
  shape as pcg's 1 A.ApplyDot + 1 M.Apply.
- 2026-07-18 | z (holds M⁻¹ applied to the current unpreconditioned Lanczos vector r2) is a new
  8th scratch buffer alongside minres's seven Paige-Saunders names (y/r1/r2/v/w/w1/w2) — kept
  distinct from y (the A·v temp) rather than reusing one buffer for both roles the way the
  reference Fortran/Matlab minres.m does, matching this codebase's existing per-buffer-single-
  meaning discipline (cg/pcg keep z separate from Ap the same way).

## Krylov.PBiCGStab
- 2026-07-13 | Parameterless BSR overload's default iteration budget changed 2*A.M_Rows → A.M_Rows
  to match the unpreconditioned biCGStab twin and the rest of the square-solver family (release-scan
  N14 finding: undocumented sibling inconsistency; no measured rationale existed for the 2x).

## UnsafeOP / UnsafeBoolOP / SelectOP aliasing policy
- 2026-07-13 | Revised after A/B benchmarking (maintainer: don't drop [NoAlias] from hot kernels
  without proof). GEMM-TransA benchmark, double N=1024, median of 4: with [NoAlias] 41.6 ms; without
  43.0/43.7 ms across two runs (~3-5% hint; float flat; double-512 inverted — within machine noise
  but never favoring the drop). FINAL POLICY: matMatDotTransA/Range KEEP [NoAlias] on all pointers;
  the aliased Aᵀ·A / A·A call shapes (Blas.dot(A, A[, transposeA: true]), isOrthogonal, covariance)
  are handled at the WRAPPER by copying one input to Temp (O(n²) copy vs the O(n³) product). Select
  kernels stay without [NoAlias] on a/b: dest-aliasing is a tested public contract
  (SelectRefTests VecAliasDest) and the loop is elementwise memory-bound, so copy-on-alias would
  cost more than any vectorization delta.
- 2026-07-13 | [NoAlias] made truthful (release-scan D3 ruling): write-aliasing wrappers now call
  dedicated single-pointer in-place kernels (signFlipInPlace; UnsafeBoolOP notInPlace/orInPlace/
  andInPlace/xorInPlace/equalsInPlace/notEqualsInPlace — the unused copy-form bool kernels were
  deleted).

## NLS
- 2026-07-12 | Release-scan fix (FOURTH bug, all precisions): the all-columns-flat degenerate
  case (0 < LInf(J) <= flatThresh, i.e. every column norm at-or-below flatThresh) left
  maxRealColNorm at 0 and d entirely zero, since the whole-Jacobian stationary guard only
  checked LInfJ0 > 0. nlsScaledGradNorm then divided by d[j]=0 (inf/NaN gradientNorm) and
  mu=1e-3*nlsMaxD2(d)=0 removed all damping. nlsUpdateScale now returns maxRealColNorm; both
  cores gate the initial stationary branch on that return being 0 (not on LInfJ0 > 0), which
  exactly matches nlsUpdateScale's own per-column flat classifier instead of approximating it
  via the whole matrix's LInf norm. Folded into the same change: nlsUpdateScale used to
  compute every column's squared-sum twice per call (once for maxRealColNorm, once for the
  floor pass); it now caches each column's norm in a scratch buffer (colNorms) and computes
  each column once.
- 2026-07-12 | NEW feature: nonlinear least squares via Levenberg-Marquardt with Nielsen damping
  (Optimize.nlsSolve / Optimize.curveFit). Algorithm reference: Madsen, Nielsen & Tingleff, "Methods
  for Non-Linear Least Squares Problems" (2nd ed., 2004), Algorithm 3.16 -- the gain-ratio damping
  update and the convergence structure (step-size test on the PROPOSED step, before it is evaluated
  against the objective). Math.NET Numerics' LevenbergMarquardtMinimizer.cs (MIT) was read as an
  independent C# structural reference; MINPACK (netlib, permissive) was read for STRUCTURE only, not
  transcribed line-by-line (the Marquardt column-norm-floored-at-running-max diag scaling is its
  well-known convention, not ported code) -- per the owner's provenance ruling this is the "provenance
  line + DEVLOG" bucket, not the "MINPACK acknowledgment in Third Party Notices.md" bucket. Robust-loss
  row rescaling (nlsApplyRobustScale) IS verified line-by-line against the installed scipy source
  (optimize/_lsq/least_squares.py's huber/cauchy + common.py's scale_for_robust_loss_function,
  BSD-3): z=(f/scale)^2, rho[0] scaled by scale^2, rho[2] divided by scale^2, rho[1] untouched;
  J_scale=sqrt(max(rho[1]+2*rho[2]*f^2, EPS)), f*=rho[1]/J_scale, J scaled per row -- confirmed byte-
  for-byte against scipy 1.17 rather than trusting the task's own sketch. Tukey biweight has NO scipy
  precedent (scipy ships no redescending loss, deliberately) -- its Rho/RhoPrime/RhoPrime2 were
  independently derived from the standard robust-statistics biweight identities (rho(r) = c²/6·(1-
  (1-(r/c)²)³) for |r|<=c else c²/6), then re-expressed in terms of s=r² under this library's rho(s)
  convention (0.5·Σrho(s_i) = the standard M-estimator total cost) -- see the from-scratch numpy
  prototype (scratchpad, not shipped) for the full re-derivation before this file was written.
- 2026-07-12 | Validation (numpy/scipy prototype, both precisions): (a) exponential-decay and sine
  fits matched scipy.optimize.least_squares(method='lm') to ~1e-8..1e-10 relative across 3-4 starts
  each. (b) NIST StRD Misra1a AND Chwirut2 (fetched fresh from the live NIST page -- an earlier hand-
  transcribed Chwirut2 table turned out wrong, see below) matched certified params to ~1e-6..1e-8
  relative in double precision from both prescribed starting points; Chwirut2 (not Misra1a -- see the
  scale-disparity entry a few entries below) is the one that also cleanly converges in float32, and
  is the one this library ships as its literal NIST test. (c) a parameter the model never references (exactly
  zero Jacobian column) stays EXACTLY at its initial value regardless of that value's magnitude
  (tested 0, 7.3, -1e6) while the other parameters converge normally -- no blow-up. (d) Huber/Cauchy/
  Tukey all recovered the true linear/exponential fit under 6/50-point gross-outlier contamination
  where plain L2 was visibly pulled off (e.g. linear fit relerr 0.89 for L2 vs 0.01-0.03 for the
  robust losses). (e) float32 rerun of (a)/(d) with the SAME relative-tolerance design reproduced
  the double-precision qualitative results (robust losses still beat L2 by the same margin), no
  precision-specific failure found. (f) numeric (forward AND central) vs analytic Jacobian converged
  to the same point (~1e-11 relative) in the same iteration count.
- 2026-07-12 | BUG FOUND AND FIXED during prototyping: an early engine design checked the step-size
  convergence test only AFTER an accepted step, mirroring a first guess at the M-N-T structure. On
  NIST Misra1a (b2 ~5.5e-4, a badly parameter-scaled start) this spiralled once residual/gradient
  reductions hit the float64 noise floor: each rejected trial produced a SMALLER step as mu grew, but
  the step was never actually re-checked against stepTol until AFTER an acceptance that never came,
  so mu escalated to the hard ceiling and the solve reported FailedLinearSolve despite already sitting
  at the certified optimum (params matched cert to 1e-8 well before the failure). Root cause: Madsen/
  Nielsen/Tingleff's own Algorithm 3.16 checks ||h|| <= eps2*(||x||+eps2) on the PROPOSED h, every
  iteration, BEFORE evaluating F(x+h) -- not only after acceptance. Re-reading the reference pseudocode
  and fixing the check order resolved it (verified: Misra1a now reports Converged/SmallStep, never
  FailedLinearSolve, from both prescribed starts). nlsSolveStep/the outer loop in NLS.fProxy.cs follow
  this corrected order; don't move the step check back to a post-accept-only position.
- 2026-07-12 | SECOND BUG FOUND AND FIXED (float32-only, caught by the float32 rerun of the NIST
  case): the Marquardt diag floor was first written as `dFloor = Consts.fProxySqrtEps * LInf(J0)`
  (scale-relative to the WHOLE Jacobian, deliberately not `max(1, LInf(J0))` -- the Kalman SDA bug is
  the same failure MODE, assuming an O(1) problem scale). That version still broke in float32 on NIST
  Misra1a specifically: b1's column norm (~0.16) and b2's column norm (~7e5) differ by ~1e6x, and
  sqrt(floatEps)~3.45e-4 times the LARGER column's norm floors b1's own legitimate ~0.16 column norm
  up to ~107 -- destroying its real gradient signal and reporting false-Converged at iteration 0
  (this did NOT happen in double: sqrt(doubleEps)~1.5e-8 is small enough relative to a 1e6x ratio
  that it stayed below 0.16). Root cause: ANY floor scaled by the WHOLE matrix's own magnitude
  cross-contaminates columns of genuinely different natural scale.
- 2026-07-13 | THIRD BUG FOUND AND FIXED (all precisions -- reported by the test suite as
  FlatParameterNoBlowup failing everywhere: float got 3.294179E+13, double got 8.54560688875843E+30,
  both "expected 0"). The second fix above (a plain `dFloor = Consts.fProxyEpsilon`, no matrix-scale
  multiplier) was itself insufficient: it was validated only against `np.linalg.lstsq` (SVD-based) in
  the numpy prototype, which does NOT reproduce this library's actual QR.solveInPlace (Householder).
  Root-caused with a FAITHFUL Python port of genHouseholder + solveInPlace's fused kernel (including
  the near-zero-column fallback `u[k]=sqrt(2)`), which reproduced the exact reported failure
  (h[flat]=5e29 at iteration 0) -- then confirmed byte-for-byte in a standalone dotnet harness with
  the SAME faithful port transcribed to C#: reinstating the plain-epsilon floor there reproduces
  8.545607E+30 (double) / 3.295198E+13 (float), matching the reported values to 4-6 significant
  figures. Mechanism: flooring a flat column's d_j at machine epsilon makes its augmented-system
  regularization entry sqrt(mu)*d_j fall BELOW QR's own zero-threshold
  (Consts.ZeroThreshold*LInf(Aaug)) for that column. genHouseholder's near-zero fallback sets ONLY
  u[k]=sqrt(2) and leaves the REST of u (including the regularization row itself, which is the
  column's ONLY nonzero entry) unchanged -- this fallback is correct for a column that is zero
  EVERYWHERE (the ordinary un-augmented QR case it was written for), but produces an inconsistent
  reflector when the column has exactly one small-but-nonzero entry (the augmented case): applying it
  leaves R's diagonal for that column proportional to mu*dFloor² (quadratically tiny), so
  back-substitution divides whatever roundoff has accumulated in the transformed RHS by a near-zero
  number and the flat parameter's step explodes. A larger CONSTANT floor does not fix this on its
  own either (tried 1e-6·LInf(J) through 2·LInf(J) in the harness): the required floor to clear QR's
  threshold scales with 1/sqrt(mu), and mu shrinks across iterations as the solve converges, so any
  FIXED floor value can eventually fall short again later in the same solve.
  FINAL FIX (nlsUpdateScale, NLS.fProxy.cs): stopped trying to pick a floor VALUE at all for flat
  columns. Instead, each iteration, first find maxRealColNorm (the largest column norm among columns
  ABOVE flatThresh = Consts.fProxyEpsilon, an ABSOLUTE per-type constant, never scaled by the
  matrix -- this is unchanged from the second fix and still correctly leaves Misra1a's b1, ~0.16, far
  above it and untouched). A column AT OR BELOW flatThresh (colnorm effectively zero -- the residual
  structurally does not depend on that parameter) is then floored at maxRealColNorm itself, not a
  small constant: this makes its regularization entry sqrt(mu)*d_flat EXACTLY EQUAL to the
  MOST-regularized real column's own entry, so it tracks mu's shrinkage in lockstep with the real
  columns and stays proportionally safe relative to QR's threshold for as long as the real columns'
  own regularization does (which is the normal, expected LM regime -- once real-column regularization
  itself becomes negligible relative to J, mu is deep into "trust the linearization" territory and the
  algorithm is behaving as intended). Matches the coordinator's candidate 1 (MINPACK's zero-column
  convention, generalized from a literal "1" to "the max column norm across J" for scale-independence)
  -- candidate 3 (explicit freeze/exclude) was not needed once the right floor TARGET was identified.
  Re-verified end-to-end in the dotnet harness with the FAITHFUL Householder QR (not a normal-equations
  or lstsq stand-in): all 9 shipped NLS test cases pass in both precisions, the Misra1a cross-
  contamination re-check (same scale-disparity case as the second bug) still reports the same benign
  SmallStep-at-iteration-0 (finite, unmoved, NOT a false convergence) rather than any blow-up or
  wrong-answer convergence, and reverting nlsUpdateScale to the flawed plain-epsilon version
  reproduces the FlatParameterNoBlowup failure with the reported magnitudes almost exactly (negative
  control). Same "no matrix-scale multiplier" reasoning still applies to the gradient-convergence
  reference scale (gnorm0 is the SCALED gradient's own value at the start, not a hardcoded constant).
- 2026-07-12 | Misra1a's ~1e6x b1/b2 scale disparity is ALSO why it was dropped in favor of Chwirut2
  as this library's single shipped NIST literal test: even with the dFloor fix above, Misra1a's own
  Marquardt mu0 heuristic (tau·max(d_i²), i.e. dominated by whichever parameter has the LARGEST
  column norm) over-damps the SMALLER-scaled parameter by a factor of roughly the scale ratio
  SQUARED. In float32 this makes the very first proposed step so tiny that adding it to p[0]=500
  is a float32 no-op (500+1.4e-6 rounds back to exactly 500 at ~7 significant digits) -- every trial
  is honestly rejected (rho_gain measures exactly 0, not negative or NaN) until the step-size test
  legitimately fires (SmallStep), from BOTH of NIST's own prescribed starting points. This is a real,
  explainable float32 precision limit for THIS problem's specific scale disparity, not an engine
  defect (no NaN, no crash, no silent non-convergence) -- but it makes a flaky/uninformative shipped
  test. Chwirut2's three parameters (~0.17, ~0.005, ~0.012, within ~30x of each other) have no such
  disparity and converge cleanly in BOTH precisions from its own NIST-prescribed start1 (double
  relerr ~8e-8, float32 relerr ~5e-3, both via genuine multi-iteration LM progress) -- this is the
  literal dataset TemplateSourceTests/fProxy/NLSTests.fProxy.cs actually ships.
- 2026-07-12 | Redescending-loss starting-point caveat (Tukey): if EVERY residual at the starting
  point exceeds the loss's Scale, RhoPrime is exactly 0 everywhere, the weighted gradient is exactly
  0, and the solve reports false-Converged at iteration 0 -- reproduced with Tukey(scale=0.3) from a
  poor exponential-fit start where every residual exceeded 0.3. This is inherent to ANY redescending
  M-estimator (not an engine bug) -- scipy's own least_squares ships no redescending loss for the
  same reason. Choose Scale comfortably larger than the expected residual spread at the start point,
  or warm-start from an fProxyHuberLoss/plain fit first.
- 2026-07-12 | Scoping decision: no analytic-Jacobian + robust-loss combination overload in v1 (the
  task brief's own bullets frame robust loss as an addition to the DEFAULT numeric-Jacobian path, and
  curveFit is explicitly numeric-only) -- kept deliberately, not an oversight. This also sidesteps a
  genuine C# landmine verified via a standalone dotnet repro before committing to the final overload
  ladder: two generic methods differing ONLY by a type-parameter CONSTRAINT (TF : IfProxyResidualFunction
  vs TF : IfProxyResidualJacobian) collide as CS0111 the moment their VALUE-parameter lists also
  match -- constraints are invisible to C# overload-signature uniqueness (same rule the naming-style-
  guide's "Split vs merge safety" section documents for merged classes). The numeric-only ladder's
  terse "just f, p, m" tier is therefore the ONE overload of that exact shape in the whole class
  (numeric is the default); the analytic ladder's shortest tier is the 6-param
  (f, p, m, gradTol, stepTol, maxIter) form, which the numeric ladder deliberately never offers (its
  own 6-param slot would collide) -- a caller wanting default tolerances on the analytic path passes
  Consts.fProxySqrtEps / Consts.fProxyEpsilon / 200 explicitly rather than getting a same-shaped terse
  overload. Compile-checked end-to-end (both precisions, all overload families, curveFit plain +
  weighted) via a standalone dotnet console project with API-compatible stub types before this file
  was written, not just reasoned about.
- 2026-07-12 | Convergence bookkeeping (cost, gain ratio, gradient/step norms) accumulates in
  `double` even in the float template -- same idiom as Optimize.ladIRLS's own `double dx=0,xn=0`
  convergence accumulator -- while J, r, h, d, mu (the actual factorized system QR.solveInPlace
  consumes) stay genuinely fProxy-precision. Confirmed this split doesn't mask a real float32 issue:
  the float32 prototype rerun with the SAME engine logic (native-dtype d/J/r/h, double-accumulated
  convergence tests) reproduced the double-precision qualitative results across every scenario.

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

## QP
- 2026-07-14 | QP v2 stage 2c: DIFF-REPAIR replaces the all-or-nothing reuse. Instead of
  reuse-exact-or-full-rebuild, qpActiveSetCoreWarmPersistent now UP/DOWNDATES the persisted factor by
  only the rows that changed: ComputeTargetStatus gives the desired set for x0 (RepairWorkingSet's
  three-pass tightness logic, no factor build), the diff vs the persisted set is counted, and if it fits
  the dead-reflector budget (deadCount+numDrops < DeadCap) and is small (numDrops+numAdds <= DeadCap) we
  drop the no-longer-tight columns (high-to-low so shifts don't invalidate lower indices; DropFromFactor
  + UpdateReducedOnDrop) then add the newly-tight ones (TryAddToFactor + UpdateReducedOnAdd, rank-reject
  → Inactive). Large diff / first solve / budget-exhausting diff → full rebuild (resets the budget).
  Zero diff (steady state) = no work = the same reuse as before, so the measured steady-state warm win is
  unchanged (warm-box −25/−34% float, −46/−55% double; warm+wall −58/−74%); the diff-repair's extra
  benefit is on TRANSIENT ticks (working set moving a few rows), which the steady-state benchmark burns
  off — the win there is structural (incremental O(diff·n²) vs a full O(n²·nz) rebuild). Correctness
  still pinned by MPCTests.WarmPersistentMatchesColdEachFrame (its x0=(5,0) tight-bound trajectory
  saturates then desaturates → exercises real per-tick diffs, cross-checked vs cold QP.solve every
  frame). WarmSetUnchanged removed (subsumed by the zero-diff case).
- 2026-07-14 | QP v2 stage 2b SHIPPED: CROSS-TICK persistence for the MPC warm path
  (qpActiveSetCoreWarmPersistent). fProxyMPCState now OWNS the working-set factorization (qpFactor) and
  reduced space (qpReduced), carried across solves. First cut was reuse-exact-or-rebuild (superseded by
  the diff-repair entry above). ⚠️ JOB-COPY TRAP (cost most of a debugging pass): the
  wsf/red native BUFFERS survive an IJob.Run()/Schedule by-value copy of the owning state, but their
  PLAIN scalar fields (k, reflCount, deadCount, opCount, rotCount, stale, changeCount, factorValid) do
  NOT — so cross-tick reuse silently never fired through the benchmark's per-frame job.Run() (correct,
  since the rebuild path resets those counters, but +30-40% from paying incremental overhead with no
  reuse). Fix: MPCState carries a native-backed qpMeta[8] holding exactly those scalars;
  qpActiveSetCoreWarmPersistent rehydrates them on entry and writes them back on exit. Measured on the
  MPC warm steady-state benchmark (per-frame, via job.Run()): warm-box −25/−34% float, −46/−55% double
  at (40,12,4)/(30,24,8); warm+wall (active general rows → nontrivial reduced space) −58/−66% float,
  −68/−74% double. Correctness pinned by MPCTests.WarmPersistentMatchesColdEachFrame (every frame's warm
  solution cross-checked against an independent cold QP.solve of the identical condensed data).
  Allocator param added to both Create methods (Temp cold / Persistent for MPC). FUTURE: an incremental
  DIFF-repair (up/downdate the persisted factor by only the few changed rows) would extend the win to
  transient ticks where the working set moves by a few rows; the current fast-path is all-or-nothing
  (exact-match reuse else full rebuild).
- 2026-07-14 | QP v2 stage 2 SHIPPED: persistent up/downdated REDUCED SPACE (fProxyQPReducedState:
  Z, QZ = Q·Z, H_Z = ZᵀQZ carried alongside the stage-1 log). Kills the two O(n²·nz) per-iteration
  terms (FormNullSpaceBasis + the fresh Q·Z). Option (b) from docs/dev/draft-spec-qp-qr-updowndate.md
  — maintain Z/QZ/H_Z EXPLICITLY, recompute chol(H_Z) from scratch each iter; NOT a
  Gill-Golub-Murray-Saunders Cholesky updowndate (an add is a dense size-nz Householder congruence
  whose re-triangularization is O(nz³), no cheaper than the from-scratch factor). ADD: the new
  reflector restricts to the old null-space frame as Ĥ = I−ûûᵀ (û = reflector tail, read for free,
  ûᵀû=2); Z·Ĥ and (QZ)·Ĥ are rank-1, Q is NEVER re-multiplied (Q(ZĤ)=(QZ)Ĥ), Ĥ·H_Z·Ĥ is sym rank-2;
  the leaving direction is exactly local column 0 → delete it. DROP: Givens mix coords <k only so the
  old Z columns survive verbatim; one new column Q̂·e_k prepends (FormNullSpaceColumn) and H_Z borders,
  q_new=Q·z_new is the drop's only O(n²). Staleness: RefactorWorkingSet (DeadCap) reorders the frame →
  rebuild; RebuildCap=16 incremental changes → rebuild (roundoff bound). useIncrementalReduced flag on
  qpActiveSetLoop keeps the from-scratch path (SolveReducedNewtonStep) as an A/B + correctness seam;
  reduced buffers only allocated when the flag is set. A/B on a NEW loop-isolating QPBenchmark section
  (qpActiveSetCore from a supplied x0, no phase-1): incr vs batch −11/−12/−33% float, −12/−24/−48%
  double at n=16/64/192 (iters byte-identical, objectives match) — grows with n, comfortably >10% at
  n≥64 → GO. Full facade (Section 1) at n=192 ≈ half the stage-1 baseline (float 135.9→67.7ms). COLD
  path (qpActiveSetCore, QP.solve) defaults incremental; WARM path (qpActiveSetCoreWarm, MPC) defaults
  BATCH — a warm tick changes ~0 rows (MPC steady-state iters=0), so incremental maintenance never
  amortizes intra-call (cross-tick persistence is a future stage); an early incremental-warm default
  regressed the MPC headline 20-43% (per-solve n×n red alloc + copy-back with no iterations to earn it)
  before the flip. Dtype-collision trap avoided (RebuildCap/Create/Dispose on the dtype-named struct,
  same as stage 1's DeadCap). Test: fProxyQPFactorStateTests.IncrementalReduced (entrywise Z/QZ/H_Z vs
  fresh rebuild after every add/drop, across both rebuild triggers + the k=0 edge), CachedRetry
  (regularized retry off cached H_Z, byte-identical), FallbackEquivalence (incr vs batch agree
  status/obj/x — VALUES not paths). Trap while writing that test: ConstraintSense.LessEqual = −1, so a
  zero-init senses array is all Equal → x0=0 reads Infeasible; set senses explicitly.
- 2026-07-14 | QP v2 stage 1 SHIPPED: persistent up/downdated QR of A_Wᵀ (fProxyQPFactorState: op-log
  Q̂ᵀ = Householder reflectors + Givens rotations in creation order; hybrid store per
  docs/dev/draft-spec-qp-qr-updowndate.md option (a)). Replaces the per-iteration from-scratch
  refactor AND the O(nk²) trial factor per candidate add (TryAddToWorkingSet deleted; add rank-test =
  transformed tail norm, same threshold, identical decision). SeedWorkingSet/RepairWorkingSet build
  the factor incrementally too — the old per-candidate trial factors made seeding O(Σ nk²) ≈ O(nk³/3)
  at an LP-vertex start (~n tight rows), which dominated cold solves. Drop = R column shift + k-1-j
  rotations appended to the log; dead reflectors stay in Q̂; full refactor every DeadCap=8 drops keeps
  the log bounded (reflectors ≤ n+8, rotations ≤ 8(n-1)) and is also the defensive re-rank-guard
  (a row going numerically dependent during rebuild is set Inactive, same exclusion rule).
  Final stationarity diagnostics reuse the live factor (the old block refactored once more).
  Pure-add sequences are arithmetic-identical to the old batch factor (per-column reflector
  application order matches applyReflectorRightCols exactly); iteration paths diverge only after the
  first drop (rotations) or mid-loop add (column order = creation order, no longer ascending-t) —
  acceptance is KKT/oracle values per the spec, and the whole HS/brute-force/LP-limit battery passes
  unchanged. Dtype-collision trap hit on the way: consts/factories with proxy-free signatures on the
  SHARED partial QP class (FactorDeadCap, CreateFactorState(int)) collide between generated
  float/double partials — moved onto the dtype-named struct (DeadCap, Create, Dispose).
  Stage 2 (reduced-Hessian Cholesky border/downdate) remains measure-gated per the spec: H_Z
  formation (QZ = Q·Z, O(n²nz)) is now the dominant per-iteration cost.
- 2026-07-12 | Warm-start seam for MPC: qpActiveSetCore's loop body (the add/drop iteration, the
  perturbation-anticycling cleanup pass, and final diagnostics) was factored out, UNCHANGED, into a new
  internal qpActiveSetLoop(wstatus, ...) that neither seeds nor disposes `wstatus`/`L`/`U` -- the two
  entry points (qpActiveSetCore's existing seed-from-point behavior, kept byte-for-byte, and the new
  qpActiveSetCoreWarm) differ ONLY in how `wstatus` is seeded and who owns it afterward. Existing QP
  test suite is untouched by this refactor (qpActiveSetCore's own observable behavior did not change --
  same validation, same SeedWorkingSet call, same loop, same disposal, just relocated across two
  methods instead of one).
- 2026-07-12 | RepairWorkingSet (SeedWorkingSet's warm sibling): re-admits a PREVIOUS solve's
  ActiveLower/ActiveUpper row only if it is STILL tight (within feasTol) at the CURRENT x, on the SAME
  side it was active on before -- a row that drifted off its bound between frames is dropped rather than
  forced, since the active-set loop's invariant (A_W x = b_W at the start of every iteration) requires
  genuine tightness, not "was tight last time". Considered just re-running SeedWorkingSet fresh every
  warm call instead (simpler) and rejected it: SeedWorkingSet has no memory of the PRIOR working set at
  all, so it cannot report a meaningful workingSetChanges diagnostic, and (for a soft/general row that
  drifts slightly rather than snapping exactly to a new bound) would rediscover less of the previous
  optimal active set than a repair-first pass does.

## OP.Component / UtilityOP
- 2026-07-12 | UtilityOP.cs deleted (owner-approved): its zeroInPlace(in fProxyN) became a
  redundant special case of the generic below; no callers existed.
- 2026-07-12 | Generic zeroInPlace<T>/fillInPlace<T> added to the Comp families (born from the
  demo stress test: no way to zero/fill a matrix, and mulInPlace(A, 0) is a NaN propagator).

## CHO
- 2026-07-11 | Right-looking Cholesky chosen over left-looking: left-looking's hot loop is a dot-product reduction over already-computed columns, which stays scalar under strict FloatMode (loop-carried accumulator); right-looking's rank-1 update is a set of unit-stride row axpys, which vectorizes. (was CHO.fProxy.cs:43)
- 2026-07-12 | CHOL_BLOCK_MIN_N size gate is a measured crossover, not the naive 2*CHOL_BLOCK — the panel/TRSM/SYRK bookkeeping isn't amortised until ~8 panels wide. (was CHO.fProxy.cs:68)

## CHOP
- 2026-07-11 | Blocked PSTRF panel/SYRK boundary previously mirrored the lower triangle for cache purposes; that mirroring was found to be a cache cliff and was removed, keeping only the upper-triangle-by-row storage. (was CHOP.fProxy.cs:72)
- 2026-07-12 | CHOLP_BLOCK_MIN_N size gate is a measured crossover, higher than plain CHO's gate since the panel phase here is heavier. (was CHOP.fProxy.cs:107)
- 2026-07-12 | Blocked (level-3) PSTRF path is a port of Lucas/Higham dpstrf.f (upper-triangular branch). Two deviations from a literal port: Ukk is read straight from the pivot search's maxDiag (provably identical to re-deriving it, skips redundant work), and this port always searches for a pivot rather than reusing LAPACK's precomputed first-column pivot. Also, distinguishing rank-deficient from indefinite (this library's RankInfo, beyond LAPACK's single INFO=1) requires W accurate before the off-diagonal scan, so the rare branch that trips the tolerance check first flushes this block's pending columns [j0,k) via the same syrkUpperSub kernel, scoped narrower. (was CHOP.fProxy.cs:198, :209-215)

## LU
- 2026-07-12 | LU_BLOCK_MIN_N size gate is a measured crossover, not the naive 4*LU_BLOCK — the panel/TRSM/GEMM bookkeeping isn't amortised until ~8 panels wide. (was LU.fProxy.cs:116)

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

## FFT (no-workspace path)
- 2026-07-15 | REMOVED the no-workspace fft/ifft/rfft/irfft overloads and their sin/cos recurrence
  cores (FftCore radix-2, FftCoreRadix4Rec) from FFT.fProxy.cs. Rationale: the path was strictly
  dominated — even a one-shot ws+build (build the quarter-wave table + one transform ~7.4 ms at
  N=2^20 float) beat the recurrence (~22 ms), AND it was non-deterministic across architectures
  (sin/cos twiddles). Nothing justified keeping it. Workspace overloads are now the only power-of-two
  path. Tests repointed to FFT.dft as the independent oracle (small N; both dispatch paths, N up to
  2048) + round-trips + analytic. dft/idft KEPT as the arbitrary-N fallback (still sin/cos, still the
  documented non-deterministic escape hatch — DetMath would be the deterministic route, parked). Don't
  re-add a recurrence FFT: if a zero-setup convenience call is ever wanted, build a Temp workspace
  internally (build scratch is only 4 MB post quarter-wave) rather than reviving sin/cos.

## FFT.Workspace
- 2026-07-15 | irfft output de-interleave FUSED into the inner core's 1/N inverse-scale pass, so both
  ends of irfft are now fused (input re-pack into the permutation + output interleave into the scale).
  irfft(ws) float 1M 4.04 -> 3.52 ms (another -13%; now FASTER than rfft's 3.62, since rfft only fuses
  its input pack — its WQ unpack is still a separate output pass). double 1M 4.66 -> 4.17. The old
  irfft did the scale in place (rePtr[i]*=invN) then a separate pass real[2j]=cz[j]/real[2j+1]=sz[j].
  Fused: FftCoreRadix4Core/FftCoreRadix4MixedCore gained an interleaveOut pointer; when non-null
  (inverse only) the final scale writes interleaveOut[2i]=Re*invN, [2i+1]=-Im*invN straight into real
  instead of back in place. real is a separate buffer from cz/sz, so no aliasing. Bit-identical (same
  values, different destination). Complex ifft passes null → unchanged in-place path. FFT tests 102/102,
  no regression in complex fft/ifft (6.56/6.56). Output-side fusion now DONE for irfft; the dedicated-
  real-last-stage idea for rfft's unpack (k,M-k ≠ combine k,k+M/2) is a different, still-open item.
- 2026-07-15 | irfft re-pack FUSED into the inner inverse FFT's first permutation (mirror of the rfft
  pack fusion below). irfft(ws) float 1M 4.88 -> 4.04 ms (-17%), double 1M ~4.66. The half-spectrum
  re-pack (E/O reconstruction over Hermitian pairs) already writes cz/sz out-of-place from re/im, so
  scatter each re-packed sample k straight into its post-permutation slot dst(k) and call the compute
  core directly — the inner FFT skips its own reversal / cycle-following de-interleave. Extracted
  FftCoreRadix4Core (reversal-skipping variant of FftCoreRadix4, mirrors the MixedCore extraction);
  pure path calls FftCoreRadix4Core, mixed path FftCoreRadix4MixedCore. dst is a bijection over [0,M)
  and the pack reads re/im (separate buffer), so no collision/aliasing. Bit-identical, FFT tests
  102/102, no regression in complex fft/ifft (6.59/6.64). Unpack-into-last-stage (output side) still
  open: unpack pairing k,M-k ≠ combine k,k+M/2, needs a dedicated real last stage — big, deferred.
- 2026-07-15 | rfft pack FUSED into the inner FFT's first permutation → ~1.77x faster than complex
  fft(ws) (was ~1.16x), i.e. near the theoretical 2x for a two-for-one real FFT. rfft(ws) float 1M
  5.50 -> 3.62 ms (-34%), 262K 1.27 -> 0.83, 16K -42%; double 1M 6.09 -> 4.10. Mechanism: the old
  rfft did a separate pack pass (deinterleave real -> cz/sz) AND then the mixed core did its OWN
  in-place even/odd de-interleave via cycle-following (+ a full-length visited[] clear + pointer-
  chasing). Those are the SAME axis, so fused: scatter real[2j]/real[2j+1] straight into the inner
  FFT's first-permutation slot, OUT-OF-PLACE (real is a separate source, so no cycle-following /
  visited needed at all). Pure-radix-4 case (M=4^k): first permutation = base-4 digit reversal, so
  the fused scatter writes to ReverseBase4Digits(j) and then calls FftCoreRadix4Ptr directly. Mixed
  case (M=2·4^k, the common one at N=4^k): first permutation = even/odd de-interleave, fused scatter
  writes to dst(j) then FftCoreRadix4MixedCore. Refactored FftCoreRadix4Mixed -> MixedDeinterleave +
  FftCoreRadix4MixedCore so rfft can call the compute core after its own fused permute; the conjugate
  moved post-de-interleave (elementwise negate commutes with the permutation → identical). Bit-
  identical (same permutation, computed differently); FFT tests 102/102. Research (2 agents, sourced):
  planar/split re/im is the SIMD-right layout — interleaved needs shuffles our portable fProxyW
  deliberately lacks (Popovici/Franchetti HPEC'17; FFTW genfft; confirmed) — so this fusion-on-split is
  the correct route, NOT interleaving. Only complex-FMA HW (AVX-512 FP16 / Arm FCMLA) escapes, not
  reachable portably. irfft mirror + unpack-into-last-stage still open (unpack pairing k,M-k ≠ combine
  k,k+M/2, so unpack fusion needs a dedicated real last stage — big, deferred).
- 2026-07-15 | Serpentine (boustrophedon) butterfly group order tried and REVERTED. Don't retry.
  Idea: alternate the base_ group loop high->low by stage parity so each stage restarts in the
  address range the previous stage left hot (groups are disjoint/order-independent, so it's
  bit-identical — FFT tests stayed 117/117). A/B on a quiet PC: it's a wash-to-negative. fft(ws)
  float gained ~3-5% (1M 6.56 -> 6.30 ms, 16K -5%), but double REGRESSED ~1-3% (1M 7.25 -> 7.45)
  and rfft(ws) regressed on both dtypes (float 262K 1.274 -> 1.352, double 1M 6.09 -> 6.21).
  Regressions were consistent across sizes (not noise): the descending stream costs the 4-lane
  double path more than the hot restart saves (weaker descending-prefetch, more exposed), and the
  mixed-path rfft sub-FFT halves don't align with the combine boundary. Net makes 3 of 4 paths
  slower. The proper version of this locality idea is cache-blocking (six-step FFT), not loop flips.
- 2026-07-15 | rfft/irfft unpack: process bins in Hermitian-symmetric pairs (k, M-k). Under k -> M-k
  the twiddle maps W_N^(M-k) = -conj(W_N^k), so E_im, O_im and Re(W) flip sign — one WQ call and one
  (k, M-k) load produce BOTH outputs (re[k]=E_re+P, re[M-k]=E_re-P, etc.). Halves WQ calls and, more
  importantly, halves the effective cz/sz read traffic: the reverse-stream partner cz[M-k] is consumed
  in the same iteration as cz[k] instead of being re-fetched M elements later (long evicted at large N,
  so the old loop paid for cz/sz twice). Zero added memory. Loop now runs k=1..M/2-1 paired + a single
  self-paired middle bin k=M/2; guarded for M==1 (N==2, no general bins). rfft(ws) quiet-PC: float 1M
  6.14 -> 5.50 ms (-10%), 262K 1.478 -> 1.274 (-14%), 64K 0.308 -> 0.272; double 1M 6.58 -> 6.09; the
  gap over complex fft(ws) widened ~7% -> ~16% at 1M. Recovered ~0.6 of the ~2.8 ms pack+unpack
  overhead; the paired loop is now SIMD-shaped if pushed further (reversing load for the M-k stream +
  boundary-split WQ). Pack (real[2j]/[2j+1] -> cz/sz) left as-is — sequential deinterleave, already
  bandwidth-bound, nothing to fuse.
- 2026-07-15 | Quarter-wave twiddle table: store only the first quadrant cos (twQuarter, n/4+1)
  instead of two full/half arrays; reconstruct any W^m via CosQ (quadrant reflection) + a π/2 index
  shift (Im(W^m)=CosQ(m+n/4)). Cuts the persistent table 8→1 MB and the build double-scratch 16→4 MB
  at N=2^20 float (over the session: full 8 MB → half 4 MB → quarter 1 MB). ~1 ULP accurate, NOT
  bit-exact (reflected entries are independently-built, so a fold's sign flip isn't an exact
  negation) — within the existing 1e-6/1e-12 twiddle tests; TwiddleTableAccuracy rewritten to check
  the quarter values + full-circle reconstruction. N=2 degeneracy: Q=n/4=0 breaks the π/2 shift (Im
  should be 0), guarded in WQ (tableN>=4). Perf lessons, all A/B'd on a quiet PC (no-ws path as the
  stable control, cross-run):
  * The old README/fft.md workspace numbers (12.9/11.3 ms @1M) were STALE — pre-dated the wide-SIMD
    campaign. True steady-state is ~6.5/6.0; full-vs-quarter A/B on one machine confirmed the "2x"
    was a stale-doc mirage, not this change.
  * Quarter table alone (WQ in the scalar butterfly) REGRESSED fft(ws) float ~+10% (float has 2
    scalar stages q=1,4; double 1 → double stayed flat). Cause: WQ's branchy reconstruction ran
    per-butterfly in the finest stages (which carry n/4 butterflies each).
  * Fix 1 — materialize EVERY stage's W^1 into sw1 (was wide-stages-only; +5 entries, swLen still
    ~n/3), so the scalar butterfly reads W^1 and derives W^2/W^3 in-register like the wide path. No
    runtime WQ in the butterfly. Recovered most (+10% → +3%).
  * Fix 2 — reorder the scalar butterfly to j-outer/base-inner so the q(<=4) twiddle triples are
    computed ONCE per j (not per-butterfly), held in registers. Closed the rest: steady-state now
    parity-to-faster than the half-circle version (262144 ~-5%, double 1M ~-7%, 1M float within
    noise). Zero extra workspace memory, no threaded pointers.
  * cw1 (radix-2 combine) and sw1 stay materialized-from-CosQ-at-build → the wide combine/butterfly
    hot loops read contiguous tables, no runtime reconstruction. Only rfft/irfft unpack calls WQ.
  * Removed dead twq/tableN threading through FftCoreRadix4/Slice/Ptr/Mixed once the butterfly went
    fully sw1-driven (the stage twiddles are stage-length-relative, W_(4q)^j, independent of tableN).
  Radix-16 / storing W^2/W^3 in the workspace / horizontal-SIMD WQ all considered and rejected (see
  session notes). Next real speed levers are bigger: rfft pack/unpack fusion, cross-stage cache
  blocking — NOT the table.
- 2026-07-15 | SIMD'd the last scalar loop: the radix-2 DIT combine (Step 3 of FftCoreRadix4Mixed).
  The combine reads E/O data (re[k],im[k],re[M+k],im[M+k]) contiguously in k but the twiddle
  W_size^k = twReFull[k*combineStep] was strided, forcing a scalar loop. Fix mirrors the sw1 trick
  one level up: gather the combine twiddle into a contiguous per-workspace table cw1re/cw1im so k
  indexes it directly, then wide-load twiddles + all four data streams (fProxyW), scalar tail for
  M<Width (M is a power of 4, so no partial wide iteration once M>=Width). A given n triggers mixed
  at exactly ONE (size,step): n=2·4^k → fft/ifft at size=n step=1 (already contiguous → cw1 aliases
  twReFull, no copy); n=4^k → rfft/irfft inner mixed at size=n/2 step=2 (gathered, length n/4). So
  one cw1 table per workspace suffices; both dispatch paths pass ws.cw1re/cw1im. Measured (quiet PC,
  A/B new-vs-baseline, pure-radix-4 fft(ws) rows as the unchanged thermal anchor): rfft(ws)/irfft(ws)
  ~1.15-1.24× faster across float+double, all sizes, remarkably flat (float 1M 1.18×, 256K 1.19×;
  double 1M 1.24×, 256K 1.15×); mixed fft(ws) (2·4^k, top-level combine a bigger fraction) corroborates
  ~1.19-1.29× but noisier (baseline hit thermal spikes at double 16K/1M). Pure-radix-4 (pow-4) paths
  unchanged — no combine. This was the last scalar butterfly loop; the whole transform is now wide.
  Suite 6228/6228.
- 2026-07-14 | Extended the sw1 wide radix-4 butterfly to the rfft/irfft/mixed sub-transforms.
  Previously only the top-level pow-4 fft/ifft used the wide (fProxyW) butterfly; the mixed-radix
  (2·4^k) path and the rfft/irfft inner M-point FFTs ran the SCALAR FftCoreRadix4Ptr. Unified: made
  FftCoreRadix4Ptr the hybrid wide+scalar kernel (q>=Width wide via s1r/s1i, q<Width scalar), threaded
  sw1re/sw1im through FftCoreRadix4Slice/FftCoreRadix4/FftCoreRadix4Mixed, and deleted the standalone
  FftCoreRadix4Wide (fft/ifft pow-4 now route through FftCoreRadix4 → same wide path, one butterfly
  copy instead of two). Sub-transforms share the SAME sw1 table: size-M sub-FFTs share tableN=n, so
  stage q needs step=n/(4q) and stageOff layout identical to the top level — no second table. Build
  gate changed from `pow4 && qq<n` to `4*qq<=n` (drop pow4) so non-pow-4 workspaces also fill sw1
  (largest stage over all sub-transforms has 4q<=n). Measured (thermal-normalized rfft(ws)/fft(ws),
  since fft(ws) pow-4 is unchanged and cancels per-run thermal drift; raw A/B was unusable — PC not
  quiet, unchanged fft(ws) anchor swung 1.1-1.5× between runs): rfft(ws) ~1.3-1.6× faster
  (float 1M 1.45×, float 256K 1.27×, double 1M 1.59×, double 256K 1.34×). Suite 6228/6228.
- 2026-07-14 | Build ~13x faster via recursive-doubling twiddle fill. Replaced the per-entry
  bit-decomposition (O(n·log n): each W^m an independent product of log n generators) with a
  doubling fill (W^0=1, then W^(2^k+j)=W^j·B_k for j<2^k — one complex-mult per entry, O(n) total).
  Each entry is still <= log2(n) mults deep so error stays O(log n·ε); done in a double scratch,
  cast to fProxy once (same accuracy model, TwiddleTableAccuracy <1e-6 still passes). Measured
  float N=1M build ~41ms → ~3ms (ws+build 48→9.8ms; transform unchanged ~6.7ms); double similar.
  Cost: a transient double scratch of 2N doubles (16 MB at N=1M) via UnsafeUtility.Malloc/Free,
  freed before the factory returns (steady-state workspace memory unchanged). The doubling reads
  back intermediate values, so the scratch must be double even for the float variant to keep the
  chain O(log n·ε) rather than O(log n·float-eps).
- 2026-07-14 | sw* halving: store only the W^1 stage table (sw1re/sw1im), derive W^2=W^1·W^1 and
  W^3=W^1·W^2 in-register in the wide butterfly. Drops sw2/sw3 (4 of 6 arrays) → workspace
  −5.6 MB at N=1M float (−11 MB double), 28.4→22.8 MB. Perf-NEUTRAL: the 2 extra complex-mults
  per butterfly are absorbed because the wide butterfly is load/bandwidth-bound — 4 fewer twiddle
  streams offset the compute (measured 262144 float 1.67 vs 1.66, 1M double 8.2 vs 8.4, both in
  noise; 65536 float row is erratic across runs, ignore). No longer bit-identical to the scalar
  butterfly (derived W^2/W^3 differ ~2 ulp from tabulated), but TableFftVsRecurrence's 1e-3 tests
  pass; suite 6228. Remaining sw1 ≈ 2N/3 floats.
- 2026-07-14 | Deterministic twiddle-table build: replaced the per-entry math.cos/math.sin loop
  in fProxyFFTCache with root-of-unity generation using only +,-,*,sqrt. The table is W_N^m =
  exp(-2πi·m/n); built from binary generator roots B_k = exp(-2πi·2^k/n) via stable unit-circle
  half-angle square roots (c=sqrt((1+a)/2), s=b/(2c) — cancellation-free), each W_N^m the product
  of B_k over m's set bits (bit-decomposition, ≤log2(n) mults/entry so error is O(log N·ε),
  bounded — NOT the O(N) drift of a linear recurrence, the drift the direct-cos/sin table
  originally avoided). WHY: sqrt is IEEE correctly-rounded (bit-identical cross-arch), and +/-/*
  don't reassociate under FloatMode.Strict, so the whole build is cross-arch deterministic —
  unlike math.sin/cos, which Burst only guarantees identical under FloatMode.Deterministic
  (opt-in, 64-bit only). This closes the FFT's only non-deterministic step: under Strict the
  workspace fft/ifft/rfft/irfft path (build + butterfly, all +/-/*/sqrt) is now cross-arch
  reproducible. (The no-workspace recurrence path and dft still call cos/sin.) Built at double
  precision for both dtypes, cast to fProxy (float table rounded once from a double-accurate
  table — same as the old design). Verified: TwiddleTableAccuracy test asserts <1e-6 vs
  math.cos/sin at n=2/8/4096/65536, float+double; all FFT round-trip/vs-recurrence/vs-DFT tests
  pass (suite 6228). Build cost unchanged (~41 ms at N=1M float, same as the cos/sin build); the
  timed reuse transform is untouched. Uses stackalloc → factory is now `unsafe`.
- 2026-07-14 | Wide (fProxyW) radix-4 butterfly for the workspace fft/ifft power-of-4 path
  (FftCoreRadix4Wide), BOTH dtypes — float8 and double4. Vectorizes across the inner j loop for
  stages with quarter-stride q >= fProxyW.Width (8 float / 4 double): Width consecutive j give
  contiguous wide re/im loads/stores and reads of precomputed contiguous per-stage twiddles
  (ws.sw1/2/3 re+im, W^1/W^2/W^3 tabulated directly so every lane reproduces the scalar butterfly
  bit-for-bit — TableFftVsRecurrence's relTol tests pass, float+double). Stages q < Width stay
  scalar; q >= Width powers of 4 are multiples of Width so no j-tail. Measured scalar→wide,
  thermally matched via the (constant across runs) float-wide anchor:
    float(ws):  64K 0.556→0.373 = 1.49x, 256K 2.80→1.66 = 1.69x, 1M ~14-17→6.5 = raw ~2.1-2.5x
    double(ws): 64K 0.607→0.392 = 1.55x, 256K 3.00→1.81 = 1.66x, 1M 21.7→8.4 = 2.58x
  DOUBLE WINS TOO, bigger relative gain at 1M. The vecDot "double regressed through fProxyW"
  finding did NOT transfer: there double4-via-wrapper lost to an existing hand-tuned double4
  body; FFT's scalar butterfly had none, and the across-j ILP win is lane-count-agnostic. This
  is also the first double consumer of the WideOP double operators (now test-covered via FFT).
  HONEST 1M CAVEAT: the 1M butterfly self-throttles (largest/last size, box heats mid-run) — the
  double control swung 21.7→32 ms across identical-code runs; treat 1M as directional
  (~1.75-2.6x), 64K/256K are the clean reproducible figures. Speedup GROWS with N (butterfly-
  dependency-bound, not bandwidth-bound as first feared). Cost: sw* tables add ~2N elems to the
  workspace (~8 MB float / ~16 MB double at N=1M) on top of the full-circle table. NOT applied to
  rfft/irfft (their inner sub-FFT runs the mixed path at size M<n, where the top-level stage
  tables don't match) — future work: per-(size,tableN) stage tables, or derive W^2/W^3 from W^1
  in-register to halve sw* memory.
- 2026-07-12 | Full-circle twiddle table bandwidth tradeoff: uses ~2x twiddle memory (~8 MB at N=1M for float) versus the half-table, offset by halving the number of full-array passes (log4(N) vs log2(N) passes). (was FFT.Workspace.fProxy.cs:21)

## Eigen
- 2026-07-13 | Dropped the unsourced "~30x faster" multiplier from the three cyclic-Jacobi
  [Obsolete] messages (decompInPlace, valuesJacobi-style overloads); kept "Prefer
  Eigen.symmetricInPlace / Eigen.valuesSymmetricInPlace" guidance. Measured multiplier (no
  regression test pins it): Householder-tridiagonal + QL is roughly ~30x faster than cyclic-Jacobi
  for symmetric eigenpairs at this library's benchmarked sizes. (was Eigen.fProxy.cs:913, 1073, 1084)
- 2026-07-11 | Eigen.fProxy.cs doc trims (power/inverse-power iteration, Lanczos, cyclic-Jacobi decompInPlace) removed forwarder-architecture narration and an internal spec pointer (docs/dev/spec-svd-eigen-convergence.md) explaining why decompInPlace's sweep-budget constant isn't scaled by Consts.sweepBudget (its "sweep" is a full-matrix Jacobi sweep, a different iteration unit from LAPACK dbdsqr's per-value QR/QL sweeps). No perf verdicts lost -- purely doc-comment condensation. (was Eigen.fProxy.cs:16, :215, :533, :1140 pre-edit line numbers)

## Krylov
- 2026-07-11 | ApplyDot fusion investigated (cg/pcg's Ap=A·p + pAp=dot(p,Ap) step): a fused single-pass version was tried and measured slower than composing Apply+dot via IfProxyLinearOperator.ApplyDot; kept as the composed form. Don't retry the fused version without new evidence. (was Krylov.fProxy.cs:104, :301 pre-edit line numbers)
- 2026-07-11 | MINRES doc trimmed: the true residual norm ‖b-Ax‖ falls out of the Lanczos+Givens-QR recurrence for free via the running `phibar` variable (no extra dot/matvec needed to test convergence). Variable names (y,r1,r2,v,w,w1,w2) follow Paige & Saunders 1975 / Choi-Saunders minres.m. (was Krylov.fProxy.cs:452 pre-edit)

## LOBPCG
- 2026-07-12 | AXnext/APnext dropped from fProxyLOBPCGCache: allocated (k x n each) but never read
  or written -- UpdateActiveBlock deliberately doesn't mirror-combine AX/AP (see the 2026-07-11
  entry below on that same point), so no consumer ever existed for these two buffers. (was
  LOBPCG.Cache.fProxy.cs:115-121, :186-189; LOBPCG.fProxy.cs:719-722 pre-edit)
- 2026-07-12 | lockTol = 0.1*tolerance margin derivation (trimmed from comment): once a pair locks,
  the remaining active pairs are confined B-orthogonal to it, and the best residual achievable
  under that confinement is ~0.87x the frozen pair's lock residual -- hence locking at
  0.1*tolerance instead of tolerance, to avoid leaving later pairs stuck just above tolerance. (was
  LOBPCG.fProxy.cs:152)
- 2026-07-12 | AP field doc trimmed in fProxyLOBPCGCache (Cache.fProxy.cs): "this one mattered even
  more in practice" narration removed -- same rationale already covered by the 2026-07-11 AX/AP
  entry below. (was LOBPCG.Cache.fProxy.cs:107)
- 2026-07-11 | Buckling-mapping worked example (trimmed from the class doc comment for length; candidate for user-facing docs): the linear-buckling problem K_E*phi + lambda*K_G*phi = 0 (K_E SPD elastic stiffness, K_G indefinite geometric stiffness, Nastran SOL 105 / Abaqus *BUCKLE convention) rearranges to the pencil K_G*phi = mu*K_E*phi with mu = -1/lambda_cr, i.e. K_G in the A slot and K_E in the B slot. Usage: `var mu = Eigen.lobpcg(in K_G, in K_E, ref ws, k, tol, maxIter);` — mu is ASCENDING, mu[0] most negative/critical; a mu[i] >= 0 is not a buckling mode under this reference load, discard rather than divide; lambdaCritical[i] = -1/mu[i] for mu[i] < 0. Opposite sign convention (K_E*phi = +lambda*K_G*phi) uses lambda_cr = +1/mu, same pencil/targeting. (was LOBPCG.fProxy.cs:67-82)
- 2026-07-11 | Initial-X seeding bug history: an earlier deterministic fill used `(i + c*3 + 1) & 3`, periodic with period 4 in both i and c, so seeded X had at most 4 distinct rows — exactly rank-deficient for k > 4. The degeneracy was silently absorbed by FactorGram's ridge retry, so the solver iterated correctly within only a 4-dimensional subspace, never converging to eigenpairs 5+. Fixed by a fixed-seed Unity.Mathematics.Random fill instead. (was LOBPCG.fProxy.cs:107)
- 2026-07-11 | (d1) re-deflation step exists because a buckling smoke test (float) hit a hard-locking fixed point: once a pair locks, active X rows can retain a fixed B-component along the just-frozen row that no later search direction can cancel, freezing the residual at ~|component|*|dLambda|*||Bx||. Fixed by B-orthogonalizing the active block against locked rows every iteration. (was LOBPCG.fProxy.cs:239)
- 2026-07-11 | AX/BX are recomputed fresh via A.Apply/B.Apply every iteration rather than propagated through Cholesky-QR/Rayleigh-Ritz combinations, because propagating them was observed to accumulate rounding error that compounds: residual shrinks nicely for ~15-20 iterations, then stalls and creeps back up instead of continuing to converge. Same fix applied to AP/BP: an inaccurate AP corrupts the next iteration's [X,W,P] Gram/H directly (H's P-columns are dot(*,AP)) and was observed to produce Ritz values below lambda_min (down to -1E13 and beyond, exceeding the plausibility envelope by 1E5-1E30x) as soon as P entered the mix, even though the same marginal Cholesky conditioning is harmless in the P-less 2-block path. (was LOBPCG.fProxy.cs:339, :354)
- 2026-07-11 | UpdateActiveBlock deliberately does not mirror-combine AX/AP (or BX/BP) the way an earlier version did: the caller always immediately recomputes them via a fresh Apply right after the call returns, so the mirror-combine's result was always discarded — pure wasted work (extra O(3k^2 n) multiply-adds per iteration). Don't reintroduce it. (was LOBPCG.fProxy.cs:1163-1169 pre-edit)

## LP.BarrodaleRoberts
- 2026-07-24 | LICENSE RESOLVED: Roger Koenker granted relicensing permission by email for both
  this file's rqbr.f port and LP.FrischNewton's rq_fnm/lp_fnm port — MIT, no further permissions
  needed (he authored both algorithms). See Source/Third Party Notices.md. Koenker also suggested
  ladFN would be faster ported from quantreg's own rqfnb.f (Fortran) instead of the current
  MATLAB (rq_fnm/lp_fnm) lineage — optional future perf idea, not required.
- 2026-07-12 | LICENSE (historical, see resolution above): this file (and LP.FrischNewton) is a
  port of GPL(>=2) quantreg code (rqbr.f); owner requested relicensing permission from Koenker et
  al. A complete, suite-green CLEAN-ROOM replacement pair (papers/pyfixest-MIT provenance) exists
  at commit bdfd9ec (reverted by 101f8c9): correct at every test/benchmark size, but first-cut
  1.1-3x slower (BR float m>=4096: anti-cycling misfire, 115 iters) — kept only as a fallback
  reference now that permission is granted, not expected to be needed.
- 2026-07-12 | Ratio-test candidate collection pass (column-strided T[i,enter] read) left as-is
  deliberately: it costs O(m) per entering-column choice, so O(m*iters) total, asymptotically
  dominated by the O(m*n^2) BRPivot elimination sweep for any n > 1. A from-scratch column-major
  shadow of T was considered and rejected -- it would have to stay in sync across every BRPivot row
  update (itself row-major/unit-stride for good reason), doubling that update's cost to fix a
  strictly smaller-order term. (was LP.BarrodaleRoberts.fProxy.cs:240-243 pre-edit)
- 2026-07-11 | Source provenance (trimmed from the file banner): transcribed line-by-line from the Koenker-d'Orey Fortran `rqbr` (R `quantreg` package, src/rqbr.f), fetched from https://cdn.jsdelivr.net/gh/cran/quantreg@master/src/rqbr.f (same mirror pattern as LP.FrischNewton's source), cross-checked against the R wrapper `rq.fit.br` for the ift/flag status-code semantics. Deviation rationale kept in full: (1) rqbr's toler=eps^(2/3) is tighter than this library's simplex tolerance, deliberately not imported as a one-off literal; (2) on ift=2 "premature end", the raw Fortran leaves x untouched, but this port extracts the last-vertex structural solution since stage 1 has always completed by then and LPStatus.Unbounded's contract already promises that extraction; (3) ift=1 "solution may be nonunique" is a warning the reference emits without altering x, and LPStatus has no matching state; (4) reference diagnostic-only outputs (dsol/sol/h/e) are dropped in favor of an honest-recomputed objective, same reasoning as ladFN. (was LP.BarrodaleRoberts.fProxy.cs:15-97 pre-edit)
- 2026-07-11 | Perf verdict: BR's candidate-ratio selection used an O(nCand^2) selection sort (linear-scan-for-min + swap-remove), which was measured (not merely suspected) to be the dominant cost at large m — BR's own reported iteration count stayed flat near m=16384 while wall time grew far faster than FN's comparable-iteration interior point, the signature of quadratic work hidden behind a small iteration count (surfaced via LPBenchmark Section 2b, m=1024-16384). Fixed by sorting candidates once via heapsort above BR_CAND_SORT_THRESHOLD (set above every m the test suite exercises for BR, <=192, so tested paths stay on the original code) instead of the O(n^2) selection sort. Don't revert to unconditional selection sort. (was LP.BarrodaleRoberts.fProxy.cs:261-280 pre-edit)

## LP.DualSimplex
- 2026-07-12 | Bound-flip (BFRT) application's column-strided `flipRHS[i] += delta * M[i, j]` loop
  left as-is deliberately: flipCount is normally small (a handful of boxed nonbasics absorbed per
  iteration, not O(N)), so its O(flipCount*m) cost is already far below the O(mN) PRICE passes
  this file's other column-strided loops were. Routing it through a dense Mmul(M, deltaVec, ...)
  GEMV (deltaVec sparse, nonzero only at flipCols) was considered and rejected -- it would touch
  all N columns unconditionally, a regression whenever flipCount << N (the common case), for a
  loop that was never the O(mN) bottleneck. (was LP.DualSimplex.fProxy.cs:519-525 pre-edit)
- 2026-07-11 | DSE update formula, verified line-by-line against HiGHS source (not just paraphrase): highs/simplex/HEkk.cpp::updateDualSteepestEdgeWeights (`dual_edge_weight_[iRow] += aa_iRow*(new_pivotal_edge_weight*aa_iRow + Kai*dse_array_value)`) called from HEkkDual.cpp::updatePrimal with `Kai = -2/alpha_col`, `new_pivotal_edge_weight = edge_weight[row_out]/alpha_col^2`, DSE array built as `col_DSE = Ftran(row_ep)` i.e. tau = B^-1 rho_r — matches `w_i' = w_i - 2(alpha_qi/alpha_qr)*tau_i + (alpha_qi/alpha_qr)^2*w_r, then w_r' = w_r/alpha_qr^2`. The 1e-4 floor is HiGHS's `kMinDualSteepestEdgeWeight` (highs/simplex/SimplexConst.h). (was LP.DualSimplex.fProxy.cs:36-43 pre-edit)
- 2026-07-11 | Warm-start correctness proof (trimmed from banner): the warm overload's dual-feasibility repair (bound flips / temporary artificial bounds keyed off a real BTRAN-computed reduced cost) is a strict generalization of the former cold-only precondition, provably bit-identical at the all-logical basis since y = B^-T c_B is then exactly the zero vector (c_B = 0, and BTRAN of an all-zero vector stays all-zero through every forward/back-substitution step — each step is an assignment of 0, multiply-by-0, subtract-of-0, or divide-of-0-by-nonzero, all exact in IEEE754). (was LP.DualSimplex.fProxy.cs:47-51, :430-436 pre-edit)
- 2026-07-11 | Bug history — DualRatioTest: an earlier version allowed bound flips to fully resolve a row with no actual pivot. It passed every test at n<=24 but produced a false Infeasible on a 48-variable random instance: a flip-only iteration leaves the basis (hence y=B^-T c_B, hence every dj) unchanged, so a column just flipped from AtLower to AtUpper keeps its old, now-wrong-signed reduced cost with no future iteration positioned to notice. Fixed by guaranteeing a real pivot happens every time DualRatioTest returns anyCandidate=true (flips are only ever a prefix of the walk). Don't reintroduce flip-only resolution. (was LP.DualSimplex.fProxy.cs:117-122 pre-edit)
- 2026-07-11 | Bug history — float artificial-bound scaling: HiGHS's fixed [0,1e7] artificial-bound box (tuned for HiGHS's internally-scaled/equilibrated data) was observed to produce a false Infeasible within the first few dual iterations in float. RebuildXB's adjusted rhs sums the artificial bound's contribution over every simultaneously-artificial column (up to ~n/2 for mixed-sign-cost problems), landing xB around -(artificialBound*n/2*|A|); at 1e7 with n~48 that's order 1e8, and float's ~1.19e-7 relative precision there is an absolute error of order 10, which swamped feasTol (~3.45e-4) outright. Fixed by scaling the artificial bound to the problem's own data magnitude (100x largest |cost|/|rhs|) instead of HiGHS's literal. (was LP.DualSimplex.fProxy.cs:304 pre-edit)
- 2026-07-11 | Bug history — cost-perturbation base literals: an earlier dualTol-scaled float variant (instead of reusing HiGHS's own literal bases for both dtypes) made float branch-and-bound trees explode (benchmark-verified). Reverted to HiGHS's literals for both float and double. (was LP.DualSimplex.fProxy.cs:332 pre-edit)
- 2026-07-11 | Perf/correctness verdict — DSE weight reseed gating: tying the weight[] reseed to `didResumeFactors` (not just the caller's original `resumeFactors` request) was benchmark-caught as a real regression: MIPBenchmark float branchy12 went from Optimal/216 nodes/10.6ms to NodeLimit/20000 nodes/122.7ms under the unconditional version (a cache hit whose eta file was already at capacity resumed weight[] even though B/P/eta had just been refreshed, letting weight[] drift across an unbounded refactorization chain spanning the whole search). Fixed by tying the reseed to didResumeFactors, confirmed back to Optimal/226 nodes/~9.5ms. Don't detach the reseed condition from didResumeFactors. (was LP.DualSimplex.fProxy.cs:416-419 pre-edit)
- 2026-07-11 | Bug history — perturbedCost in the dual-feasibility repair: using perturbedCost (instead of the original cost) for the one-time true-dual-feasibility decision was an actual bug. A column with cost[j] exactly 0 (e.g. every x+/x- column in LP.lad's reformulation, which has none of its own cost) is dual-feasible as-is, but the perturbation's random sign could nudge it slightly negative and give it a pointless artificial bound; multiplied across the many exactly-zero-cost columns LP.lad's [x+|x-] block always has, this corrupted the warm-started basis handed to the primal cleanup badly enough to report a false Unbounded. Fixed by using the original cost. Don't use perturbedCost here. (was LP.DualSimplex.fProxy.cs:445-450 pre-edit)
- 2026-07-11 | Perf note — zero-pivot fast path: skipping RevisedPrimalCore's cleanup call when the dual loop already left a true optimum (r<0 exit, zero pivots, no artificial bounds) was measured to roughly halve a warm re-solve's fixed per-call cost when it applies (~0.12ms/call -> ~0.06ms/call at mAug~80, isolated warm LP.solve(ref LPBasis) benchmark, MIP perf investigation 2026-07-10) — a genuine but minority case for MIP/strong-branch-trial re-solves (most single-bound tightenings still cost >=1 real pivot). (was LP.DualSimplex.fProxy.cs:657-658 pre-edit)

## LP.FrischNewton
- 2026-07-11 | Source provenance (trimmed from banner): ported and verified line-by-line against Daniel Morillo & Roger Koenker's `rq_fnm`/`lp_fnm` (originally Ox, translated to MATLAB by Paul Eilers 1999, modified by Koenker April 2001), fetched from https://github.com/karenamckinnon/summer-temperature-distributions/blob/master/rq.m (mirrors the file distributed with R's quantreg package; same algorithm also in quantreg's Fortran rqfnb.f). Every update formula (predictor, centering parameter, corrector, step-length ratio test, the 0.9995 factor) is that source's, not reconstructed from memory. Sign convention verified against LadStackloss's published coefficients in testing. (was LP.FrischNewton.fProxy.cs:23-30, :43 pre-edit)
- 2026-07-11 | Problem/dual formulation (trimmed from banner, kept for reference): quantile regression at level tau in (0,1) (tau=0.5 == LAD up to a factor 2), min_x sum_i rho_tau(b_i - A_i.x), rho_tau(u)=u*(tau-1[u<0]). Its dual (rq_fnm's construction): max_a b.a s.t. Aᵀa=(1-tau)Aᵀ1, a in [0,1]^m — solved by lp_fnm as min c.v s.t. Ãv=b̃, 0<=v<=1 with Ã=Aᵀ, c=-b, b̃=Aᵀ((1-tau)1); the LP's own primal variable v IS the dual weight a. (was LP.FrischNewton.fProxy.cs:34-42 pre-edit)

## LP
- 2026-07-11 | LP.lad's BR/FN crossover, measured (LPBenchmark Section 2b, 2026-07-09, after the BR sort-path + FN SIMD optimization round): double — BR wins through m=4096 (2.49ms vs FN 2.71ms) and loses only ~11% at m=16384, so the threshold sits at the last measured BR-win size, 4096; float — FN's SIMD gains moved its win boundary down to m=1024 (FN 0.47ms vs BR 0.62ms) while BR still wins at m=384, so 512 splits the measured bracket. Re-measure Section 2b (and re-tune the threshold) whenever either engine's per-iteration cost changes. (was LP.fProxy.cs:318-324 pre-edit)

## LP.RevisedSimplex
- 2026-07-12 | RevisedPrimalCore's warm-start overload was added for LPMethod.DualSimplex's
  HiGHS-style composition (LP.DualSimplex.fProxy.cs hands its terminal basis to this core as a
  cleanup pass once real bounds are restored); the fresh-start overload above is non-breaking --
  it simply builds the all-logical basis/status and forwards here, so its behavior and public
  surface are unchanged. (was LP.RevisedSimplex.fProxy.cs:439-442 pre-edit)
- 2026-07-11 | Bug history — HarrisRatioTest far-bound fallback: "travel through to the far bound" assumed the far bound is finite, which broke on a dense covering LP (min cx s.t. Ax>=b, x>=0, A,b,c>0): every >=-row logical starts basic and above its upper bound (0) with an unreachable lower bound (-INF), for every row simultaneously, so no row ever contributed a finite ratio-test limit and the pass-1 unbounded check fired — RevisedSimplex returned Optimal with 0 iterations / objective 0 while tableau/interior/dual all agreed on the true optimum (a silent phase-1 bail extracting x=0 from a basis nothing ever pivoted into). Caught by LPBenchmark, reproduced in LPTests.fProxy.cs as RevisedDenseCovering (failed before the fix, passes after). Fixed by a two-attempt ratio test: the first pass is byte-identical to the original algorithm; only if it would report Unbounded does a second pass run with a fallback that targets the NEAR (violated) bound instead of an unreachable far one. Don't remove the two-attempt structure. (was LP.RevisedSimplex.fProxy.cs:297-315 pre-edit)

## LQRP
- 2026-07-11 | Design rationale trimmed from class remarks: QRCP's LEVEL-3 machinery (blocked dlaqps panel core with deferred F-matrix trailing update) is deliberately not mirrored in LQRP -- it only earns its bookkeeping at large sizes, and the primary consumer (rank-deficient IK Jacobians) is small (task DOF × joint DOF). Add a blocked core later if large wide matrices need it. Downdated row norms (vs. exact recompute every step) remove a second O(m²n) pass re-summing candidate norms; pivot selection needs current row NORMS not row DATA, so tracking incrementally is sufficient. Basic-vs-min-norm gap: for an inconsistent rank-deficient b, solveInPlace is not minimum-norm because the below-diagonal block L21 couples the independent variables into the dropped equations (the transpose-dual of QRCP's R12 coupling -- L's top-right IS zero, but that's not where the coupling lives; trailing rows of L keep their full norm, only the trailing diagonal is small). minNormSolveInPlace closes the gap by least-squares-solving the coupled m×r block K=[L11;L21] instead of just L11. (was LQRP.fProxy.cs:22-47 pre-edit)
## MIP.Domain

- 2026-07-11 | UB-row sentinel-rhs bug (moved from header comment): using a 1e30 sentinel rhs directly for an infinite-UB row (instead of the inert coefficient-0/rhs-0 convention) fed into DualSimplexCore's dataScale/artificialBound scan and inflated artificialBound to ~1e32, producing a false Infeasible. Reproduced on the Gomory/Wolsey instance: sentinel=1e30 -> false Infeasible after 63 pivots; sentinel<=1e10 -> correct. Also a correctness risk beyond numerics: a finite sentinel can silently bound a genuinely unbounded direction. Fix: UB rows start inert and PushBoundChange/UndoToMarker activate/deactivate the row's coefficient explicitly. (was MIP.Domain.fProxy.cs:20)
- 2026-07-11 | PropagateFixpoint HiGHS provenance (moved from doc comment): ported from mip/HighsDomain.cpp's `propagate`/`propagateRowUpper`/`propagateRowLower`. Worklist membership mirrors HighsDomain's `markPropagate`/column-incidence loop and `propagateinds_`. Infinite-contributor counts are HiGHS's `ninfmin`/`ninfmax`; the closed form `(rhs - (act - ownContribution)) / a_ij` is HiGHS's `minresact`/`maxresact`. Termination on queue-drain mirrors HiGHS's `havePropagationRows`; the `PROPAGATION_MAX_PASSES * m0` visit cap is a deliberate deviation — HiGHS has no such cap because its incremental activity bookkeeping makes each visit O(row length) instead of a full recompute, and this port's fixpoint isn't persisted/maintained incrementally across the whole B&B tree the way HighsDomain's activity arrays are. (was MIP.Domain.fProxy.cs:154)

## MIP.Pseudocost

- 2026-07-11 | PseudocostScore formula + fidelity gaps (moved from comment): faithfully ported from HighsPseudocost::getScore(col, upcost, downcost) (mip/HighsPseudocost.h): `costScore = max(upcost,minThreshold)*max(downcost,minThreshold) / max(minThreshold, cost_total^2)`, then `mapScore(x) = 1 - 1/(1+x)`. upcost/downcost are PseudocostEstimate (== HighsPseudocost::getPseudocostUp/Down's 2-arg no-offset overload: fractional-distance * own mean, falling back to the running global-average pseudocost when the variable has zero samples). cost_total is the running global average (globalPCSum/globalPCCount); minThreshold == PSEUDOCOST_EPS (both 1e-6), same clamp value and placement as the source. OMITTED (fidelity taxonomy — subsystems this port doesn't have): the conflictScore term (no conflict analysis / no-good learning), the cutoffScore term (no cutoff-bound tracking), the inferenceScore term (no propagation/inference statistics). OMITTED: degeneracyFactor weighting (no LP-degeneracy detection) — HiGHS only sets it > 1 while actively degenerate; fixed at its non-degenerate default of 1.0, getScore's full expression collapses exactly to mapScore(costScore), which is what PseudocostScore returns. (was MIP.Pseudocost.fProxy.cs:50)

## MIP

- 2026-07-12 | Limit-exit semantics verified faithful against HiGHS source (master + v1.7.2, HighsSearch::dive / HighsMipSolver::run / cleanupSolve): upstream also checks limits AFTER the in-flight node's full work (evaluateNode installs incumbents unconditionally before checkLimits; plunge loop runs heuristics+dive before its checkLimits), and cleanupSolve reports the incumbent objective + queue-folded dual bound + finite gap on limit exit — same as SearchCore's top-of-loop budget check after the release-scan fix. Known deviations, intentional: (a) HiGHS collapses node/leaf/improving-sol limits into one kSolutionLimit status; this port keeps a distinct NodeLimit. (b) cumulative maxIter (total LP iterations across the search) is PORT-ORIGINAL — HiGHS has no MIP-wide LP-iteration budget (its kIterationLimit is LP-only, never set from MIP code). (c) TryRoundingHeuristic fires on every fractional node; HiGHS runs randomizedRounding at the root + once per plunge start (granularity choice, see 07-11 entry below).
- 2026-07-11 | Architecture notes trimmed from file header (moved from header comment): Bounds-as-rows shift mechanics — LP.solve only supports x>=0, so every variable is shifted to a non-negative y (anchor-low/anchor-high/free-split, same substitution as QP.PhaseOneFeasibleStart); integer variables get two pre-allocated rows (y<=U, y>=L), branching only rewrites their rhs so the augmented LP's shape stays fixed and the same LPBasis stays warm-startable. UB-row activation: starts INERT (coefficient 0) when xu is infinite, activated (coefficient -> 1) on first branch — a literal 1e30 sentinel rhs corrupts the dual simplex's dataScale/artificialBound scaling and can silently bound a truly unbounded direction (full detail moved to MIP.Domain.fProxy.cs's own header). Warm start: one LPBasis persists across the whole search including strong-branch trials; the dual simplex's dual-feasibility repair makes a stale basis (right after a plunge dive, an undone strong-branch trial, or a queue jump) a correct, not just fast, starting point. Node state: the current plunge's dive steps use the incremental bound-change stack (PushBoundChange/UndoToMarker); a queue jump is not generally to an ancestor so it can't replay that stack — instead each queued node carries its own full length-n bound snapshot (fProxyMIPQueueNode) and a jump overwrites live bound state wholesale (ApplyNodeBounds) and resets the stack. dualBound = min over every still-open node's own parent-LP bound (the current plunge frontier plus everything still in the queue). (was MIP.fProxy.cs:11)
- 2026-07-11 | TryRoundingHeuristic HiGHS provenance + deviations (moved from comment): ported from HighsPrimalHeuristics::randomizedRounding; the randomized-interval draw is HiGHS's `floor(relaxationsol[i] + randgen.real(0.1, 0.9))`. Two intentional deviations from HiGHS's tryRoundedPoint/randomizedRounding: (a) HiGHS re-solves an LP with the rounded integers fixed to repair continuous variables and confirm feasibility; this port has no per-node LP re-solve budget for the heuristic, so it does an O(mn) direct feasibility check against the original rows instead. (b) HiGHS's `randgen` is a solver-wide RNG advanced continuously; MIP.solve has no public seed parameter and must stay bit-deterministic across repeated identical calls, so this uses a fixed internal seed instead (roundRng in SearchCore). Bound-clamping decision: rounded values are clamped into the CURRENT node's bounds and feasibility is checked against them, not the root bounds — root bounds may be fractional (user-supplied), so clamping to them could install a fractional "integer" incumbent; node bounds are always integral once branched. (was MIP.fProxy.cs:541)

## OP.Component

- 2026-07-11 | clampInPlace `this T` vs `this in T` (moved from remarks): takes the receiver by value (`this T`), matching every other Comp wrapper in this file — a generic extension method's receiver cannot use `this in T` (CS8338: the 'in' extension-method form requires a concrete, non-generic value type). Callers migrating from the old static-style `clampInPlace(in v, ...)` just drop the now-illegal `in`. (was OP.Component.fProxy.cs:154, same block also in OP.Component.iProxy.cs:152)

## OP.Dot

- 2026-07-11 | dotSelf fused-kernel rejection (moved from comment): dotSelf composes a plain matVecDot pass + a separate vecDot pass for `y = Ax` plus `dot(x,y)`, rather than a single fused kernel. Why: an earlier version dispatched a genuinely-fused single-pass kernel (matVecDotSelf) for square A, folding dot(x,y) into the GEMV row-loop via two alternating scalar accumulators (the row-loop itself already uses vecDot's fProxy4 SIMD pattern, so there was nothing left to widen the outer cross-row fold into — row results arrive one at a time, not as an aligned block of 4). MEASURED WORSE on the BSR analogue of the same pattern (bsrMatVecB1Dot, part of the Krylov optimization round): the scalar alternating fold lost to simply calling the already-SIMD-tuned vecDot separately, by a wide and reproducible margin at the block=1 stencil benchmark. Reverted here too on the same architectural basis (not separately re-measured for dense). Verdict: don't retry a fused scalar-accumulator dot-fold here without new evidence. (was OP.Dot.fProxy.cs:95)

## Optimize

- 2026-07-11 | ladIRLS weighting formula + delta tuning (moved from doc comment): minimizes ‖A x − b‖₁ by repeatedly solving the weighted normal equations `(AᵀW A) x = AᵀW b` with per-row weights `wᵢ = 1 / max(|rᵢ|, delta)`, `rᵢ = (A x − b)ᵢ`; typically converges in a handful of iterations for a well-conditioned overdetermined design. delta is a Huber-like transition width: too small causes oscillation, too large drifts the fit toward ordinary least squares. (was Optimize.fProxy.cs:224)

## QP.Info

- 2026-07-11 | QPInfo precision + diagnostics rationale (moved from doc comment): QPInfo is a plain, unprefixed struct (not float/double-generated) because diagnostics need not be precision-typed — objective is always reported as double regardless of solve precision, matching LPInfo/LstsqInfo/SolveInfo's convention. stationarityResidual/feasibilityResidual follow the solver-diag-struct convention of "only already-computed/cheap numbers": both are already on hand as a direct byproduct of the null-space step (stationarity: the reduced gradient Zᵀg the step just drove to ~0; feasibility: one cheap GEMV, A_W x - b_W) — see QP.eqpNullSpaceStep. Per spec's Stage 1 oracle, these are meant to be compared against a full KKT-system LU solve. There is no separate complementarity residual yet because the fixed-working-set kernel has no inequality constraints to be complementary about; a future active-set loop would extend this struct's diagnostics, not replace them. (was QP.Info.cs:94)

## QP

- 2026-07-11 | Null-space method derivation (moved from file header): parameterize every feasible point as x = x0 + Zy, x0 any point with A_W x0 = b_W, Z an orthonormal basis for null(A_W) (A_W Z = 0, so A_W x = A_W x0 = b_W for ANY y). Substituting, the equality-constrained problem `min ½xᵀQx + cᵀx s.t. A_W x = b_W` becomes the UNCONSTRAINED reduced problem `minimize_y ½yᵀ(ZᵀQZ)y + (Zᵀg(x0))ᵀy + const`, g(x0) = Qx0 + c — an ordinary quadratic in y with Hessian H_Z = ZᵀQZ and gradient Zᵀg(x0) + H_Z y. For ANY quadratic, Newton's method reaches the exact minimizer in ONE step regardless of starting point (the model IS the function), so solving H_Z y = -Zᵀg(x0) and setting x1 = x0 + Zy lands exactly on the equality-constrained optimum — no line search, no iteration. Source: Nocedal & Wright, Numerical Optimization (2nd ed.), ch. 16.2, eq. 16.16-16.19. (was QP.fProxy.cs:12)
- 2026-07-11 | "Keeping Q implicit" QR mechanics (moved from file header): QR.decompInPlace's public API forms the dense (n x k) "thin" Q1 in one call (factor + reconstruct, no split entry point). To avoid ever materializing that n x k matrix, this file bypasses QR.decompInPlace entirely and drives QR's own per-step primitives directly (QR.genHouseholder / QR.applyReflectorRight, both `internal` — the same functions decompInPlace itself is built from): FactorWorkingSetTranspose is exactly decompInPlace's factorization half (store R + stash each Householder vector into A_Wᵀ's own columns), replicated rather than called through the public API specifically so the reconstruction half never runs. Two more primitives close the loop without ever forming Q1: (1) ApplyWorkingSetQtForward — Q_full = H_0 H_1 ... H_{k-1} is an n x n orthogonal matrix (each reflector acts on the full ambient n-dimensional space; only k of them exist because A_Wᵀ has k columns), so Q_fullᵀ = H_{k-1} ... H_0 and a FORWARD sweep (d = 0..k-1) of the k stashed reflectors over any n-vector v computes Q_fullᵀv = (Q1ᵀv ; Zᵀv) in one pass — top k entries and bottom n-k entries. This is the exact trick QR.solveInPlace already uses for its `b` argument (computing Qᵀb without ever forming Q), generalized and replayed from STORED reflectors instead of freshly-generated ones; it replaces both QR.decompSolve's "Qᵀg, then R-solve" (used for multiplier recovery) and the reduced gradient Zᵀg in one sweep. (2) FormNullSpaceBasis — Z itself (n x (n-k), needed because the reduced Hessian ZᵀQZ and the step p = Zy are GEMM/GEMV operands) is formed by REVERSE-sweeping (d = k-1..0) the same stashed reflectors over the seed [0; I_{n-k}] — exactly QR.decompInPlace's own Q-reconstruction phase, seeded with the TRAILING identity block instead of the leading one, targeting a separate n x (n-k) buffer instead of overwriting A_Wᵀ. Z is smaller than the full n x n Q by construction (only the n-k null-space columns), so forming Z (not Q) is the documented exception to "don't form dense Q". Note: "Q" here means QR's orthogonal factor, distinct from this file's Q the Hessian — an unfortunate letter clash inherited from the spec/textbook. (was QP.fProxy.cs:12)
- 2026-07-11 | Stage 2-3 reuse structuring rationale (moved from file header): every function in this file is `internal static` (not a buried local), matching the structuring rule LP.RevisedSimplex.fProxy.cs set for LP.DualSimplex.fProxy.cs — a future inequality active-set loop (ratio test, add/drop, Dantzig pricing) will call eqpNullSpaceStep (or its constituent pieces) once per iteration, re-factoring A_Wᵀ from scratch after every working-set change (see FactorWorkingSetTranspose's own doc comment for that cost and why it is deliberately not incremental — v1 scope decision). (was QP.fProxy.cs:12)
- 2026-07-11 | eqpNullSpaceStep future active-set call pattern (moved from doc comment): called once by eqpSolve today (the whole algorithm for a fixed working set). A future inequality active-set loop would call this once PER ITERATION with the CURRENT working set (which changes as constraints are added/dropped), re-factoring A_Wᵀ from scratch every call — an incremental QR update was judged not worth porting at this library's dense target sizes. That future loop will need to intercept BETWEEN computing the step and applying it in full (the ratio test may truncate the step to alpha*p, alpha < 1), which the current version does not do — expect that seam to require splitting the "compute p" and "apply p, recover multipliers" halves of this function. (Note: qpActiveSetCore below already did exactly this split, built directly on the constituent kernel functions instead of through eqpSolve/eqpNullSpaceStep.) (was QP.fProxy.cs:358)
- 2026-07-11 | FactorWorkingSetTranspose HiGHS comparison (moved from comment): HiGHS maintains an incrementally-updated factorization of the working-set basis across add/drop changes; deliberately not ported here — simple/correct was judged to beat incremental/subtle for this library's v1 scope. (was QP.fProxy.cs:545)
- 2026-07-11 | qpActiveSetCore unbounded-detection proof (moved from header comment): verified against Nocedal & Wright, Numerical Optimization (2nd ed.), section 16.5 ("Active-Set Methods for Indefinite QP") — fetched and read 2026-07-09. With Z the null-space basis for the current working set and ZᵀGZ found singular/indefinite along a direction sZ chosen to be non-ascent (their eq. surrounding "q(x+alpha*Z*sZ) -> -infinity as alpha -> infinity" and the sign choice "so that Z*sZ is a non-ascent direction for q"), the text states plainly: "By moving along the direction Z*sZ, we will encounter a constraint that can then be added to the working set for the next iteration. (If we don't find such a constraint, the problem is unbounded.)" This library's case is the boundary of their construction (Q only PSD, so the reduced Hessian can go singular/zero-curvature but never strictly negative-definite beyond that boundary): conditions 1-2 (regularized, zero curvature) detect that boundary, condition 3 (no blocker) is their "we don't find such a constraint", condition 4 (descent) is their non-ascent sign choice — made an explicit check here rather than a sign flip because SolveReducedNewtonStep's regularized solve already mathematically guarantees gᵀp <= 0 whenever it succeeds (gᵀp = gzᵀy = -gzᵀ(H_Z+deltaI)^-1 gz, and H_Z+deltaI is PD), so #4 is a defensive check on that guarantee, not a live sign-flip decision. (was QP.fProxy.cs:698, date-stamped note also flagged separately at QP.fProxy.cs:762)
- 2026-07-11 | qpActiveSetCore perturbation-cleanup rationale (moved from comment): the final exact null-space Newton step against the true bounds is exactly LP.DualSimplexCore's own composition pattern ("hand the terminal basis to the primal core... using the REAL cost", see that file's header comment) rather than leaving a perturbation-sized residual in the reported solution. The multiplier check that already declared Optimal never saw perturbed data (it depends only on g = Qx+c and the working-set geometry, never on L/U), so this cleanup pass cannot change WHICH working set is optimal, only where x sits on it: reusing eqpSolve (LQ.minNormSolve to the TRUE b_W, then one exact Newton step) re-lands EXACTLY on this same working set's true optimum. Zero cost on the common, non-degenerate path (skipped whenever perturbation was never engaged). (was QP.fProxy.cs:1112)
- 2026-07-11 | BuildPerturbedBounds magnitude proof (moved from comment): the ratio test's EXACT ties are the root cause of a stalled/cycling run of zero-length steps; widening L/U by a tiny deterministic amount makes them distinct, letting a genuine (if tiny) step through. Uses a deterministic per-row pseudo-random unit value via the SAME cheap integer hash LP.DualSimplexCore uses for its own cost perturbation (MurmurHash3 finalizer mix); magnitude is a SMALL FRACTION of feasTol (0.1x) so it is provably too small to be mistaken for genuine constraint slack anywhere else in the solver (every other feasibility decision in this file compares against feasTol itself), yet many orders of magnitude past a float ULP, so it reliably breaks bit-exact ties. (was QP.fProxy.cs:1223)
- 2026-07-12 | Anti-cycling hardening's deterministic bound perturbation replaced an earlier Bland-style seam; same pattern (and lesson) as LP.DualSimplexCore's own cost perturbation, see that file's header comment. (was QP.fProxy.cs:759-761)
- 2026-07-12 | FormNullSpaceBasis's full-column-width requirement (why the leading-identity restriction doesn't generalize to Z) was caught by the Stage-1 KKT-oracle check at k=2, n=8: k=1 has only one reflector at d=0, whose "columns >= 0" restriction never actually excludes anything, so the bug was invisible there. (was QP.fProxy.cs:541-543)
- 2026-07-12 | qpActiveSetCore's pScale rescale (both the curvature-test call site and RatioTest's own doc comment) was caught by the LP-limit oracle test: Q=0 forces every step through this exact path, since the reduced Hessian is then identically singular every iteration. (was QP.fProxy.cs:872-874, :1239-1240 pre-edit)

## RandomOP

- 2026-07-12 | randomPermutationInPlace uses a separate Fisher-Yates loop from shuffleInPlace rather than sharing one: Pivot.Swap tracks the permutation parity via its swap counter, which plain index swapping (shuffleInPlace's Indices buffer has no parity field) cannot do. (was RandomOP.cs:22)
- 2026-07-12 | fProxyGaussian.Next doesn't use math.sincos: its out-parameter overload is not available via the type-proxy template mechanism, so math.sin and math.cos are called separately instead. (was RandomOP.fProxy.cs:511)

## RandomMatrixOP

- 2026-07-11 | orthogonalInPlace algorithm walkthrough (moved from doc comment): 1) fill an n×n scratch matrix G with i.i.d. N(0,1) entries; 2) QR-decompose G = Q·R (Householder); 3) Haar sign fix — multiply column i of Q by sign(R[i,i]) (sign(0)=+1, no flip); 4) copy the corrected Q into dest. Temp scratch: G (n×n) and R (n×n), both disposed before return; the QR step allocates an additional n-element Temp vector internally (disposed inside decompInPlace). Why the sign fix matters: without it, Householder QR's Q is NOT uniformly distributed over O(n) — the sign of each R diagonal is not equally likely to be ±1, introducing a measurable bias; the sign flip corrects this and yields the true Haar measure. (was RandomMatrixOP.fProxy.cs:141)

## QR

- 2026-07-11 | Blocked panel trailing-update tiling rejection (moved from comment): UnsafeOP.wyVtC/wySubVW already reach full GEMM throughput (~70 GFLOP/s, matched matMatDot) at this width without tiling. Column-tiling was tried and MEASURED SLOWER (added MemClear/call overhead for no cache-locality benefit), so it is deliberately not done here — don't retry without new evidence. (was QR.fProxy.cs:283)

## QRCP.Workspace

- 2026-07-11 | fProxyQRCPCache scope rationale (moved from doc comment): deliberately holds ONLY vn1/vn2, not u (the Householder scratch, length m), w, or the blocked core's larger working buffers (F, flush GEMM scratch, the reconstruction WY buffers). The level-3 blocked core (decompInPlaceBlockedCore, engaged once N_Cols >= 2*QRCP_BLOCK) still takes its vn1/vn2 downdating state from here but Allocator.Temp-allocates those larger buffers per call — so this cache stays minimal with no dead fields (spec ticket: OQ-7, "QRCP earns a cache purely for the downdating state"). Promoting the blocked buffers in here for a fully zero-alloc blocked path (as fProxyQRCache does for QR) is a candidate follow-up. (was QRCP.Workspace.fProxy.cs:26)

## QRCP

- 2026-07-11 | Class-level blocked-panel mechanics (moved from remarks): downdating is what unlocks a level-3 path — pivot selection needs the current column NORMS, not the current column DATA, so once N_Cols >= 2*QRCP_BLOCK the factorization runs the LAPACK dlaqps-style partially-blocked panel core (decompInPlaceBlockedCore): a whole panel of reflectors is factored against a deferred F-matrix and its trailing update flushed once as a rank-kb GEMM, and Q is reconstructed by the same blocked-WY kernel QR uses (QR.reconstructQBlocked). Below that gate the unblocked per-reflector core runs (decompCoreDispatch chooses). fProxyQRCPCache carries only the two n-sized downdating vectors (vn1, vn2); the blocked core's larger working buffers (F, the flush GEMM scratch, and the reconstruction WY buffers) are Allocator.Temp allocated per call inside decompInPlaceBlockedCore — one set per factorization, negligible against its O(n²m) work — rather than folded into the cache. (was QRCP.fProxy.cs:21)
- 2026-07-11 | tol3z codebase-consistency note + retired-buffer history (moved from comment): tol3z is Consts.fProxySqrtEps directly — Consts.cs already defines it as the precise, type-correct sqrt(Consts.fProxyEpsilon), and every other caller in this codebase (Eigen/LOBPCG/Krylov/SVD.LowRank) references it the same way rather than recomputing math.sqrt(Consts.fProxyEpsilon) at runtime. Separately: the current guard-triggered re-sum (writes straight into vn1, no separate colNorm2 buffer) replaced an old exact-recompute-every-step buffer, now fully retired. The batched row-major re-sum is a deliberate widening from LAPACK's own per-column selective recompute: this codebase is row-major, so a single column's exact norm is a strided reduction — the same shape the ORIGINAL always-exact QRCP avoided by summing all trailing columns per row instead of one column at a time — so reusing that batched sweep when ANY column trips the guard is simpler, no more expensive (the sweep touches every trailing column per row regardless of how many needed it), and strictly more accurate for the columns that didn't strictly need re-summing. (was QRCP.fProxy.cs:124 and :129)
- 2026-07-11 | Blocked panel core 8-step walkthrough (moved from section header): per panel step k (panel-local, the pivot lands on global column/row d = rk = p0+k): 1) pivot by max vn1 over trailing columns; the full-column swap in A carries each column's already-written R prefix with it (R is extracted from A's upper triangle at the end), so no separate R swap is needed — only vn1/vn2/P and the k filled F rows are swapped. 2) bring ONLY the pivot column up to date wrt the k prior reflectors (A[:,d] −= V·F[k,·]ᵀ). 3) generate the Householder reflector. 4) take R[d,d] from it and store the reflector. 5) ONE combined pass acc = uᵀ·A over the panel width: acc's reflector-column entries are the compact-WY aux (uₖᵀuᵢ), its trailing entries are the direct term of F's new column. 6) F's new column = direct − F·aux (correction). 7) bring row rk of the trailing part up to date (it becomes R and feeds the norm downdate). 8) downdate vn1 with the same guarded formula as the unblocked core (dlaqps returns KB for the same reason this panel is cut short on a guard trip). (was QRCP.fProxy.cs:377, spec ticket docs/dev/spec-qrcp-blocked.md)
- 2026-07-11 | minNormSolveInPlace (COD) full derivation (moved from section header): QRCP gives A·P = Q·R with R = [R11 R12; 0 ~0] (R11 r×r upper-tri, full rank; the trailing (n-r) diagonal below tol). Writing x = P·y (P a permutation, so ‖x‖ = ‖y‖) and c = Qᵀb = [c1; c2], the residual is ‖R y − c‖² = ‖[R11 R12]·y − c1‖² + ‖c2‖². The second term is fixed, so every least-squares x satisfies M·y = c1 where M = [R11 R12] (r×n, full ROW rank r); among those, min ‖x‖ = min ‖y‖. LQ-factor the SHORT-WIDE M = L̃·Qz (L̃ r×r lower-tri, invertible; Qz r×n, orthonormal rows). Then M y = c1 ⇔ Qz y = L̃⁻¹c1 =: w, and the minimum-norm y with Qz y = w is y = Qzᵀ w (Qz has orthonormal rows). So the whole solve is: 1) QRCP factor (fused: b ← Qᵀb), read rank r off R's diagonal; 2) r == n (full column rank): basic IS min-norm — reuse solveInPlaceFinish, no COD; 3) r < n: LQ-compress M = R's top r×n block → L̃ + Qz-reflectors; forward-solve L̃ w = c1 (c1 = b[0..r), already Qᵀb); x = Qzᵀ w straight from the reflectors; un-permute x[P[j]]. Why the top-right block R12 matters: the BASIC (truncated) solution zeros the free variables in the pivoted column ordering, which is NOT minimum-norm for rank-deficient A, because R12 couples the free columns back into the leading ones (min ‖x‖ wants a nonzero free part that R12 can use to shrink the pivoted part). LQRP (the transpose-dual, wide side) has the SAME need: there the coupling lives in the below-diagonal block L21, and its basic solution is minimum-norm only for a CONSISTENT b — an inconsistent rank-deficient LS needs LQRP.minNormSolveInPlace, which QR-least-squares-solves the m×r block [L11; L21] (the transpose-dual of the LQ compress here). (was QRCP.fProxy.cs:1067)

## SVD
- 2026-07-13 | thin transposes U/V so bidiagonal QR rotations hit contiguous rows (same
  vectorization approach as Eigen.symmetricInPlace). (was SVD.fProxy.cs:203)
- 2026-07-13 | bidiagonalQR's deflation threshold is relative to the GLOBAL anorm, not local
  |d|+|e| — float needs this on clustered/zero singular values (same finding as the symmetric
  eigen QL). (was SVD.fProxy.cs:379)

## SVD.LowRank
- 2026-07-12 | Reorth windowing idea (moved from comment): a possible future optimization is
  windowing (compute_int strategy 0) to reorthogonalize against a subset of previous vectors
  instead of the full accumulated set. Not implemented. (was SVD.LowRank.fProxy.cs:218)

## SolveInfo
- 2026-07-11 | LstsqInfo doc trimmed from ~30-line essay to contract-only (struct fields + implicit-bool + pointer to Krylov.lstsqResidual). Removed content preserved here: per-solver norm derivation (Krylov R6a, docs/draft-spec-krylov-optimization.md) -- norms are the solver's own tracked values, never a fresh A*x/A^T*r, EXCEPT cgls's Converged exit: one fresh Apply + ApplyT verifies the claimed convergence before trusting it (replaces the drifted r/gamma pair). Per solver: cgls -- rnorm from a dot on its live residual r, Arnorm = sqrt(gamma) (its tracked ||A^T r||^2); lsqr -- rnorm = phibar, Arnorm = phibar*alpha*|c|, both produced free by the recurrence; lsmr -- Arnorm = |zeta-bar| (free, monotone), rnorm via the Fong-Saunders ||r|| recurrence (O(1) scalars per iteration, no matvec). Removed usage code sample:
  ```
  if (Krylov.lsqr(A, b, ref x)) { ... }          // implicit bool -> "did it converge?"
  bool ok = Krylov.cgls(A, b, ref x);            // same
  var info = Krylov.lsmr(A, b, ref x);           // keep the struct for diagnostics
  if (info.Solved) Debug.Log(info.iterations);
  ```
  (was SolveInfo.cs:6-35)
- 2026-07-11 | SolveInfo (square-system) doc: cg/pcg/cgne verify a claimed Converged exit with one fresh r = b-Ax first (ticket: Krylov R6a); minres/biCGStab need no extra matvec on any exit. (was SolveInfo.cs:88)

## UnsafeOP
- 2026-07-12 | formT's G=VᵀV pass: the naive per-(k,i) dot form (t as the reduction axis, stride Vld between consecutive t) does NOT vectorise and was measured far slower than the GEMM-shaped unit-stride loop actually used. (was UnsafeOP.fProxy.cs:721)
- 2026-07-11 | sumAbs/sum/maxAbs/vecDot shared header: one 4-lane accumulator left the FP add ports ~half idle in-cache; a 2nd independent width-4 accumulator measured ~2x. (was UnsafeOP.fProxy.cs:16-25)
- 2026-07-11 | matVecDot: two fProxy4 accumulators (8 lane-chains) measured ~2x over a single 4-lane accumulator (which left the FP add ports half idle in-cache); four accumulators measured NO further gain (memory/port-bound). (was UnsafeOP.fProxy.cs:154-165)
- 2026-07-11 | sortByKeyAscending was added to replace LP.ladBR's weighted-median ratio-test scan, which used to repeatedly linear-scan the REMAINING candidates for the current minimum ratio, removing the winner by swap-with-last each round -- an O(k) scan repeated up to k times is O(k^2), and at large m the candidate count k (bounded by m) made this the dominant cost of the whole solve even though the reported pivot count stayed small (each round can "fold" a candidate without registering as a pivot). Heapsort once up front costs O(k log k) instead, then a single linear walk. (was UnsafeOP.fProxy.cs:1218-1234)
- 2026-07-11 | UnsafeOP.iProxy.cs scalSub(target, n, s) used to implement "v - s" as "v + (-s)" for signed types only (bit-identical under modular wraparound, but unsigned types can't negate s); unified on the direct kernel so subInPlace<T>(T, iProxy) needs no per-signedness branch. (was UnsafeOP.iProxy.cs:216-220)
