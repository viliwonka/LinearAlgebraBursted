# Spec: LQR (discrete-time linear quadratic regulator)

User-approved 2026-07-11. First feature of the roadmap's dynamics-and-control corner.

## Problem

Discrete-time linear dynamics x_{k+1} = A x_k + B u_k (A: n×n, B: n×m), quadratic cost
Σ (xᵀQx + uᵀRu), Q sym PSD (n×n), R sym PD (m×m). Compute the optimal feedback gain K
(m×n) so u = −K x. Game sizes: n ≤ ~12, m ≤ ~6 — correctness and API ergonomics are the
product, not throughput. Runtime application (u = −Kx) is the caller's own matvec; this
feature is the SOLVERS.

Two usage regimes drive the API (user ruling: regime 2 is a first-class citizen):
1. Fixed dynamics: compute K once (authoring/load time), use forever.
2. Per-frame re-linearization: A/B change slightly every frame (nonlinear system
   linearized at the current state); K must be re-solved cheaply — warm-started from the
   previous frame's Riccati solution S, exactly the LPBasis warm-start philosophy.

## Entry points (new template `OP/Control.fProxy.cs` + type-agnostic `OP/Control.Info.cs`)

1. `Control.lqr(in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
   ref fProxyMxN K)` → `LQRInfo`. Infinite-horizon DARE via **SDA** (structure-preserving
   doubling). Cold solve, quadratic convergence, ~10-25 doubling steps.
2. `Control.lqr(..., ref fProxyMxN K, ref fProxyLQRState state)` → `LQRInfo`. Warm path:
   `fProxyLQRState` (buffer-carrying struct, allocator + Dispose + IsCreated, house
   pattern = fProxyLPCache/fProxyCHOPCache) carries S (n×n) across calls. If
   `state.populated`: iterate the plain Riccati recursion from the carried S until
   converged (a slightly-changed A/B converges in a few steps). If not populated (or
   dimensions mismatch → throw): cold SDA, then store S. Always stores the terminal S +
   sets populated on any Converged exit.
3. `Control.lqrSchedule(in A, in B, in Q, in R, in Qf, int N, ref fProxyMxN Kschedule)`
   → `LQRInfo`. Finite horizon: backward recursion from terminal cost Qf, N steps;
   Kschedule is (N·m)×n, gain for step k in rows [k·m, (k+1)·m). Secondary entry — it
   exists because the recursion kernel is shared and it is SDA's test oracle.

## Algorithms

- **Shared Riccati step kernel** (one internal method, used by the schedule, the warm
  path, and the SDA oracle tests):
  S⁻ = Q + Aᵀ S A − Aᵀ S B (R + Bᵀ S B)⁻¹ Bᵀ S A, and K = (R + Bᵀ S B)⁻¹ Bᵀ S A.
  The m×m solve uses **CHOP unconditionally** (rank-revealing pivoted Cholesky): m ≤ 6
  makes the pivoting cost irrelevant, and CHOP degrades gracefully on semidefinite R /
  early-iteration near-singularity (the LAD Frisch-Newton precedent — plain CHO
  hard-fails on a non-positive pivot). If CHOP reports rank < m, complete the solve with
  its rank-deficient branch but record `rankDeficientControl = true` in the info (the
  optimal control is non-unique; do NOT silently hide that).
- **SDA** for the cold infinite-horizon solve. FIDELITY RULE: fetch the actual SDA-for-
  DARE recurrences from a real source and cite it — do not implement from this sketch or
  from memory. The sketch (to be VERIFIED against the source, not trusted): with
  A₀ = A, G₀ = B R⁻¹ Bᵀ, H₀ = Q, iterate
  A_{k+1} = A_k (I + G_k H_k)⁻¹ A_k,
  G_{k+1} = G_k + A_k G_k (I + H_k G_k)⁻¹ A_kᵀ,
  H_{k+1} = H_k + A_kᵀ (I + H_k G_k)⁻¹ H_k A_k;
  H_k → S quadratically. The (I + G H) solves are NONSYMMETRIC n×n → LU, not Cholesky.
  Sources: Chu–Fan–Lin (structure-preserving doubling literature) or the Huang–Li–Lin
  SIAM book chapter on SDA; any authoritative statement of SDA-1 for DARE is acceptable
  — cite what was fetched in the spec-addendum section below.
- **Hygiene (all paths)**: re-symmetrize S (and SDA's G/H) every iteration
  ((M + Mᵀ)/2); divergence detection (‖S‖ or ‖H_k‖ above a data-scaled blowup threshold
  → `LQRStatus.Diverged` — unstabilizable/undetectable systems must FAIL FAST with a
  status, never hang or return garbage); iteration caps (SDA default ~50, warm recursion
  default ~500, both overridable via a maxIter parameter defaulting 0 = library default).
- Convergence: relative change ‖S_new − S_old‖_F / max(1, ‖S_new‖_F) ≤ tol, per-dtype
  tol via the usual Consts-derived expressions (mirror the house pattern; no bare float
  literals outside //+choose).

## Numerics / house rules

fProxy storage throughout; double only as local scalar control math (norms, convergence
ratios). No double-only kernels. Reuse library kernels: Blas GEMM/dot for the products,
CHOP for the m×m solve, LU for the SDA n×n solves. All buffers Temp inside the solve
except what fProxyLQRState owns. Comment style: short factual; the SDA derivation
citation goes in a short addendum section appended to THIS spec, not in code comments.

## Types (Control.Info.cs, type-agnostic; fProxyLQRState in the template)

- `enum LQRStatus { Converged, MaxIterations, Diverged }`
- `struct LQRInfo { public int iterations; public double residual; public LQRStatus
  status; public bool rankDeficientControl; }` — every field from already-computed
  numbers (house diag-struct rule).
- `struct fProxyLQRState { fProxyMxN S; bool populated; ... IsCreated/Dispose }`.

## Validation (managed-thread throws, house pattern)

Dimension checks on every entry (A square, B rows = n, Q n×n, R m×m, K m×n, Kschedule
(N·m)×n, N ≥ 1); Q/R symmetry NOT validated numerically (documented caller contract,
consistent with how the library treats SPD inputs elsewhere) — but R with a non-finite
or negative diagonal is cheap to reject and should throw.

## Tests (test-writer, after coder)

1. Literature vectors: at least two published dlqr instances with known K (double
   integrator is the canonical one; fetch a documented MATLAB/SciPy example with printed
   gains, cite it). Both dtypes, tolerance per dtype.
2. SDA vs oracle: SDA's S/K == the shared recursion run to convergence (10k-step cap) on
   random stabilizable instances (n ∈ {2,4,8,12}, m ∈ {1,2,4}), tight tolerance.
3. Properties on random stabilizable instances: S symmetric PSD; closed-loop stability —
   max |λ(A − BK)| < 1 via the existing Eigen.valuesQR (values-only is enough).
4. Warm path: solve cold, perturb A entries by ~1e-3-relative, warm re-solve → converges
   in ≤ a small iteration count (assert with margin) to the same S/K as a cold solve of
   the perturbed system (tolerance).
5. Schedule: with N large and Qf = Q, K_0 of the schedule ≈ the infinite-horizon K
   (tolerance); schedule length-1 sanity vs a hand-computed single Riccati step.
6. Failure modes: an unstabilizable instance (e.g. uncontrollable unstable mode:
   A = diag(2, .5) with B = [0; 1]) → Diverged status, finite time. Semidefinite R with
   redundant actuator (duplicate B columns, R = diag(1, 0)) → rankDeficientControl
   flagged, returned K still stabilizes (property 3 check).
7. Determinism ×2; dimension-mismatch throws.

## Benchmark

Small `LQRBenchmark` (template + hand harness, house pattern): cold SDA vs cold
recursion vs warm recursion (after 1e-3 A-perturbation), n ∈ {4, 12}, m ∈ {2, 4}, both
dtypes, med ms + iterations. Expect: everything in microseconds (that is the point —
record it); warm ≪ cold-recursion; SDA ≪ cold-recursion. Budget: trivial; add to
AllBenchmarks.

## Out of scope (recorded so they don't creep in)

Continuous-time (CARE), c2d discretization helper (belongs to the ODE-integrator
feature), Kalman/LQG (natural sequel, same Riccati machinery), iLQR/MPC (the warm-state
API is deliberately shaped so these can build on it later), Schur-based DARE (rejected:
ordered Schur is the library's biggest-possible kernel investment for thin gamedev
customers — SDA reaches the same accuracy class without it), cart-pole demo scene
(roadmap's demo batch).

## SDA source addendum (coder fills in)

**Source fetched (primary):** Chun-Yueh Chiang, Hung-Yuan Fan, Wen-Wei Lin, "A structured doubling
algorithm for discrete-time algebraic Riccati equations with singular control weighting matrices"
(NTHU/NCTU preprint, jupiter.math.nycu.edu.tw/~wwlin/papers_new/prep2007-5-001.pdf), Section 2,
Algorithm 2.1 and equations (2.5a-c)/(2.12a-c). This paper's Algorithm 2.1 builds on, and cites as its
origin, E. K.-W. Chu, H.-Y. Fan, W.-W. Lin, "Structure-preserving algorithms for periodic discrete-time
algebraic Riccati equations," Int. J. Control, 77 (2004), pp. 767-788 -- the Chu-Fan-Lin lineage the
spec names. **Cross-checked (secondary):** Federico Poloni, "Iterative and doubling algorithms for
Riccati-type matrix equations: a comparative introduction" (arXiv:2005.08903), Section 4.2, equations
(33a-c) -- an independent restatement of the identical doubling recursion (there derived from the
inverse-subspace-iteration/repeated-squaring argument rather than the SSF/Sherman-Morrison-Woodbury
argument Chiang-Fan-Lin use), confirming the recurrence is the field's standard SDA-1, not one paper's
idiosyncratic variant.

**Recurrence as fetched (Chiang-Fan-Lin's general form, DARE with cross-term C and singular R):**
Algorithm 2.1 first reduces the general DARE `X = AᵀXA + Q - (C+BᵀXA)ᵀ(R+BᵀXB)⁻¹(C+BᵀXA)` to standard
symplectic form via a chosen symmetric `Y` with `R+BᵀYB` invertible: `A0 = (I-G0Y)A - BR̄⁻¹C`,
`G0 = BR̄⁻¹Bᵀ`, `H0 = Q - Y - CᵀR̄⁻¹BᵀYA - AᵀYBR̄⁻¹C - CᵀR̄⁻¹C + AᵀY(I-G0Y)A` (R̄ = R+BᵀYB), then doubles:

```
A_{k+1} = A_k (I + G_k H_k)⁻¹ A_k                          (2.12a)
G_{k+1} = G_k + A_k G_k (I + H_k G_k)⁻¹ A_kᵀ                (2.12b)
H_{k+1} = H_k + A_kᵀ (I + H_k G_k)⁻¹ H_k A_k                (2.12c)
```

with the loop's own stated breakdown rule: "If I + GkHk is ill-conditioned, then break down." H_k
converges monotonically to the (almost-)stabilizing solution X_s = lim H_k + Y.

**What this codebase implements:** the no-cross-term, nonsingular-R specialization the library's LQR
problem statement actually needs (C = 0, R already invertible/PSD so Y = 0 is a valid choice -- no SSF
reduction needed). Substituting C = 0, Y = 0 into (2.5a-c) collapses the initialization to exactly the
spec's own sketch: `A0 = A`, `G0 = BR⁻¹Bᵀ`, `H0 = Q`, and (2.12a-c) is implemented VERBATIM (see
`Control.SDACore` in `Control.fProxy.cs`) as:

```
A_{k+1} = A_k (I + G_k H_k)⁻¹ A_k
G_{k+1} = G_k + A_k G_k (I + H_k G_k)⁻¹ A_kᵀ
H_{k+1} = H_k + A_kᵀ (I + H_k G_k)⁻¹ H_k A_k
```

`H_k -> S` (the DARE solution) with `X_s = H_k` (Y = 0, so the `+ Y` correction is a no-op). Fidelity
verdict: **the spec's own sketch was correct** for this codebase's problem shape -- both the primary
source (specialized to C=0, Y=0) and the independent secondary source (Poloni eq. 33a-c) confirm it
verbatim. `G0 = BR⁻¹Bᵀ` is built via CHOP (rank-revealing) rather than a bare inverse, so a semidefinite
R degrades gracefully there too, consistent with Chiang-Fan-Lin's own motivation (their paper's whole
point is handling singular R without inverting it away with an ad hoc regularization). The `(I+GH)`/
`(I+HG)` solves are solved via `LU.decompInPlace` + multi-RHS `LU.decompSolve` (nonsymmetric, matching
the paper); the paper's own "ill-conditioned -> break down" rule maps to this implementation's
`LU.decompInPlace` returning `Singular` (forces `LQRStatus.Diverged`), backstopped by the data-scaled
Frobenius-norm blowup check on `H_k` (hygiene rule, both paths fail fast on an unstabilizable system).
