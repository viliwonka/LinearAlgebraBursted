# DRAFT spec (for review): convex QP — HiGHS-style primal active set

Status: DRAFT FOR USER REVIEW (2026-07-09). Nothing here is committed to; every section is
negotiable. Companion drafts: `draft-spec-mip.md`, `draft-spec-sparse-dual-simplex.md`.

## OPEN QUESTIONS FOR USER

1. **Method: active set (recommended) vs OSQP-style ADMM?** Active set is the HiGHS-faithful
   choice — exact solutions, warm-startable, natural MIQP building block later — at ~4 rounds.
   ADMM (per `research-lp-qp-solver-landscape.md` §3.2) is ~2 rounds and handles sparse via
   `Krylov.pcg`, but converges to medium accuracy only and has no basis to warm-start. This
   draft specs the active set; say the word and it becomes the ADMM spec instead (or both, as
   v1 + v2).
2. **API home: new `QP` facade class (recommended, mirrors `LP`) or overloads on `LP`?**
3. **Convexity contract:** v1 requires Q positive SEMIdefinite and regularizes a singular
   reduced Hessian (δ·‖Q‖∞·I on Cholesky breakdown) instead of implementing negative-curvature
   handling. Acceptable? (HiGHS also assumes PSD; indefinite QP is NP-hard and out of scope.)
4. **Sparse QP deferred entirely?** The natural sparse route is ADMM/Krylov (matrix-free), not
   active set — it would arrive as a later, separate feature if wanted.
5. **Pricing:** v1 uses Dantzig (most-negative multiplier). HiGHS ships Dantzig/Devex/steepest
   edge (`qpsolver/devexpricing.hpp` etc.); Devex is a small v2 upgrade. OK to start Dantzig?

## What HiGHS actually does (verified)

HiGHS's QP solver ("quass", by Michael Feldmeier, `highs/qpsolver/`) is a **primal null-space
active-set method** — verified from the qpsolver README and the source file map
(`quass.cpp`, `basis.cpp`, `factor.hpp`, `ratiotest.cpp`, `dantzigpricing.hpp`,
`devexpricing.hpp`, `steepestedgepricing.hpp`, `feasibility_highs.hpp`, `perturbation.cpp`):

- Problem: min ½xᵀQx + cᵀx s.t. L ≤ Ax ≤ U, l ≤ x ≤ u, Q PSD. Same row+bound form as our
  simplex computational form.
- The method is explicitly described as "a generalization of the primal simplex algorithm".
  It maintains a **working set** of active constraints treated as equalities, parameterizes
  feasible moves as x = Y·b_A + Z·y where A_A·Y = I and A_A·Z = 0 (Z spans the null space of
  the active rows), and solves the **reduced-Hessian system ZᵀQZ·y = −Zᵀ(c + Q·Y·b_A)** for
  the step. It keeps a factorization of an augmented basis B_A = [A_A; V] (V = conditioning
  rows) rather than forming Y/Z explicitly.
- Loop: solve the equality-constrained subproblem in the null space → if step ≈ 0, check KKT
  multiplier signs (optimal or drop a constraint) → else ratio test over inactive constraints
  (add the blocker). Feasible starting point comes from an LP solve ("can be found by using
  the simplex algorithm" — `feasibility_highs.hpp` calls HiGHS's own LP solver).
- Support machinery: pricing variants, a Harris-style ratio test, bound perturbation, scaling.

Judgment (not verified detail): for a dense v1 at this library's target sizes (n up to a few
hundred), the B_A=[A_A;V] factor-update machinery is not worth porting — recomputing a QR of
the working-set matrix per active-set change is O(n³) but simple, correct, and dense-fast;
HiGHS needs the incremental machinery because it targets sparse instances. Updates are a v2.

## v1 scope

Dense convex QP: **min ½xᵀQx + cᵀx s.t. A x {≤,=,≥} b, xl ≤ x ≤ xu**, Q symmetric PSD,
everything `fProxy`-templated, job-safe, new files only.

Explicitly deferred: sparse QP (would be ADMM over BSR + `Krylov.pcg`, a different spec),
factorization updates (QR up/downdate — the QRCP downdating machinery is the natural donor),
Devex/steepest-edge pricing, dual active set (Goldfarb–Idnani), MIQP, indefinite Q,
crossover-style refinement.

## Algorithm (v1, dense null-space active set)

State: feasible x; working set W = a subset of the active constraints (rows of A at their
bound + variable bounds as eᵢ rows) whose normals are independent; k = |W| ≤ n.

1. **Feasible start (phase 1):** solve the LP feasibility problem with the existing
   `LP.solve` (dual simplex, zero cost) — reuse, exactly as HiGHS does. The returned vertex's
   tight constraints seed W (drop dependents via the QR rank test below).
2. **Null-space step:** assemble A_Wᵀ (n×k dense), QR-factor it with the library's `QR`
   (Householder). Z = the trailing n−k columns of Q (apply Q to unit vectors — or better,
   keep Q implicit and apply reflectors, same trick the QR solve kernels use). Gradient
   g = Q·x + c (GEMV). Reduced Hessian H_Z = ZᵀQZ (two `matMatDot` calls), then
   `CHO.decomp` + `CHO.decompSolve` for H_Z·y = −Zᵀg; step p = Z·y.
   - If `CHO` breaks down (H_Z singular — Q only PSD): retry with H_Z + δ·‖Q‖∞·I,
     δ = sqrt(Consts.fProxyEpsilon); if the regularized step is a descent direction proceed,
     else declare Unbounded (only possible with Q singular along an unbounded ray).
3. **If ‖p‖ ≤ tol:** compute multipliers λ from A_Wᵀλ = g — least squares via the SAME QR
   factor (R-solve of Qᵀg). Sign check against each constraint's side (≤ vs ≥ vs bound).
   All signs correct → **Optimal**. Else drop the worst-sign constraint from W (Dantzig
   pricing), loop.
4. **Else ratio test:** α = min(1, min over inactive constraints i with aᵢᵀp violating-ward
   of (bᵢ − aᵢᵀx)/(aᵢᵀp)), Harris-style two-pass with feasTol relaxation (port the shape of
   `HarrisRatioTest`, not the code — the candidates here are constraints, not basics).
   x ← x + αp; if α < 1 add the blocking constraint to W (skip if it would make A_W
   rank-deficient — detectable from the QR R-diagonal). Loop.
5. **Anti-cycling / degeneracy:** iteration cap; after a run of α=0 steps, apply HiGHS-style
   bound perturbation (deterministic seed, removed at the end — same pattern as the dual
   simplex cost perturbation, and the same lesson applies: decisions about optimality use
   ORIGINAL data, see spec-revised-simplex.md "Numerics").

Per-iteration cost: one QR of n×k + one Cholesky of (n−k)² + GEMMs — O(n³) worst case, fine
for dense v1. Objective/diagnostic sums accumulate in plain `double` locals per the
`simplexCore` convention.

## Reuse (existing kernels, by name)

- `LP.solve` / `DualSimplexCore` — phase-1 feasible point.
- `QR.decomp` + the Householder apply kernels — working-set factorization, Z application,
  multiplier least squares.
- `CHO.decomp` / `CHO.decompSolve` — reduced Hessian.
- `matMatDot` (+ transA variant) — ZᵀQZ; GEMV kernels for g and constraint activities.
- `HarrisRatioTest` — pattern donor for the two-pass constraint ratio test.
- `Consts.fProxyZeroThreshold` / `Consts.fProxyEpsilon` — tolerance derivation, INLINE at call
  sites (the CS0111 return-type-only-helper trap from spec-revised-simplex.md applies here).

## New infrastructure (genuinely required — all small)

- Working-set bookkeeping: `NativeArray<int>` active list + per-constraint status byte
  (Inactive / ActiveLower / ActiveUpper / Equality). No new math machinery.
- `QPInfo` / `QPStatus` result structs (Optimal / Infeasible / Unbounded / MaxIter +
  iterations, KKT residuals) — mirror `LPInfo` per the solver diag-struct conventions.
- A `QP` facade class + `QP.fProxy.cs` template. No new factorization technology at all —
  this is the cheapest of the three drafts on infrastructure.

## Staged plan (coder-agent rounds) + oracles

- **Stage 1 — EQP core (~1 round):** fixed working set, steps 2–3 only (equality-constrained
  QP). Oracle: solve the full KKT system [[Q, A_Wᵀ],[A_W, 0]] directly with the library `LU`
  on the same instance — x and λ must agree to factor tolerance. Random SPD Q (the random
  property-matrix generators exist) + random independent A_W, both dtypes.
- **Stage 2 — inequality loop (~1.5 rounds):** ratio test, add/drop, Dantzig pricing.
  Oracles: Hock–Schittkowski knowns with published optima (HS21, HS35, HS52, HS76 — tiny,
  embed as literals; they are the standard convex-QP acceptance set and the small end of
  Maros–Mészáros); box-constrained random QP vs a brute-force active-set enumeration at n ≤ 8;
  LP limit case (Q = 0 must reproduce `LP.solve`'s objective).
- **Stage 3 — phase 1 + hardening (~1–1.5 rounds):** LP feasible start, infeasible detection,
  degenerate/redundant-constraint instances, PSD-singular Q (e.g. Q = LLᵀ with rank n/2),
  perturbation, `QPBenchmark` section. Oracle: KKT residual checks (stationarity,
  complementarity, feasibility) on every random instance — QP's advantage is that optimality
  is cheaply verifiable without a reference solver.

Effort: **~3.5–4.5 rounds ≈ 1.3–1.5× the dense simplex build** (3 rounds). Risk is
concentrated in step-3 multiplier signs and degenerate ratio tests, not in infrastructure.

## Files

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs` — facade + core.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.Info.cs` — `QPInfo`/`QPStatus`
  (hand-written shared, like `LP.Info.cs`).
- Tests: `TemplateSourceTests/fProxy/QPTests.fProxy.cs` (enum+dispatch pattern,
  `CompileSynchronously = true`, no enum Asserts in jobs).
- Repo constraints: identical to spec-revised-simplex.md's list (templates source of truth,
  `Allocator.Temp` scratch, CS1750 forwarding overloads, don't touch existing solver cores).

## References

- HiGHS qpsolver README + source tree (`highs/qpsolver/` — verified 2026-07-09 via mirror).
- Nocedal & Wright, *Numerical Optimization*, ch. 16.5 (primal active-set for convex QP —
  the textbook version of exactly this method).
- Gill, Golub, Murray & Saunders (1974) — factorization updating (the deferred v2).
- Maros & Mészáros (1999) — convex QP test set; Hock & Schittkowski (1981) — test examples.
- OSQP: Stellato et al., arXiv:1711.08013 (the ADMM alternative, open question 1).
