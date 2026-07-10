# Spec: HiGHS-style dense revised simplex (primal + dual)

Goal: a new dense LP backend with HiGHS-lineage algorithms — bounded-variable **revised**
simplex over an LU-factored basis — fully Burst-compatible and job-safe. This is stage one of
porting the HiGHS solver family (dense simplex now; sparse simplex / QP / MIP later).

**The existing two-phase tableau simplex (`simplexCore` in `LP.fProxy.cs`) stays untouched** —
it is the correctness baseline the new solver is validated against. New code goes in new files.

## Why revised simplex

The tableau method updates the full m×(n+m) tableau every pivot: O(mn) writes/iteration and
O(mn) memory traffic, with error accumulating in the tableau itself. The revised method keeps
the ORIGINAL A and a factorization of the m×m basis matrix B, reconstructing what it needs via
solves (FTRAN/BTRAN). Error is bounded by periodic refactorization, bounded variables are
native (no row-doubling for ranges), and per-iteration cost is O(m²)+O(mn) pricing with a far
smaller constant. This is what every serious LP code (HiGHS, CPLEX, Gurobi) does.

## Computational form

Public entry stays `LP.solve(A, b, c, senses, ref x, out objective, method, maxIter)`:
min cᵀx s.t. Aᵢ·x {≤,=,≥} bᵢ, x ≥ 0. Internally build the HiGHS computational form:

    min cᵀx   s.t.   A x + s = b,    l ≤ (x, s) ≤ u

- N = n + m variables: j < n are structurals (column A[:,j], bounds [0, +INF)), j = n+i is the
  logical of row i (column e_i, bounds by sense: `≤` → [0, +INF); `=` → [0, 0]; `≥` → (−INF, 0]).
- INF = 1e30 sentinel. No free variables arise in this form.
- State: `basis[m]` (variable index basic in row i), `status[N]` ∈ {Basic, AtLower, AtUpper}.
  Nonbasic variables sit exactly ON a bound. Fixed variables (l==u) count as AtLower.
- Initial basis: all logicals (B = I), structurals nonbasic at lower bound 0.

## Basis factorization + FTRAN/BTRAN (the kernel layer)

- Refactorize: gather B (m×m dense, column j of B = column basis[j] of [A|I]) and LU-factor
  with partial pivoting — reuse the library's `LU` (compact in-place decomp + the triangular
  solves in `Blas`); do NOT reimplement GETRF.
- FTRAN (solve B·α = a): permute, L-solve, U-solve, then apply the eta file in order.
- BTRAN (solve Bᵀ·ρ = e): apply eta file in REVERSE (transposed), then Uᵀ-solve, Lᵀ-solve,
  un-permute.
- Product-form (PFI) eta updates: after a pivot with entering column α_q (=B⁻¹a_q) and leaving
  row r, store eta = (r, α_q). Applying an eta to v: `v[r] /= α_q[r]; v[i] -= α_q[i]*v[r]` (i≠r);
  transposed apply for BTRAN is the mirror. Refactorize when the eta count hits
  REFACTOR_INTERVAL = 64, when |α_q[r]| < pivot tolerance, or on detected instability
  (see accuracy check below).
- Accuracy check at refactorization: recompute basic values xB fresh (FTRAN of the adjusted
  rhs) and compare against the incrementally-updated xB; if they disagree beyond tolerance,
  tighten (drop etas, refactor more often is automatic since refactor just happened — flag and
  continue with the fresh values, which are authoritative).

## Numerics (fProxy throughout — REVISED, see "Numerics design history" below)

Everything is `fProxy`, matching this library's templated-everything convention: matrix/vector
storage, the basis factorization, FTRAN/BTRAN, the eta file, reduced costs, ratios, dual weights.
There is no double-only kernel and no `//+skipFor[float]` single-emission trick — every kernel
function is a normal per-dtype template, duplicated into the float and double generated files
exactly like every other solver in this library (`simplexCore`, `interiorCore`).

The basis factorization reuses the library's own `LU` class (`LU.decompInPlace`, the compact
in-place form, plus `Blas`'s compact-LU triangular solves via `LU.decompSolve`) for FTRAN — no
reimplementation of GETRF. BTRAN (solve Bᵀy = v) has no library primitive to reuse (`Blas` only
ships forward compact-LU solves), so it is a small hand-written transposed forward/back
substitution over the SAME compact `fProxyMxN` + `Pivot` factor `LU.decompInPlace` produces — the
transposed read pattern of an existing factor, not a different algorithm (see
`LP.RevisedSimplex.fProxy.cs`'s `SolveTranspose` for the derivation).

Tolerances are derived from the SAME per-dtype `Consts` the tableau simplex already uses
(`Consts.fProxyZeroThreshold`, `Consts.fProxyEpsilon`), which resolve to the float or double
constant via the ordinary token substitution — e.g. `pivTol = max(Consts.fProxyZeroThreshold,
1e-9)`, `feasTol = max(sqrt(Consts.fProxyEpsilon), 1e-7)`. These are computed INLINE at each call
site, not behind a shared helper method: a helper returning `fProxy` with no `fProxy`-typed
parameter (e.g. a bare `Tol()`) differs only in RETURN TYPE between the float- and
double-generated fragments of the same partial class, and C# does not overload on return type
alone, so both copies collide as duplicate members (CS0111) — this bit both the tolerance helpers
and a `PerturbUnit(int)` cost-perturbation helper during stage 2 development; the fix in both
cases was to inline the computation rather than name it. Objective and other pure-local
diagnostic SUMS that are never fed back into array storage (e.g. the phase-1 infeasibility sum,
the final cᵀx recompute) still accumulate in plain `double` for reporting precision, matching
`simplexCore`'s own convention — this is a normal, narrow use of a double local, not a
double-only code path.

Two numeric pitfalls surfaced only during stage 2 (dual simplex) testing, both fixed and worth
recording since they generalize beyond this file:
- **Artificial-bounds dual phase 1 needs a data-scale-relative bound, not a fixed literal.**
  HiGHS's own `[0, 1e7]` box is tuned for its internally-scaled (equilibrated) problem data; this
  solver does not scale, so 1e7 against O(1) problem data is a scale mismatch that is invisible in
  double (headroom to spare) but corrupts float outright — summing many simultaneously-artificial
  columns' contributions into the adjusted-rhs during `RebuildXB` lands the basic values around
  1e8, and float's ~1.19e-7 relative precision at that magnitude is an ABSOLUTE error of order 10,
  which swamps `feasTol` (~3.45e-4) and produced a false `Infeasible` within a handful of dual
  iterations on a 48-variable random instance. Fixed by scaling the artificial bound to the
  problem's own data magnitude (100× the largest `|cost|`/`|rhs|` entry) instead of a bare
  constant.
- **A cost-perturbation-driven decision must use the ORIGINAL cost, not the perturbed one.**
  Deciding which nonbasic structurals need an artificial bound is a one-time TRUE-dual-feasibility
  check (`cost[j] < 0`), not part of the iterative pricing the perturbation exists to stabilize.
  Checking the sign of the *perturbed* cost instead meant a column with cost EXACTLY 0 — which
  `LP.lad`'s `[x⁺|x⁻]` reformulation always has many of — could get a pointless artificial bound
  purely from perturbation noise, corrupting the warm-started basis handed to the primal cleanup
  badly enough to report a false `Unbounded`.

### Numerics design history

An earlier version of stage 1 made this kernel layer pure `double` (`NativeArray<double>`, no
`fProxy` token at all), wrapped in a `//+skipFor[float]` marker so it was emitted once and shared
by both generated builds, with a hand-rolled compact LU ported from `LU.decompInPlace`'s
algorithm. This was REJECTED and reverted: it meant the float build secretly ran a double solver
(contrary to this library's templated-everything philosophy — the tableau simplex and interior
point are fully `fProxy` and work fine in float), and the hand-rolled GETRF bypassed the
library's own matrix types and `LU` class, which existed only because the double-only choice made
the library's proxy-typed APIs unreachable. The fProxy redesign above is the corrected design;
this paragraph is kept as a pointer so the same mistake is not repeated.

## Stage 1 — primal revised simplex (`LPMethod.RevisedSimplex`)

Bounded-variable primal simplex, phases 1+2:

- Pricing: Dantzig — most negative "effective" reduced cost: d_j < −tol for AtLower j,
  d_j > +tol for AtUpper j (candidate direction is then ±1). Compute y = B⁻ᵀc_B fresh via
  BTRAN each iteration, then price d_j = c_j − a_jᵀy over nonbasics (O(mn)); no incremental
  d maintenance in stage 1. Anti-cycling: after 3m consecutive degenerate pivots switch to
  Bland's rule (lowest index) until a nondegenerate step occurs.
- Ratio test (bounded-variable, Harris two-pass):
  entering j moves by t ≥ 0 in direction σ = ±1; basic values move xB_i ← xB_i − t·σ·α_i.
  Pass 1: relaxed max step Θ with tolerance δ = feasTol: for each i with |α_i| > pivTol, the
  step at which xB_i passes its violated-by-δ bound. Also Θ_self = u_j − l_j (a BOUND FLIP:
  entering variable reaches its own opposite bound — no basis change, just flip status and
  update xB). Pass 2: among candidates with step ≤ Θ pick the one with the LARGEST |α_i|
  (numerical stability), take the exact (unrelaxed) step of that row. If Θ = +INF and no self
  bound → Unbounded.
- Phase 1 (composite / sum of infeasibilities): with the all-logical start basis some basics
  violate their bounds (e.g. `≥` rows with b_i > 0). Phase-1 cost: +1 for basics above upper,
  −1 for basics below lower, 0 otherwise, on the BASIC variables (nonbasic phase-1 cost 0);
  reprice each iteration (costs change as variables cross bounds — the ratio test in phase 1
  must allow an infeasible basic to travel THROUGH its violated bound to the far bound,
  standard composite-ratio treatment: such a row contributes its far-bound step). Phase 1 ends
  when infeasibility sum ≤ feasTol·m (→ phase 2) or phase-1 optimum > 0 (→ Infeasible).
- Termination: no candidate entering → Optimal (report objective from fresh recompute cᵀx,
  not the incrementally-updated value). Iteration cap → MaxIter, return best current point.
- Result: reuse `LPInfo` / `LPStatus` (`Optimal`/`Infeasible`/`Unbounded`/`MaxIter`),
  iterations = total pivots (phase 1 + 2).

## Stage 2 — dual simplex (`LPMethod.DualSimplex`)

The HiGHS workhorse. Requires stage 1's kernel layer. Phase 2 first:

- Precondition: dual-feasible basis. From the all-logical basis, flip each nonbasic structural
  to the bound matching its reduced-cost sign where possible (d_j = c_j initially). Structurals
  with c_j < 0 and u_j = +INF are dual-infeasible → dual phase 1 (below).
- Leaving row: **dual steepest edge (DSE)**, Forrest–Goldfarb "steepest edge for the dual".
  Maintain w_i ≈ ‖B⁻ᵀe_i‖² (init: all-logical basis → w_i = 1 exactly). Choose the row r
  maximizing infeas_r²/w_r where infeas_r = bound violation of xB_r.
- Pivot row: ρ_r = B⁻ᵀe_r (BTRAN), then PRICE α_r = Nᵀρ_r over nonbasic columns (O(mn)).
- Dual ratio test (Harris two-pass + BFRT bound flips): direction fixed so the leaving
  variable moves to its violated bound. Candidates: nonbasic j with sign-correct α_rj
  (AtLower needs α̂_rj > pivTol, AtUpper needs α̂_rj < −pivTol, where α̂ absorbs the leaving
  direction). Ratios d_j/α̂_rj ≥ 0. Pass 1 relaxed by dualTol → Θmax; pass 2 pick largest
  |α_rj| within Θmax. **Bound-flipping ratio test**: if the blocking j is BOXED (finite l,u),
  flipping it to its other bound keeps dual feasibility and absorbs Δ = |α_rj|·(u_j−l_j) of
  the leaving row's infeasibility; while the remaining infeasibility stays positive, flip and
  continue to the next breakpoint (accumulate flips, apply their xB update with ONE extra
  FTRAN of the summed flip columns). Classic long-step dual ratio test — Koberstein's thesis
  ch. 6 is the reference implementation description; consult it or HiGHS's
  `HEkkDual::chooseColumn` if in doubt. No sign-correct candidate at all → primal Infeasible
  (dual unbounded).
  **A real pivot always terminates the walk — flips are only ever a PREFIX, never a substitute.**
  Verified against HiGHS's `HEkkDualRow::chooseFinal`: it always selects a `workPivot` and
  unconditionally FTRANs its column (`updateFtran`); the flip list from `updateFlip` is an
  ADDITIONAL side effect applied ALONGSIDE that pivot, never instead of one. This is not just an
  implementation-cleanliness point — a "flip-only" iteration (all of row r's infeasibility
  absorbed by flips, no entering variable) would leave the basis, hence y = B⁻ᵀc_B, hence every
  d_j, COMPLETELY UNCHANGED, so a column just flipped from AtLower to AtUpper would still carry
  its OLD (AtLower-side) reduced cost — generally the wrong sign for its new status — with no
  future iteration positioned to notice. Only a real pivot changes c_B and therefore y, which is
  what makes a flipped column's stale d_j become the correct, fresh-BTRAN-recomputed value next
  iteration. (The first implementation allowed flip-only iterations; it passed every test at
  n≤24 but produced a false Infeasible on a 48-variable random instance — exactly the failure
  mode this derivation predicts, since the corruption needs enough columns/iterations to
  surface. Fix: walk candidates by ascending ratio; a boxed one whose full flip would STILL leave
  positive remaining infeasibility is flipped and the walk continues; the first candidate that is
  either unboxed, or boxed but sufficient to finish the absorption on its own, becomes the actual
  pivot — never flipped, even if boxed.)
- After the pivot: FTRAN α_q = B⁻¹a_q, DSE update (needs τ = B⁻¹ρ_r, one extra FTRAN):
  w_r ← w_r/α_q[r]²; for i≠r: w_i ← max(w_i − 2·(α_q[i]/α_q[r])·τ_i + (α_q[i]/α_q[r])²·w_r_old/α_q[r]²·…)
  — use the exact Forrest–Goldfarb formulas (w_i' = w_i − 2(α_qi/α_qr)τ_i + (α_qi/α_qr)²w_r,
  then w_r' = w_r/α_qr²); guard with w_i ≥ 1e-4 floor. Verify against a reference before
  trusting: a wrong DSE update silently degrades to ~Dantzig.
- Cost perturbation (faithful port of `HEkk::initialiseCost`): on entry to dual phase 2,
  perturb structural columns by xpert = (1+r)·(|c_j|+1)·base with base = 5e-7·max|c| (max|c|
  dampened by sqrt(sqrt()) above 100 and clamped to 1 when <1% of variables are boxed); xpert
  is positive and SIGNED by bound structure (+ for lower-bounded, − for upper-bounded,
  sign(c_j) for boxed; free/fixed columns never perturbed — keeps a dual-feasible d_j
  dual-feasible). Logical (row) columns get only a symmetric ±0.5·1e-12 tie-breaker, ~7 orders
  smaller. Both bases use HiGHS's exact literals (5e-7 / 1e-12) for BOTH dtypes — they are
  representable in float, and a dualTol-scaled float variant (larger perturbations "to match
  float's tolerance") was benchmark-falsified: it exploded a float B&B tree from 29 nodes to
  a 20000-node limit. Deterministic hash r∈[0,1) replaces HiGHS's random vector. Remove the perturbation at the end and clean up any
  resulting dual infeasibilities with a few primal iterations (stage 1's primal is the cleanup
  engine — this is exactly how HiGHS composes them). History: the first version applied a
  symmetric ±1e-5·(1+|c_j|) to EVERY column including logicals — two independent fidelity
  reviews flagged that slack columns were getting a perturbation ~100× the dual tolerance
  (HiGHS's row scale is 1e-12); replaced by this faithful port.
- Dual phase 1 (only when needed): use the **artificial-bounds** method — give every
  dual-infeasible nonbasic a temporary finite box ([0, artificialBound]), making the basis
  dual-feasible after flips; run dual phase 2; afterwards restore real bounds — variables stuck at
  an artificial bound become primal-infeasible basics/nonbasics handled by the primal cleanup.
  (HiGHS's subproblem dual phase 1 is stronger; artificial bounds is acceptable for dense v1
  and is what several production codes shipped for years.)
  `artificialBound` is DATA-SCALE-RELATIVE (100× the largest `|cost|`/`|rhs|` entry in the
  problem), not HiGHS's literal `1e7` — HiGHS's own bound is tuned for its internally-scaled
  (equilibrated) problem data, and this solver does not scale, so a bare `1e7` against typical
  O(1) problem data is invisible in double but corrupts float (see "Numerics" above for the
  precision derivation). The "which nonbasic needs an artificial bound" DECISION uses the
  ORIGINAL cost, not the perturbed one — see "Numerics" above; using the perturbed cost there
  was a real bug (`LP.lad`'s exactly-zero-cost columns could get a pointless artificial bound
  from perturbation noise alone).

- Default routing: `LPMethod.DualSimplex` runs dual; `RevisedSimplex` runs primal. Keep them
  user-selectable — the benchmark compares all four backends (tableau, revised primal, dual, IPM).

## Files

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.RevisedSimplex.fProxy.cs` — kernel layer
  (computational form, basis, factorization wrapper, FTRAN/BTRAN, eta file) + primal core.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.DualSimplex.fProxy.cs` — stage 2.
- `LPMethod` enum: add `RevisedSimplex` (and `DualSimplex` in stage 2) in the hand-written
  shared file where `LPMethod`/`LPStatus`/`LPInfo` live; add dispatch cases in `LP.solve`/`LP.lad`.
- Tests: `TemplateSourceTests/fProxy/LPTests.fProxy.cs` — follow the existing enum+dispatch
  pattern, `[BurstCompile(CompileSynchronously = true)]`, no `Assert.Fail` inside jobs
  (use `Assert.IsTrue` with `==` plus a Fail[] diagnostics array, as the existing tests do).

## Repo constraints (mandatory)

- Templates are the source of truth; `fProxy` token → float/double. Literal `double` is NOT a
  token. Regen+test: `Tools/run-tests.ps1 -Filter "*LP*"` (auto-regenerates). Do not hand-edit
  generated files.
- Proxy-typed parameters cannot have default values (CS1750) — use forwarding overloads.
- Job-safe: all scratch from `Allocator.Temp`, disposed before return; no managed allocations;
  no arena types inside the core.
- Do not modify `simplexCore`, `RatioTest`, `Pivot`, or the interior-point files.

## Acceptance criteria

1. Wyndor known-answer test passes (both new methods, both dtypes).
2. On the benchmark Section-1 random feasible LP family (same construction, several seeds,
   n ∈ {24, 48, 96}): objective matches tableau simplex within 1e-6 relative (double) /
   1e-3 (float); status Optimal.
3. A `≥`/mixed-sense instance (forces phase 1 / dual phase 1) solves correctly.
4. A degenerate instance (duplicated rows) terminates (no cycle) with the right objective.
5. `LP.lad` via the new method matches the simplex L1 residual.
6. Full LP test filter green: `Tools/run-tests.ps1 -Filter "*LP*"`.
