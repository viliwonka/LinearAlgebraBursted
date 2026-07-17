# Spec: generic multigrid solver (geometric MVP → algebraic AMG)

Status: specced (Fable, 2026-07-18). Not implemented. This is a FOUNDATION spec for a multi-phase
feature — Phase 1 is a realistic first implementation unit; Phase 2 (AMG) is a separate, larger
project that this spec scopes but does not fully de-risk. Every unilateral design call is listed in
"Open questions for the user" at the bottom for veto.

User ask: "a generic multigrid solver, like AMG or matrix-free multigrid." Killer app:
MG-preconditioned CG (`Krylov.pcg` with a V-cycle as `TPre`) — optimal O(n) solves with
grid-independent iteration counts, where plain CG/PCG-Jacobi iteration counts grow with n.

## ⚠️ REVISED RECOMMENDATION (post-research 2026-07-18) — SUPERSEDES the phasing below; PENDING USER REVIEW

A web survey of MG flavours + the fastest implementations (hypre, AmgX, AMGCL, MueLu, PETSc GAMG,
deal.II, PyAMG) against THIS library's constraints (cross-arch determinism, SIMD-first, BSR, **no
spGEMM yet**, targets = structured Poisson AND **FEM elasticity / penalty-conditioned truss-frame**)
changes the recommended MVP. **The geometric-MG MVP below is DEMOTED to an optional throwaway** —
it validates the cycle/PCG plumbing but does NOT serve the elasticity target (vector problems whose
smooth error is the rigid-body near-nullspace; geometric MG and classical Ruge–Stüben AMG cannot
represent those, so iteration counts blow up under refinement).

**Recommended MVP: unsmoothed nodal-aggregation AMG, rigid-body near-nullspace tentative
prolongator, Chebyshev smoother, K-cycle, as a Flexible-CG (FCG) preconditioner.** Rationale:
- **Elasticity-capable**: the tentative prolongator `T` carries the rigid-body modes per aggregate
  (2D: 3, 3D: 6) — the decisive SA property — without SA's cost.
- **Dodges the missing spGEMM**: with *unsmoothed* `T` (piecewise-constant over aggregates), the
  Galerkin coarse operator `RAP = TᵀAT` collapses to a **segmented reduction over BSR nonzeros with
  a fixed accumulation order** (deterministic) — no general sparse matrix-matrix product needed.
  This is the single biggest reason to start here rather than SA.
- **Deterministic + SIMD smoother**: **Chebyshev(-Jacobi)** is pure spMV + AXPY + diagonal-scale
  (the modern default in PETSc GAMG / deal.II / MueLu precisely because Gauss–Seidel is sequential).
  Needs a per-level `λ_max(D⁻¹A)` estimate via the ∞-norm row-sum bound or a pinned-iteration
  Lanczos with fixed-order reductions. `ℓ1`-Jacobi is the robust fallback if the eigen-estimate is
  deferred.
- **K-cycle** (Krylov acceleration per level) recovers the grid-independence that unsmoothed
  aggregation loses under a plain V-cycle → the outer solver must be **Flexible CG**, not plain PCG
  (small, reusable addition; the preconditioner becomes variable/nonlinear).

**End-state (Tier 2): Smoothed Aggregation (SA) + Chebyshev + plain V-cycle**, once a **deterministic
spGEMM** (fixed merge order) exists — that is the one genuinely new subsystem, reusable beyond AMG.
Add smoothed prolongation `P = (I − ωD⁻¹A)T`; MVP and end-state share ~80% of the code (aggregation,
tentative `T`, Chebyshev, cycle driver, FCG); only spGEMM + prolongator-smoothing differ.

**No-regret building blocks to add FIRST** (useful standalone, headless-testable, independent of the
flavour verdict): (1) **Chebyshev(-Jacobi) smoother/preconditioner** — also a valid `pcg` `TPre`;
(2) **Flexible CG (`fcg`)** — a clean variant of the existing `cg`/`pcg`. These de-risk the AMG and
stand on their own.

Model the implementation on **AMGCL** (header-only, fast, cleanly factored coarsen/relax/cycle
policies — maps onto BSR); read **PyAMG** for algorithm clarity; take the `ℓ1`-smoother idea from
**hypre** and default Chebyshev parameters from **PETSc GAMG / deal.II**. Sources in the research
appendix at the bottom of this file.

Everything from "## Phasing decision" down is Fable's original geometric-first plan, KEPT as the
throwaway-MVP reference and because its cycle-driver / smoother / transfer / SolveInfo / Arena-safe
hierarchy design carries over verbatim to the aggregation MVP.

## Phasing decision

- **Phase 1 (MVP): matrix-free GEOMETRIC multigrid** for the scalar Dirichlet Laplacian / Poisson
  problem on structured 2D/3D grids. Small, fully deterministic, no new sparse kernels, exercises
  the whole cycle machinery (smoothers, transfers, coarse solve, V/W cycles, standalone +
  preconditioner API). Validates the signature property (grid-independent convergence) against the
  existing gallery Laplacian.
- **Phase 2: smoothed-aggregation AMG** over general BSR. The big one — needs BSR spGEMM (new
  kernel), strength/aggregation/prolongation setup, Galerkin RAP. Targets the matrices geometric MG
  cannot touch: `fProxyPenalizedGrid3D` truss stiffness (vector-valued, 3 dof/node), general
  user-assembled BSR.
- **Phase 3 (optional, unscheduled): extensions** — red-black smoothing for parallelism,
  variable-coefficient stencils, FMG/F-cycle, semicoarsening. Listed so they aren't accidentally
  designed out; none block v1.0 thinking.

Rejected phasing alternatives:
- AMG-first: rejected. spGEMM + deterministic aggregation is weeks of kernel work before the first
  convergence test can run; geometric MVP proves the cycle/API/test harness first, and its cycle
  driver, coarse solve, SolveInfo plumbing, and preconditioner wrapper carry over.
- Matrix-free AMG (stenciled Galerkin operators): rejected — a research topic, not a library feature.
- Geometric MG on `fProxyPenalizedGrid3D` in MVP: rejected. It is a vector-valued (3 dof/node) truss
  stiffness with penalty-pinned DOFs; scalar full-weighting/linear transfers are wrong for it
  (near-nullspace = rigid translations, not constants). It is the motivating Phase 2 AMG test case.

---

## Phase 1 — matrix-free geometric multigrid (MVP)

### Problem class and grid

Scalar SPD system A x = b where A is the 5-point (2D) / 7-point (3D) Dirichlet Laplacian stencil on
a grid of nx×ny(×nz) INTERIOR points, unit stencil weights (diag 4 / 6, off-diag −1), matching the
assembled `Gallery.fProxyLaplacian2D` convention (no h² scaling — MG convergence is scale-invariant,
and this keeps the BSR cross-check trivial). Right-hand side arbitrary.

Coarsening: standard per-axis factor-2 vertex coarsening. For a Dirichlet interior grid with n
points per axis, the coarse grid has (n−1)/2 points, well-defined iff n is odd. Setup rule:

- every axis dim must be odd and ≥ 3 (throw otherwise; doc recommends n = 2^k − 1);
- coarsen all axes simultaneously while every dim is odd and > 3 and total unknowns > `coarseMax`;
- stop → coarsest level, solved directly.

Rejected: requiring n = 2^k − 1 exactly (needlessly restrictive — any odd dims work, coarsening just
stops earlier and the direct solve gets a bit bigger); semicoarsening (anisotropy handling — Phase 3);
cell-centered coarsening (vertex-centered matches the Dirichlet-interior gallery convention).

Level operators by REDISCRETIZATION (same unit stencil on each coarse grid), not Galerkin RAP.
For the constant-coefficient Laplacian with full weighting + linear prolongation these agree up to a
scalar factor and both give textbook MG convergence; rediscretization needs no assembled matrices at
all. Galerkin is Phase 2's job. (Consequence, documented in the preconditioner contract: the MVP is
correct for THIS operator family only — it is not a generic-BSR preconditioner.)

### Smoothers (all matrix-free stencil kernels, fixed deterministic order)

- **Damped Jacobi**: x += ω·D⁻¹·(b − A x), D = constant (4 or 6). Needs one scratch vector per
  level (the residual). Default ω = 2/3 (classic; 2D-optimal 4/5 is an options knob, not a default —
  one default across 2D/3D is simpler and W(1,1)/V(1,1) with SGS is the primary path anyway).
- **Symmetric Gauss–Seidel (default)**: forward lexicographic sweep (i,j,k ascending, row-major
  x-fastest — one fixed order, stated in the kernel doc) then backward sweep (exact reverse).
  In-place, no scratch, sequential by nature. Deterministic because the order is fixed.
- **Red-black GS: DEFERRED to Phase 3.** It buys job-parallel smoothing (each color is independent)
  and is deterministic if the color order is fixed (all red, then all black, each in lexicographic
  order), but it doubles the kernel count, changes the smoothing factor, and the MVP is
  single-threaded-in-a-job like every other solver here. Not needed to prove the feature.

Symmetry note (required for PCG validity, see preconditioner below): the V-cycle defines a linear
operator B ≈ A⁻¹; PCG needs B symmetric. Guaranteed by construction: ν₁ pre-sweeps and ν₂ = ν₁
post-sweeps with the post-smoother being the ADJOINT of the pre-smoother — for Jacobi (symmetric)
any counts work; for GS the pre-smooth is forward-then-backward (SGS) and post-smooth
backward-then-forward (reversed SGS), plus symmetric transfers (full-weighting = prolongationᵀ up to
the standard factor). The cycle driver hard-codes this pairing; it is not user-configurable
independently (a config that silently breaks PCG is a trap, not a feature).

### Grid transfers (matrix-free, no assembled P/R)

- **Restriction: full weighting.** 2D 9-point stencil [1 2 1; 2 4 2; 1 2 1]/16, 3D 27-point
  analogue (tensor product [1 2 1]/4 per axis). Dirichlet boundary: neighbors outside the interior
  grid contribute zero (the stencil just reads a zero halo — implemented as bounds-checked reads or
  an interior/boundary loop split, coder's choice, order fixed lexicographic over coarse points).
- **Prolongation: (bi/tri)linear interpolation**, the transpose of full weighting scaled by the
  standard factor (2D: ×4, 3D: ×8). Coarse point coincident with fine point → direct injection of
  the value; fine points between → 2-, 4-, or 8-point averages. Loop order fixed lexicographic over
  fine points; prolongated correction is ADDED into the fine x (e is a correction).
- Rejected: injection restriction (loses the O(h²) transfer accuracy needed for mesh-independent
  rates); cubic interpolation (unnecessary for a 2nd-order operator).

### Coarse solve

At setup, assemble the coarsest-level Laplacian into a dense `fProxyMxN` (arena) and factor once via
`CHO.decompInPlace` (SPD by construction). Per cycle: copy coarse residual → coarse e, two
triangular solves. `coarseMax` default 128 unknowns (options knob) — small enough that the O(n³)
factor and O(n²) per-cycle solves are noise, large enough that 3–5 levels cover practical grids.
Rejected: iterative coarse solve (CG at the bottom) — nondeterministic iteration counts leak into
the outer operator and break the fixed-linear-operator property PCG needs.

### Cycle driver

V-cycle and W-cycle, selected by `cycle` option (γ = 1 / 2). Implemented ITERATIVELY with an
explicit per-level visit-counter schedule (down/up loops for V; a small counter array drives W) —
no recursion (avoids any Burst recursion risk and makes the state layout explicit). F-cycle and FMG
deferred (Phase 3); the visit-counter driver extends to them without restructuring.

Defaults: cycle = V, ν₁ = ν₂ = 1, smoother = SGS. (V(1,1)-SGS is the standard workhorse; W-cycle is
for verification and tough cases.)

### Data structure + Arena

New struct `fProxyGMG` (one per problem shape), built ONCE on the main thread by an arena extension,
then used from jobs. Contents, all arena-owned (no Dispose of its own, same pattern as `fProxySSOR`):

- level count, per-level dims (nx,ny,nz) — fixed-size fields or a small int buffer;
- per level: x (solution/correction), b (rhs/restricted residual), r (residual scratch) vectors
  (`fProxyN`), fine level aliases the caller's x/b only during solve — levels 1..L own their buffers;
- coarsest: dense factored Cholesky matrix + one scratch vector;
- options snapshot (cycle, ν₁, ν₂, ω, smoother enum) as plain readonly fields.

Job-safety rule (lesson from the warm-state audit, [[job-struct-copy-warmstate-audit]]): NO mutable
scalar fields on the struct — all mutable state lives behind the native buffers, so an IJob struct
copy is harmless. The struct is `readonly` where possible; setup returns it by value.

Setup API (arena authoring, main thread):

```csharp
fProxyGMG mg = arena.fProxyGMG2D(nx, ny);                    // defaults
fProxyGMG mg = arena.fProxyGMG2D(nx, ny, in MGOptions o);    // options struct
fProxyGMG mg = arena.fProxyGMG3D(nx, ny, nz /*, in o*/);
```

`MGOptions` is a plain struct: `cycle` (enum V/W), `pre`, `post` (ints; post is validated == pre,
see symmetry note), `omega` (fProxy), `smoother` (enum Jacobi/SGS), `coarseMax` (int). Options live
on SETUP, not on solve — the solve ladder keeps only `maxIter`/`tol` like every other solver
(short-param rule).

### Solve API (standalone) — new static class `MG`

```csharp
// full-control primitive (zero-alloc; all workspace inside mg)
SolveInfo MG.solve(in fProxyGMG mg, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol);
// default ladder, matching Krylov: maxIter = 50 cycles, tol = Consts.fProxySqrtEps
SolveInfo MG.solve(in fProxyGMG mg, in fProxyN b, ref fProxyN x);
```

Semantics identical to `Krylov.cg` conventions: x is a warm-startable initial guess, overwritten;
`tol` is relative (converged when ‖b − Ax‖ ≤ tol·‖b‖, threshold compared as squared norms); zero-b
short-circuit copies b into x (NaN-sanitizing, same rationale as cg); size/aliasing guards up front
(b/x distinct, dims match mg's fine grid). `maxIter` counts CYCLES. Returns the shared `SolveInfo`
(rnorm/iterations/status, implicit bool). Each cycle starts from a freshly computed fine-grid
residual, so the convergence test is on a true residual every cycle — no verify-at-exit pass needed
(cg only needs one because its r is recursively updated). Status: Converged / MaxIterations;
Breakdown unused in Phase 1 (nothing can break down: fixed stencil, Cholesky checked at setup).

### Preconditioner API (the killer app)

```csharp
public readonly struct fProxyGMGPreconditioner : IfProxyPreconditioner
{
    // Apply(in r, ref z): z = one V-cycle on A z = r with zero initial guess.
    // Symmetric + positive definite by the symmetric-cycle construction above -> valid for pcg.
}
```

Zero initial guess is REQUIRED for the preconditioner to be a fixed linear operator (warm-starting
inside a preconditioner makes M change per iteration — breaks PCG). One cycle regardless of the
`cycle` option? No — it applies exactly one cycle of the CONFIGURED type (V or W); what must be
fixed is the count (always 1), not the shape.

Matrix-free A for the pcg call: new operator structs

```csharp
public readonly struct fProxyStencilLaplace2D : IfProxyLinearOperator   // 5-point apply
public readonly struct fProxyStencilLaplace3D : IfProxyLinearOperator   // 7-point apply
```

Apply = stencil matvec (fixed lexicographic loop), ApplyT = Apply (symmetric), ApplyDot composes
Apply + `Blas.dot`, ApplyBlock = per-row fallback loop (LOBPCG is not a target here). These make the
existing generic `Krylov.pcg<TOp,TPre>` work with zero new pcg code:

```csharp
var A = new fProxyStencilLaplace2D(nx, ny);
var M = new fProxyGMGPreconditioner(in mg);
var info = Krylov.pcg(in A, in M, in b, ref x, maxIter, tol);
```

Also works with assembled BSR A (`fProxyBSROperator`) + MG preconditioner when the caller has both —
the preconditioner never inspects A. Concrete convenience overloads
`Krylov.pcg(in fProxyBSR A, in fProxyGMGPreconditioner M, ...)` follow the existing per-preconditioner
overload pattern (BlockJacobi/SSOR/IC0 precedent) — 3-step ladder each, mechanical.

### Determinism (Phase 1)

Operations used: + − * / and sqrt (norms) only — no transcendentals, no DetMath involvement. All
sweep/transfer/stencil loops have ONE fixed lexicographic order each, stated in the kernel docs.
Reductions: only the convergence-test `Blas.dot` (existing kernel, deterministic per arch;
cross-arch under the same pre-1.0 reordering waiver as every other dot in the library). Under
FloatMode.Strict the elementwise kernels are cross-arch deterministic; SIMD is allowed where it
does not reassociate. No RNG anywhere in Phase 1. Same input → bit-identical output per arch,
suitable for the determinism conformance harness as a new A-group later.

### File layout (Phase 1)

New template folder `Assets/LinearAlgebra/CodeGen/TemplateSource/MG/`:

- `MG.fProxy.cs` — cycle driver + `MG.solve` ladder.
- `fProxyGMG.cs` — hierarchy struct + `MGOptions` + arena builders (`fProxyGMG2D/3D`).
- `MG.Smooth.fProxy.cs` — Jacobi/SGS stencil sweep kernels.
- `MG.Transfer.fProxy.cs` — full-weighting restriction + linear prolongation kernels.
- `MG.Stencil.fProxy.cs` — `fProxyStencilLaplace2D/3D` operators (+ raw stencil apply kernel shared
  with the smoothers' residual computation).
- `fProxyGMGPreconditioner.cs`.
- `DEVLOG.md`.

pcg convenience overloads go in the existing `OP/Krylov.fProxy.cs` next to the SSOR/IC0 ones.
Codegen: fProxy → float/double only (inherently-ℝ per the integer-surface policy). Tests templated
under the SourceTests template area; one benchmark file in the Benchmarks template area.

### Tests + acceptance (Phase 1)

1. **Stencil-vs-gallery cross-check**: `fProxyStencilLaplace2D.Apply` vs `fProxyLaplacian2D` BSR
   spMV on random vectors, tol 1e-6 float / 1e-12 double (not bit — different accumulation orders).
2. **Exactness at coarsest**: grid small enough for zero MG levels → solve == direct Cholesky.
3. **Grid-independent convergence (the signature)**: standalone V(1,1)-SGS on 2D Poisson,
   sizes 31², 63², 127² (3D: 15³, 31³): cycles to ‖r‖ ≤ 1e-8·‖b‖ must be ≈ constant (assert a cap,
   e.g. ≤ 12 cycles at every size, and assert counts across sizes differ by ≤ 2).
4. **MG-PCG grid independence**: pcg with `fProxyGMGPreconditioner` on the same series: iteration
   count ≤ 15 at every size AND far below plain-cg's count at the largest size.
5. **Preconditioner symmetry** (PCG validity): for random r₁, r₂: dot(M⁻¹r₁, r₂) == dot(r₁, M⁻¹r₂)
   to solver precision — catches any future ν₁/ν₂ or sweep-order regression.
6. **W-cycle + Jacobi paths**: same convergence tests, looser caps.
7. **Determinism smoke**: two identical solves → bit-identical x (per arch).
8. **Through-an-IJob test** (LOBPCG cache-copy lesson): run solve + pcg inside a Burst IJob and
   compare against main-thread results.

---

## Phase 2 — algebraic multigrid over BSR (smoothed aggregation)

### Variant decision: smoothed aggregation (SA), not classical Ruge–Stüben

Chosen: **smoothed aggregation** (Vaněk–Mandel–Brezina 1996; PyAMG's `smoothed_aggregation_solver`
as the behavioral reference). Justification, in order of weight:

1. **Block-native.** SA aggregates NODES (block-rows) and its coarse block size = the near-nullspace
   dimension — a perfect fit for the existing BSR type and its unrolled B1–B6 kernels. RS coarsening
   is scalar-point-based; blockifying it is awkward and nonstandard.
2. **Elasticity/truss is the actual target.** `fProxyPenalizedGrid3D` and any FEM stiffness need the
   rigid-body near-nullspace to coarsen well; SA takes near-nullspace vectors as a first-class input.
   RS on elasticity is known-weak without extensions.
3. **Deterministic implementation is simpler.** Greedy aggregation in fixed row order with
   index-based tie-breaks is straightforwardly deterministic; RS's two-pass C/F splitting with
   dynamic-priority queues takes real care to make order-independent.
4. Setup cost is lower and the algorithm surface is smaller (no separate interpolation-formula
   zoo).

Rejected: classical RS (above); aggregation-only AMG without smoothing P (worse convergence,
kept as a debug/stepping-stone mode `smoothP=false` since tentative-P must exist anyway).

### Pipeline (per level, until coarse size ≤ coarseMax)

1. **Strength of connection** (block): block i,j strong iff ‖A_ij‖_F > θ·sqrt(‖A_ii‖_F·‖A_jj‖_F).
   θ default 0 (keep all — PyAMG's SA default); options knob. Output: strength graph as an int
   CSR (reuses builder machinery or plain int buffers).
2. **Aggregation** (standard greedy, deterministic): pass 1 in ascending block-row order — an
   unaggregated node whose strong neighbors are all unaggregated seeds a new aggregate (itself +
   those neighbors); pass 2 — remaining nodes attach to the neighboring aggregate with the
   strongest connection, ties broken by LOWEST aggregate index; pass 3 — leftovers form singleton
   aggregates. Every loop ascending index order; zero RNG.
3. **Tentative prolongation P₀** from near-nullspace B (n × m dense, row-major): restrict B's rows
   to each aggregate, thin-QR the aggregate-local block (existing dense QR at ≤ aggregate-size × m,
   Temp), Q → P₀'s block column, R → coarse-level B. Coarse block size = m. Default B: the constant
   vector (m=1, scalar problems). For elasticity the caller passes rigid translations (m=3 for the
   truss — rotations optional later; API takes any m ≤ 6).
4. **Prolongation smoothing**: P = (I − ω·D⁻¹A)·P₀ with ω = 4/(3·ρ̂), ρ̂ an upper bound on
   ρ(D⁻¹A) by GERSHGORIN row sums (fully deterministic, no iteration, no RNG). Rejected for now:
   power-iteration ρ estimate (tighter → slightly better ω, but introduces an iteration count and a
   seed vector into setup; revisit as an option if convergence disappoints).
   `smoothP=false` skips this step (aggregation-only mode).
5. **Galerkin coarse operator**: A_c = Pᵀ A P. Needs **BSR spGEMM** — the one genuinely new large
   kernel (see below). R = Pᵀ via the existing `SparseOP.Transpose`.
6. **Smoother setup per level**: reuse the EXISTING `fProxyBlockJacobi` + `BSR.sweepLower/sweepUpper`
   (= block SGS, the same pieces `fProxySSOR` composes) — pre-smooth forward+backward, post-smooth
   backward+forward, same symmetry construction as Phase 1. Damped block-Jacobi as the alternative.
7. **Coarsest**: assemble to dense (existing BSR→dense export or a small gather), `CHO.decompInPlace`.

### BSR spGEMM (new kernel, the main cost of Phase 2)

`SparseOP.spGEMM(in fProxyBSR A, in fProxyBSR B, ref Arena) → fProxyBSR`, block-compatible dims
(A.BC == B.BR). Row-by-row Gustavson with a dense scalar accumulator row of block slots (workspace:
one BlockCols-sized marker array + one block-row accumulator, reused across rows — setup is
main-thread arena authoring, so Temp workspace is fine). DETERMINISM RULE: for each output row,
merge contributions in ascending k (A's column order) and emit output blocks in ascending column
order (sort the touched-column list per row — fixed comparison, no hash iteration order anywhere).
Two passes (count then fill) or grow-and-compact via the existing builder — coder's choice, but the
accumulation ORDER is contract. RAP = spGEMM(spGEMM(R, A), P) — triple-product fusion is an
optimization for later, not the MVP of Phase 2.

### Data structure + API (Phase 2)

`fProxyAMG` hierarchy struct: per level A_l (BSR), P_l (BSR), block-Jacobi data, level vectors
(x/b/r), coarsest dense Cholesky — all arena-owned, same no-mutable-scalar-fields rule. Setup:

```csharp
fProxyAMG amg = arena.fProxyAMG(in A);                          // scalar default (B = ones, m=1)
fProxyAMG amg = arena.fProxyAMG(in A, in Bnear, in AMGOptions o); // Bnear: n x m near-nullspace
```

`AMGOptions`: `theta`, `cycle`, `pre`/`post`, `smoother`, `coarseMax`, `maxLevels`, `smoothP`.
Solve + preconditioner mirror Phase 1 exactly: `MG.solve(in fProxyAMG, in b, ref x, maxIter, tol)`
ladder returning `SolveInfo`, and `fProxyAMGPreconditioner : IfProxyPreconditioner` (one cycle,
zero guess) + concrete `Krylov.pcg(in fProxyBSR, in fProxyAMGPreconditioner, ...)` overloads.
Setup can FAIL usefully (aggregation degenerates, coarse Cholesky not SPD): setup returns the
struct plus a status out-param or a small `AMGSetupInfo` (levels built, coarse size, status) —
follow the solver-diag-struct preference (dedicated Info struct, fields only from already-computed
numbers).

### Phase 2 sub-milestones (each independently landable)

- 2a: BSR spGEMM + tests (associativity/against-dense oracles, determinism).
- 2b: scalar SA-AMG (m=1, constant B), unsmoothed P — convergence on gallery Laplacian2D vs
  Phase 1 GMG as the oracle (AMG should be within ~2x of GMG cycle counts on the same problem).
- 2c: smoothed P + Gershgorin ω; grid-independence test series.
- 2d: block near-nullspace (m=3) → `fProxyPenalizedGrid3D` MG-PCG beating the current best
  (SSOR/IC0-PCG) iteration counts; this is the acceptance headline for Phase 2.

### Determinism (Phase 2)

Setup: all graph passes in ascending index order, index tie-breaks, Gershgorin (no RNG, no
transcendentals), spGEMM ordered-merge contract above. Solve: fixed sweep orders as Phase 1.
Everything stays in + − * / sqrt. Same input → bit-identical hierarchy and solution per arch.

---

## Open questions for the user (every unilaterally-resolved fork)

1. **Class name `MG`** (vs `Multigrid`) for the static solver class; struct names `fProxyGMG` /
   `fProxyAMG`. Veto/rename freely — nothing depends on it yet.
2. **SA over Ruge–Stüben** for Phase 2 (reasons above). If you specifically want classical AMG
   semantics, say so before 2a starts — the setup pipeline differs substantially.
3. **Rediscretization (not Galerkin) on Phase 1 coarse levels** — correct for the constant-coefficient
   Laplacian family only; the MVP preconditioner is therefore Poisson-specific by contract.
   Acceptable for an MVP whose generic path is Phase 2?
4. **Red-black smoothing deferred to Phase 3** (kept sequential lexicographic SGS). Deterministic
   parallel smoothing is available later at the cost of extra kernels.
5. **Grid restriction: all dims odd, ≥ 3** (recommend 2^k − 1), even dims throw at setup. OK, or do
   you want even dims supported via a padded/one-sided boundary transfer (more kernel cases)?
6. **Options on setup, `maxIter`/`tol` only on solve** — matches the short-param ruling and keeps
   the ladder flat. The alternative (per-solve cycle/sweep params) was rejected as ladder bloat.
7. **Preconditioner = exactly one cycle, zero initial guess, symmetric-cycle hard-coded** (ν₂ = ν₁
   enforced, adjoint post-smoother) so pcg validity is structural, not user-maintained.
8. **θ default 0** (keep all connections) for SA strength — PyAMG's SA default; anisotropy tuning is
   the user's knob.
9. **Gershgorin ρ̂** for prolongation smoothing (deterministic, no RNG) vs tighter power iteration —
   revisit only if 2c convergence disappoints.
10. **Near-nullspace API**: dense n×m `fProxyMxN` input, default constant vector; truss demo passes
    3 rigid translations. Rotations (m=6) deliberately out of scope for 2d.
11. **Float-only concerns**: none specific — but note the LOBPCG float lesson: Phase 2 setup QRs are
    tiny (aggregate-local) and should be fine in float; if float SA-AMG misbehaves on the penalty
    truss, the fallback position is documenting double-recommended for AMG setup.
12. **Scope check**: Phase 1 is roughly one focused implementation cycle (kernels are small and
    dense-free); Phase 2 is a multi-week track gated on spGEMM. Confirm Phase 1 is worth landing
    before any Phase 2 investment.

## Reference notes

- Existing seams reused: `IfProxyLinearOperator` / `IfProxyPreconditioner`
  (`TemplateSource/Interfaces/LinearOperator.fProxy.cs`), `Krylov.pcg<TOp,TPre>` generic core +
  per-preconditioner concrete-overload pattern (`TemplateSource/OP/Krylov.fProxy.cs`),
  `SolveInfo`/`IterativeSolveStatus` (`TemplateSource/OP/SolveInfo.cs`), `fProxyBlockJacobi` +
  `BSR.sweepLower/sweepUpper` (`TemplateSource/Sparse/SparseOP.fProxy.cs:293`), `fProxySSOR`'s
  arena-composition pattern, `SparseOP.Transpose`, `CHO.decompInPlace`,
  `Gallery.fProxyLaplacian2D` / `Gallery.fProxyPenalizedGrid3D`
  (`TemplateSource/Sparse/Gallery.Sparse.fProxy.cs`).
- Literature: Briggs–Henson–McCormick "A Multigrid Tutorial" 2nd ed. (Phase 1 algorithms verbatim);
  Vaněk–Mandel–Brezina 1996 (SA); PyAMG `smoothed_aggregation_solver` defaults (behavioral
  reference); Tatebe 1993 (MG as PCG preconditioner, symmetry requirement).

## Research appendix (flavour survey 2026-07-18) — sources

Empirical/perf claims behind the "REVISED RECOMMENDATION" section:
- AMG survey (flavour taxonomy): https://multigrid.org/xu/paper/xu2017algebraic.pdf
- Adaptive AMG for structural mechanics / rigid-body near-nullspace: https://arxiv.org/pdf/1902.01715
- PETSc GAMG (SA, Chebyshev default, near-nullspace): https://web.cels.anl.gov/projects/petsc/vault/petsc-3.20/docs/manualpages/PC/PCGAMG.html
- DOLFINx elasticity-AMG demo (6 rigid-body modes): https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html
- Adams et al., polynomial vs Gauss–Seidel smoothing (JCP 2003): https://ui.adsabs.harvard.edu/abs/2003JCoPh.188..593A/abstract
- Baker/Falgout et al., ℓ1 smoothers for ultra-parallel MG: https://www.osti.gov/servlets/purl/1117969
- deal.II matrix-free Chebyshev GMG: https://arxiv.org/pdf/1910.13247
- MueLu User's Guide (SA level-build, near-nullspace API): https://trilinos.github.io/pdfs/mueluguide.pdf
- hypre BoomerAMG docs (ℓ1-GS/Jacobi, coarsening): https://github.com/hypre-space/hypre/blob/master/src/docs/usr-manual/solvers-boomeramg.rst
- AMGCL paper + CPU benchmarks (model reference): https://arxiv.org/pdf/1811.05704 ; https://amgcl.readthedocs.io/en/latest/benchmarks.html
- Notay AGMG — unsmoothed aggregation + K-cycle (ETNA 2010): https://etna.ricam.oeaw.ac.at/vol.37.2010/pp123-146.dir/pp123-146.pdf
- Sparse triple product (RAP) in MG: https://arxiv.org/pdf/1905.08423
- PyAMG (algorithm reference): https://github.com/pyamg/pyamg

Bottom line: determinism + no-spGEMM + elasticity ⇒ **unsmoothed nodal aggregation + RBM tentative
prolongator + Chebyshev + K-cycle as an FCG preconditioner** (MVP), → **SA + Chebyshev + V-cycle**
once a deterministic spGEMM exists.
