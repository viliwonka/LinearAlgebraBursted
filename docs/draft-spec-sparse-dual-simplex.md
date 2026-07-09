# DRAFT spec (for review): sparse revised dual simplex (HEkk/HFactor lineage)

Status: DRAFT FOR USER REVIEW (2026-07-09). Nothing here is committed to; every section is
negotiable. Companion drafts: `draft-spec-qp.md`, `draft-spec-mip.md`.

## OPEN QUESTIONS FOR USER

1. **Is this worth building now at all?** Sparse LP is already served twice over: the
   matrix-free interior point (`LP.Sparse.fProxy.cs`, spec-sparse-lp.md) and PDLP
   (`PDLP.Sparse.fProxy.cs`). What sparse simplex uniquely adds: exact vertex solutions,
   true Infeasible/Unbounded certificates, and — the strategic one — **warm-startable LP
   re-solves at scale, which is what a sparse-capable MIP would run on**. If MIP is a
   long-term direction, this is its scaling engine; if not, this is the most deferrable of
   the three drafts. Both earlier docs (spec-sparse-lp.md §9, the landscape research §2a)
   explicitly listed sparse revised simplex as a non-goal *for those features* — building it
   is a deliberate reversal that should be a conscious decision.
2. **Storage grain:** simplex needs scalar column access to A (basis column gather, ratio
   test) AND scalar row access (pricing α_r = row combination). BSR blocks are the wrong
   grain. Proposal: the public API takes `fProxyBSR` (and dense), converted ONCE at entry to
   an internal scalar CSC + CSR working pair — no new public matrix type. OK?
3. **Update method staging:** v1 ships product-form (PF) eta updates first (the dense solver
   already proved the eta-file + REFACTOR_INTERVAL=64 discipline), Forrest–Tomlin as a later
   stage. HiGHS defaults to FT; PF costs more FTRAN/BTRAN work on long runs but is far
   simpler. Accept PF-first staging, or require FT in v1?
4. **Hyper-sparsity depth:** v1 does sparse-vector solves with a density-dispatch (sparse
   scatter path vs dense loop), NOT the full Gilbert–Peierls DFS reachability machinery of
   Hall & McKinnon. That costs some performance on genuinely hyper-sparse instances
   (network LPs) but roughly halves the kernel complexity. Acceptable v1 cut?
5. **Netlib fixtures:** validate against small netlib instances (afiro 27×32, sc50a/sc50b,
   adlittle, share2b, blend — public domain) embedded as C# literal arrays (a few KB each).
   OK to embed? (License-clean; it's the standard LP known-answer set.)
6. **Float:** template per convention, but a sparse simplex in float is genuinely fragile
   (Markowitz threshold pivoting exists precisely to trade stability for sparsity, and float
   has no headroom to trade). Proposal: template it, gate float tests to tiny instances,
   document double as the production dtype. Same posture as MIP. OK?
7. **Target scale:** first target m, n ~ 2–10k with nnz ~ 10⁴–10⁵ (where dense simplex dies
   on memory/time but the factor still fits comfortably)? Sets benchmark expectations.

## What HiGHS actually does (verified)

Verified 2026-07-09 from `highs/util/HFactor.h` and the `highs/simplex/` source tree:

- **HFactor** computes PBQ = LU in two phases: `buildSimple` (triangularization — peel row
  and column singletons, which on real LP bases eliminates most of the matrix without any
  fill) then `buildKernel` (Markowitz pivoting on the residual kernel: row/column nonzero
  counts maintained in linked lists — `col_link_first/next/last` etc. — pick the pivot
  minimizing (r−1)(c−1) subject to a threshold test |candidate| ≥ pivot_threshold·max|col|,
  with configurable `pivot_threshold`/`pivot_tolerance`). This is the Suhl–Suhl scheme.
- **Four update methods** behind one `update()` call: Forrest–Tomlin (`updateFT`, plus
  collective `updateCFT` for PAMI), product form (`updatePF`), modified product form
  (`updateMPF`), approximate product form (`updateAPF`) — the Huangfu–Hall 2015 paper's
  taxonomy, in code.
- **Hyper-sparsity**: `ftranCall`/`btranCall` take an `expected_density` argument and switch
  between sparse (index-list) and dense solve kernels per call; densities are tracked as
  running averages by the caller (HEkk). This is the Hall–McKinnon 2005 exploit.
- **Dual simplex** (`HEkkDual` + `HEkkDualRow` + `HEkkDualRHS` + `HEkkDualMulti`): DSE
  weights and infeasibility bookkeeping live in HEkkDualRHS; the BFRT + Harris pass is
  `HEkkDualRow::chooseFinal` (already studied line-by-line for the dense build — see
  spec-revised-simplex.md stage 2); `HEkkDualMulti` is PAMI (parallelism across multiple
  iterations, Huangfu–Hall 2018). PAMI is explicitly out of scope here (single-threaded
  determinism is a library invariant).

## v1 scope

`LP.solve` sparse overload routed to a **serial sparse revised dual simplex**: same
computational form, same algorithm layer as the dense dual simplex (DSE pricing, Harris +
BFRT dual ratio test, artificial-bounds dual phase 1, cost perturbation, primal cleanup) —
**the algorithm layer is a port, not a redesign** — over a new sparse kernel layer (CSC/CSR
+ sparse LU + eta updates + sparse FTRAN/BTRAN).

Explicitly deferred: Forrest–Tomlin (stage 5 here, or v2 — open question 3), full
Gilbert–Peierls hyper-sparse solves, PAMI/any parallelism, presolve, crossover (IPM→basis),
primal sparse simplex, MPF/APF updates, scaling beyond a Ruiz equilibration pass.

## Algorithm / kernel design (v1)

- **Working form:** entry converts the caller's `fProxyBSR` (or dense) into internal scalar
  CSC (values/rowIdx/colStart) + CSR mirror of [A | I-logicals-implicit]. Logical columns are
  implicit unit vectors (never stored). The BSR→CSC conversion is one O(nnz) pass over
  blocks (BSR iteration pattern already exists in `BSR.spMV`/`rowSquaredWeighted`).
- **Sparse LU (`SpLU`):** Suhl–Suhl as verified above — singleton peel, then Markowitz with
  linked-list counts and threshold pivoting (threshold 0.1 default, tightened on instability
  exactly like HFactor's `pivot_threshold`). Storage: L and U in compressed column/row form
  with a fill-estimate growth policy (`UnsafeList` doubling, `Allocator.Temp`). NOT built on
  the dense `LU` class — flagged honestly: **zero reuse from the existing dense
  factorizations; this subsystem is written from scratch.**
- **FTRAN/BTRAN:** sparse right-hand sides carried as (index list, values) pairs — the HVector
  pattern. Two solve kernels each (sparse-scatter when the rhs index list is short relative
  to m; dense loop otherwise), dispatch on a running density average per solve type
  (`expected_density`, the verified HFactor mechanism). Eta file application ports
  `ApplyEtaForward`/`ApplyEtaTransposed` to sparse etas (store only the nonzeros of α_q).
- **Pricing:** α_r = Nᵀρ_r computed hyper-sparsely: iterate the nonzeros of ρ_r, accumulate
  via CSR rows — O(nnz of touched rows), not O(mn). DSE weights stay a dense length-m array
  (they are dense by nature). Reduced-cost maintenance switches from the dense solver's
  recompute-by-BTRAN-each-iteration to INCREMENTAL updates (d_j ← d_j − θ·α_rj over the
  touched columns only) with periodic fresh recompute at refactorization — the dense code's
  always-fresh approach is exactly the O(mn) cost sparse exists to avoid.
- **Numerics:** everything `fProxy`; tolerance derivation inline (CS0111 trap); Ruiz
  equilibration pass at entry reusing the PDLP scaling pass structure (`pdlpScaledSolve`'s
  Ruiz loop — and note the PDLP lesson: Ruiz-only, no Pock–Chambolle pass); the two dense
  stage-2 lessons carry over verbatim (data-scale-relative artificial bound; original-cost
  decisions under perturbation).

## Reuse (existing, by name) — and the honest non-reuse

Reused: the ENTIRE dual simplex algorithm layer as a port template (`DualSimplexCore`,
`DualRatioTest`, `PriceRow`/`PriceReducedCosts` shapes, DSE update formulas, dual phase 1,
perturbation — spec-revised-simplex.md is the port source of truth); `BuildComputationalForm`
logic (bounds/senses); `LPInfo`/`LPStatus`/`LPMethod`; `fProxyBSR` as the public input type +
its block iteration patterns; PDLP's Ruiz pass; `RevisedPrimalCore` for post-perturbation
cleanup (needs the same sparse kernel underneath — the cleanup primal is part of stage 4).

NOT reused (flag): dense `LU`/`Blas` triangular solves — inapplicable to compressed sparse
factors. **The sparse LU + eta + sparse-solve subsystem (~stages 1–2, half this build) is new
infrastructure with no existing donor**, and its numerical quality gates everything above it.
This is the single biggest infrastructure item across all three drafts.

## Staged plan (coder-agent rounds) + oracles

- **Stage 1 — sparse working form + SpLU (~2–2.5 rounds):** CSC/CSR + BSR→CSC converter;
  singleton peel + Markowitz kernel + threshold pivoting; solve B x = b (no updates yet).
  Oracles: factor random sparse bases (fProxyRandomSparse generators exist) and LP-like bases
  (I-heavy + structural columns), compare solves against the dense `LU` on materialized B to
  factor tolerance; fill-in sanity (nnz(L+U) within a factor of MATLAB/scipy reference
  numbers recorded offline for fixed seeds); singular-basis detection (duplicate column →
  clean failure, feeds the refactorization repair path).
- **Stage 2 — FTRAN/BTRAN + PF etas + refactor control (~1.5–2 rounds):** sparse rhs
  carriers, density dispatch, sparse eta file, REFACTOR_INTERVAL + accuracy check (port the
  dense `RebuildXB` comparison). Oracle: **pivot-sequence replay** — run the DENSE dual
  simplex on an instance, record its (leaving, entering) sequence, force the sparse kernel
  through the same sequence, assert xB/y/d agree at every refactorization point. Bit-level
  agreement is not expected (different op order); factor-tolerance agreement is.
- **Stage 3 — dual simplex core port (~2–2.5 rounds):** pricing (incremental d, hyper-sparse
  α_r), DSE, BFRT walk (the flip-only-iteration trap from spec-revised-simplex.md is already
  documented — port the fixed logic), dual phase 1, perturbation + primal cleanup. Oracles:
  dense-agreement suite (materialize each sparse instance densely, objective equality vs
  `DualSimplex` within 1e-9 double / 1e-4 float relative); netlib smalls with published
  optima (afiro −464.753..., sc50a/sc50b, adlittle, share2b, blend); mixed-sense + degenerate
  instances from the dense acceptance list, sparsified; sparse-vs-IPM/PDLP objective
  agreement on the LargeSparseBenchmark generators.
- **Stage 4 — performance + hardening (~1.5–2 rounds):** density-dispatch tuning, benchmark
  section (vs sparse IPM and PDLP at 2k–10k, netlib mediums if feasible), float gating,
  Forrest–Tomlin update (`updateFT` port — U-row spike elimination + eta) IF open question 3
  answers "FT in v1"; otherwise FT becomes the v2 headline.

Effort: **~7.5–9.5 rounds ≈ 2.5–3× the dense simplex build** — the largest of the three
drafts, and unlike the other two it is mostly NEW kernel code rather than composition of
existing pieces. Roughly half the effort is stages 1–2 (the factorization subsystem), which
is also independently reusable (a sparse direct solve `SpLU.solve` falls out for free —
spec-sparse-lp.md §9 wanted exactly that as a future normal-equations preconditioner).

## Files

- `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/SpLU.fProxy.cs` — sparse LU + solves
  (+ its own tests file; deliberately a standalone public-ish subsystem, not LP-internal).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.SparseSimplex.fProxy.cs` — working form,
  eta file, FTRAN/BTRAN wrappers, dual core.
- `LPMethod`: no new member needed if routing is "sparse overload of `LP.solve` with
  `DualSimplex`"; sparse overload dispatch in the `LP` facade.
- Tests: `TemplateSourceTests/fProxy/SpLUTests.fProxy.cs` + additions to `LPTests.fProxy.cs`;
  netlib fixtures in a hand-written shared data file (not templated — they are data).
- Repo constraints: identical to spec-revised-simplex.md's list.

## References

- HiGHS `highs/util/HFactor.h`, `highs/simplex/HEkkDual*` — verified 2026-07-09.
- Huangfu & Hall, "Novel update techniques for the revised simplex method", Comput. Optim.
  Appl. 60(4), 2015 — FT/PF/MPF/APF taxonomy (the HFactor update methods, verbatim).
- Huangfu & Hall, "Parallelizing the dual revised simplex method", Math. Prog. Comp. 10(1),
  2018 — the HiGHS dual simplex system paper (PAMI; also the best overall description).
- Hall & McKinnon, "Hyper-sparsity in the revised simplex method and how to exploit it",
  Comput. Optim. Appl. 32(3), 2005.
- Suhl & Suhl, "Computing sparse LU factorizations for large-scale linear programming
  bases", ORSA J. Computing 2(4), 1990 — the SpLU design source.
- Forrest & Tomlin (1972) — the FT update; Koberstein's thesis (2005) — the dual simplex
  implementation bible, already used for the dense build.
- netlib LP test set (www.netlib.org/lp) — public-domain known-answer instances.
