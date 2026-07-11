# DRAFT spec (for review): mixed-integer programming — branch & bound over the dual simplex

Status: DRAFT FOR USER REVIEW (2026-07-09). Nothing here is committed to; every section is
negotiable. Companion drafts: `draft-spec-qp.md`, `draft-spec-sparse-dual-simplex.md`.

## OPEN QUESTIONS FOR USER

1. **No cutting planes in v1 — acceptable?** v1 is branch & bound + domain propagation +
   pseudocost branching. Cuts (CMIR + lifted knapsack cover, the highest-value pair in HiGHS)
   are a well-scoped v2 (+~2 rounds). Pure B&B already solves small/medium instances exactly;
   cuts mostly buy node-count reductions on harder ones.
2. **Dense-only v1 — acceptable size ceiling?** v1 runs LP relaxations on the DENSE dual
   simplex (`DualSimplexCore`), so instances are bounded by what dense simplex handles
   comfortably (m, n in the low hundreds; thousands of B&B nodes). Scaling MIP beyond that is
   exactly what `draft-spec-sparse-dual-simplex.md` would later plug in — the two drafts
   compose: MIP calls whatever simplex backend exists.
3. **Warm-start basis API shape:** MIP needs basis reuse across LP re-solves. Proposed public
   surface: an `LPBasis` struct (status[N] + basis[m] `NativeArray<int/byte>`) + an
   `LP.solve(..., ref LPBasis basis)` overload that both consumes and returns it. This is
   useful standalone (any user re-solving perturbed LPs). OK as a public API, or keep
   internal-only for MIP?
4. **Integrality mask API:** `NativeArray<byte> integrality` (0 = continuous, 1 = integer)
   parallel to `c` — matching HiGHS's `integrality` array. Binary is just integer + [0,1]
   bounds. OK? (Explicit variable bounds also enter the public API here — the dense LP facade
   currently assumes x ≥ 0; MIP needs general `xl ≤ x ≤ xu`, which the computational form
   already supports internally.)
5. **Float:** templated `fProxy` per convention, but MIP correctness leans hard on tolerance
   stacking (integrality 1e-6, bound propagation roundoff). Recommendation: template it, test
   float only on tiny instances, document double as the serious dtype. OK?
6. **Determinism:** single-threaded with fixed tie-breaking → bit-deterministic node sequence
   (fits the library's determinism story). No parallel tree search planned at all. Confirm.

## What HiGHS actually does (verified)

HiGHS's MIP solver (by Leona Gottwald, `highs/mip/`) is **branch-and-cut**. Component map
verified from the source tree and headers (2026-07-09):

- **Search** (`HighsSearch.h`): pseudocost branching with a reliability mechanism
  (`markBranchingVarUpReliableAtNode`, `evalUnreliableBranchCands` — unreliable candidates get
  strong-branched with an LP-iteration cap via `selectBranchingCandidate`). Depth-first diving
  (`dive`, `solveDepthFirst`) with plunge-backtracking (`backtrackPlunge`) against a node
  queue; child selection rule enum: `kUp, kDown, kRootSol, kObj, kRandom, kBestCost,
  kWorstCost, kDisjunction, kHybridInferenceCost`.
- **Domain propagation** (`HighsDomain`, `HighsDomainChange`): activity-based bound
  tightening with an explicit bound-change stack, local domains per node.
- **Cuts** (`HighsCutGeneration.h`): single-row relaxations by bound substitution + row
  aggregation; `cmirCutGenerationHeuristic` (complemented MIR), `separateLiftedKnapsackCover`,
  `separateLiftedMixedBinaryCover`, `separateLiftedMixedIntegerCover`; separate tableau
  (Gomory), path, and mod-k separators; cut + conflict pools with aging.
- **Structure engines**: clique table, implications, conflict pool, redcost fixing,
  GF(k) solve, symmetry — all v2+/never for us.
- **Heuristics** (`HighsPrimalHeuristics`, `HighsSearch`): RINS/RENS neighbourhoods,
  feasibility jump, rounding.
- **LP relaxations** (`HighsLpRelaxation`): solved by the dual simplex, warm-started from the
  parent basis — the reason dual simplex is the MIP workhorse: after a branching bound change
  the parent basis stays DUAL feasible, so the child re-solve is a handful of dual pivots.

Judgment: the sensible v1 subset for this library is **B&B + propagation + pseudocost +
warm-started dual simplex + a rounding heuristic**, no cuts, no cliques/conflicts/symmetry,
no presolve. That subset is a complete, exact MIP solver (what commercial codes were in the
1990s); everything else is node-count optimization layered on top of it.

## v1 scope

`MIP.solve`: **min cᵀx s.t. A x {≤,=,≥} b, xl ≤ x ≤ xu, x_j ∈ ℤ for flagged j** — dense A,
`fProxy`-templated, job-safe, exact (proves optimality via bounds), returns incumbent +
bound + gap on any limit.

Explicitly deferred: cutting planes (v2: root-node CMIR + lifted knapsack cover =
"cut-and-branch"; full branch-and-cut with local cuts later still), presolve, clique
table/implications/conflict analysis, symmetry, RINS/RENS/feasibility-jump (v1 keeps only
rounding), parallel search, sparse LP relaxations (arrives free when the sparse dual simplex
exists), MIQP.

## Algorithm (v1)

Standard LP-based branch & bound:

1. **Root:** propagate bounds to fixpoint (capped rounds), solve the LP relaxation
   (`DualSimplexCore`). Integral → done. Infeasible → Infeasible.
2. **Node loop (DFS dive + best-bound queue):** take the current dive node; re-solve its LP
   **warm-started from the parent basis** (bound changes only — dual feasible start, few
   pivots). Prune if LP ≥ incumbent (within gap tolerances). If LP solution integral
   (|x_j − round(x_j)| ≤ 1e-6·max(1,|x_j|) for all flagged j): new incumbent, backtrack.
3. **Branch:** select the fractional variable by pseudocost score (product rule,
   score = max(pcUp,ε)·max(pcDown,ε)); a variable is *reliable* once it has ≥ RELIABLE=8
   observed branchings per direction — unreliable candidates get strong-branched (dual
   simplex with a small iteration cap, e.g. 100) exactly as `HighsSearch` does. Fallback
   before any history exists: most-fractional. Children: x_j ≤ ⌊v⌋ and x_j ≥ ⌈v⌉; dive into
   the child on the round-to-nearest side first (HiGHS default flavor of `kBestCost`; exact
   rule = tuning, fixed at implementation time for determinism).
4. **Propagate** in each child: activity-based row propagation (for each row, min/max
   activity from current local bounds → tighten variable bounds; integer bounds rounded).
   Empty domain → prune without an LP solve. Record every tightening on the bound-change
   stack so backtracking is an O(#changes) undo.
5. **Heuristic:** on each LP solve with few fractionals, try simple rounding + a
   feasibility check (one GEMV per row block); accept improving incumbents.
6. **Backtrack/terminate:** plunge-style — dive while the child LP bound stays within a
   factor of the best queue bound, else push the sibling to the best-bound queue and pop the
   queue's best. Terminate when queue empty (Optimal — incumbent is proven) or on limits
   (node cap, LP-iteration cap, gap ≤ absGap/relGap → `GapLimit` with certified bound).

Bounds/objective bookkeeping in `double` locals per the `simplexCore` convention; the LP
kernel itself stays `fProxy`.

## Reuse (existing kernels, by name)

- `DualSimplexCore` + `BuildComputationalForm` + the eta/`Ftran`/`Btran`/`Refactorize` layer —
  the entire LP engine, called per node. (The revised primal — `RevisedPrimalCore` — remains
  the cleanup path after perturbation removal, as today.)
- `LPInfo`/`LPStatus` — per-node LP result; new `MIPInfo` wraps the tree-level result.
- `Consts.fProxyEpsilon`-derived tolerances, inline (CS0111 trap noted in
  spec-revised-simplex.md).
- Random property-matrix generators + `fProxyGallery` for test instance construction.

## New infrastructure (genuinely required)

- **Warm-start basis API** (open question 3): `LPBasis` + a `DualSimplexCore` entry that
  accepts an initial (status[], basis[]) instead of the all-logical start, with a
  dual-feasibility repair (bound flips) on entry. This is the ONE change inside existing
  solver files — everything else is new files. Also independently useful (re-solve
  workflows), and the QP draft's phase 1 could use it later.
- **Bound-change stack**: `UnsafeList<BoundChange{ varIndex, isUpper, newBound, oldBound }>`
  + per-node marker — the undo log for DFS backtracking. `Allocator.Temp`.
- **Node queue**: binary heap over `UnsafeList` keyed by LP bound (best-bound), each entry
  storing its bound-change slice (replayed from root on activation — bounded by tree depth,
  avoids storing a basis per queued node; the DIVE path reuses the live basis, which is where
  warm starts actually pay).
- **Pseudocost tables**: 4 `fProxyN`-length arrays (up/down cost sums + counts).
- **`MIPInfo`/`MIPStatus`**: Optimal / Infeasible / Unbounded / GapLimit / NodeLimit /
  MaxIter + objective, dual bound, gap, nodes, total LP iterations — per the diag-struct
  conventions (fields only from already-computed numbers).

No new numerical machinery at all — the risk profile is bookkeeping correctness
(undo-log discipline), not numerics.

## Staged plan (coder-agent rounds) + oracles

- **Stage 1 — warm-start API (~0.5–1 round):** `LPBasis` + warm entry + repair. Oracle: solve
  an LP, tighten one bound, re-solve warm — identical objective to a cold solve, and assert
  iteration count strictly less (a real warm-start test, not just "it ran").
- **Stage 2 — B&B core (~1.5–2 rounds):** most-fractional branching, pure DFS, bound-change
  stack, no propagation/heuristics. Oracles: (a) small knapsacks with known optima
  (brute-forced at build time, n ≤ 20); (b) an assignment-problem LP (totally unimodular —
  MUST finish at the root with 0 branches: catches false fractionality); (c) the classic
  Gomory/Wolsey 2-variable textbook instance; (d) an infeasible and an unbounded MIP;
  (e) exhaustive cross-check vs enumeration on random tiny MIPs (n ≤ 12, both dtypes).
- **Stage 3 — pseudocost + reliability + queue/plunging (~1 round):** same oracle suite must
  stay green with node counts ≤ the stage-2 counts on the harder instances (assert it — that
  is the point of the feature); determinism test (two runs, identical node sequence).
- **Stage 4 — propagation + rounding heuristic + limits/gap (~1–1.5 rounds):** propagation
  correctness oracle = propagate then brute-force-verify no integer-feasible point was cut
  off (random tiny instances); MIPLIB tiny instances embedded as literals — e.g. `stein9`
  (9 binaries, optimum 5), `stein15` (15 binaries, optimum 9), `p0033` (33 binaries, optimum
  3089) — the standard public known-answer set at embeddable size; gap/node-limit behavior
  tests (incumbent + bound sane on early stop).
- **v2 (separate round set, ~+2 rounds):** root-only CMIR + lifted knapsack cover per
  `HighsCutGeneration`'s single-row-relaxation scheme, cut pool with aging.

Effort: **v1 ≈ 4.5–5.5 rounds ≈ 1.5–1.8× the dense simplex build**; +2 rounds for v2 root
cuts. The LP engine — normally most of a MIP solver's cost — is already built and hardened.

## Files

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.fProxy.cs` — facade + search core.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.Domain.fProxy.cs` — propagation +
  bound-change stack (separate file: it is the piece a future cut/conflict v2 grows around).
- `MIP.Info.cs` — `MIPInfo`/`MIPStatus` (hand-written shared, like `LP.Info.cs`).
- `LPBasis` + warm entry: inside `LP.DualSimplex.fProxy.cs`/`LP.Info.cs` (the one existing
  file touched).
- Tests: `TemplateSourceTests/fProxy/MIPTests.fProxy.cs`.
- Repo constraints: identical to spec-revised-simplex.md's list.

## References

- HiGHS `highs/mip/` source tree; `HighsSearch.h`, `HighsCutGeneration.h`, `HighsDomain` —
  verified 2026-07-09.
- Achterberg, *Constraint Integer Programming* (PhD thesis, 2007) — domain propagation +
  the SCIP architecture HiGHS's design follows; Achterberg, Koch & Martin, "Branching rules
  revisited" (OR Letters 33, 2005) — pseudocost + reliability branching.
- Marchand & Wolsey (2001) — MIR/CMIR cuts (v2); Letchford & Lodi — lifted cover cuts (v2).
- MIPLIB 2017 (miplib.zib.de) + the MIPLIB 3 "stein/p0033" family — known-answer instances.
