# DEVLOG — MG (multigrid / AMG)

Code comments state contracts only; history lives here (see CLAUDE.md).

Building the unsmoothed nodal-aggregation AMG MVP (spec docs/dev/spec-multigrid-solver.md, the
post-research REVISED recommendation; user chose it 2026-07-18 over the geometric-first plan).
Path: elasticity-capable (rigid-body near-nullspace per aggregate), dodges general spGEMM (the
unsmoothed Galerkin RAP collapses to a deterministic segmented assembly), Chebyshev smoother
(already shipped), Flexible-CG outer solver. SA + deterministic spGEMM is the later Tier-2 end-state.

## AMG K-cycle
- 2026-07-18 | K-cycle (Notay 2008 / AMGCL): AMGOptions.cycle = MGCycle.K. Each non-coarsest level's
  coarse correction is computed by TWO steps of Flexible CG on the coarse operator, preconditioned by
  the next level's K-cycle — the per-level Krylov acceleration that recovers grid-independence
  unsmoothed aggregation loses under a plain V-cycle. Recursive (KCycle calls itself twice per level);
  RECURSION COMPILES UNDER BURST (the geometric-spec's "avoid recursion" caution was unfounded here).
  Cost: the 2^level tree is bounded because per-level work shrinks faster than 2×, so total work stays
  O(N) with a larger constant than a V-cycle. 6 extra per-level scratch buffers (rc, c1, c2, v1, v2, e)
  allocated only when cycle==K. Breaks down gracefully to a single unaccelerated apply on non-positive
  curvature. A K-cycle is a VARIABLE operator → NOT pcg-valid: the symmetry guard moved from the
  preconditioner ctor to the pcg overloads (ctor now unguarded — valid for fcg); added
  Krylov.fcg(in fProxyBSR, in fProxyAMGPreconditioner, ...) rungs. IsCycleSymmetric = V && pre==post;
  IsKCycle exposed. Driven standalone by MG.solve(cycle=K) or as an fcg preconditioner. Tests:
  KCycleSolves, KCycleTighterThanV (K iters <= V iters), FcgKCycleConverges, AsymmetricCyclePcgRejected.
  🔴 CS8156/CS0206: a readonly method cannot pass a field's UnsafeList indexer (_P[l]/_Z[l]) by ref/in
  — copy to a local first (as VCycle already did).

## AMG review pass (3 adversarial reviewers) — fixes + hardening
- 2026-07-18 | Three code-review agents over the whole AMG track: verdict NO critical/major algorithm
  bug (V-cycle P/Pᵀ directions, residual signs, aliasing, coarse-factor reuse, job-safety, Galerkin
  indexing, builder determinism, fcg beta/aliasing, reconstruction identity all independently
  confirmed correct). Fixes applied:
  - 🔴 REAL BUG (aggregate `Strong`): for a ZERO-diagonal block, `theta*sqrt(0*d)==0` made every
    incident edge spuriously STRONG — the OPPOSITE of the documented "zero-diag → weak → singleton"
    (would pull a constraint/interface row into a real-DOF aggregate on saddle-point systems). Fix:
    `StrongNorm` returns false when either diagNormF is 0; also switched to `sqrt(di)*sqrt(dj)`
    (avoids float overflow of di*dj). θ=0 path unchanged. Regression test: ZeroDiagonalNodeIsolated.
  - pcg-validity footgun: nothing enforced pre==post (required for the AMG preconditioner to be SPD).
    Added `fProxyAMG.Pre/Post/IsCycleSymmetric`; the `fProxyAMGPreconditioner` ctor now throws on an
    asymmetric cycle. Tests: CycleSymmetryFlag + managed AsymmetricCyclePreconditionerThrows.
  - NaN-poison guard: a NotPositiveDefinite build returned a hierarchy whose coarse solve emits NaN.
    Added `_usable` flag; Solve/ApplyCycleFromZero throw. Test: NotPositiveDefiniteFailsCleanly.
  - setup leak on build-failure throw: wrapped the build in try/finally (ok-flag) that disposes the
    UnsafeList containers on the exception path (managed-only under Burst, matching the Arena ctor).
  - minor: pass-2 duplicate BlockFrobenius hoisted; prolongator dropped a redundant Bcoarse zeroing
    (arena.fProxyMat already clears) + added an absolute rank-collapse floor; singular-file crefs to
    a non-existent `fProxyAMG` type reworded to plain text.
  - Coverage tests added (were gaps): SingleLevelHierarchy (L==1), ExplicitOptions, ReuseAcrossRhs,
    ZeroRhs, WarmStart, MaxIterationsExit; strengthened PcgAmgBeatsPlainCg with a plain-CG residual
    check. FCG note: on a verify-fail-continue the PR beta pairs the true residual with the recursive
    r_old — tiny perturbation, typically helpful, no fix (reviewer-confirmed informational).

## fProxyAMG / MG.solve / fProxyAMGPreconditioner
- 2026-07-18 | AMG-5: the hierarchy + V-cycle + solve/preconditioner API wiring the four setup
  kernels together. Setup loop (arena.fProxyAMG): aggregate → tentativeProlongator (scalar B=ones,
  m=1) → galerkinRAP → per-level Chebyshev smoother, until a level ≤ coarseMax scalar unknowns /
  aggregation stops coarsening / maxLevels; coarsest = dense ToDense + CHO. ITERATIVE V-cycle
  (down: pre-smooth+restrict via spMVT(P); coarse: CHO solve; up: prolong via spMV(P)+post-smooth) —
  no recursion. MG.solve = outer V-cycle loop to relative residual; fProxyAMGPreconditioner = one
  symmetric cycle from zero (pcg-valid when pre==post) + 3-rung Krylov.pcg BSR overloads.
  STORAGE: fProxyAMG : IDisposable holds per-level handles in UnsafeList<> containers (freed by
  Dispose); the level DATA is arena-owned. No mutable scalar fields → IJob-copy safe. The arena has
  no generic array-of-handles slot, hence the container+Dispose approach (like fProxyBlockJacobi).
  🔴 TWO CODEGEN TRAPS caught at compile: (1) AMGOptions/AMGSetupInfo defined in a fProxy file →
  duplicated in float+double (CS0101); moved to a `//singularFile//` AMGOptions.cs with theta as
  `double` (not fProxy) — the SchwarzOptions pattern. (2) The `partial struct Arena` build method was
  in namespace LinearAlgebra.Sparse, but the real Arena is in `LinearAlgebra` — the phantom
  Sparse.Arena shadowed the real one for EVERY Sparse file (cascade of "Arena has no fProxyVec").
  Fix: Arena partial in namespace LinearAlgebra. Currently V-cycle only; K-cycle (fcg-accelerated per
  level, for grid-independence under unsmoothed aggregation) is the remaining enhancement.

## AMG.galerkinRAP
- 2026-07-18 | Unsmoothed Galerkin coarse operator A_c = TᵀAT, AMG-4 — the spGEMM-dodging kernel.
  Each fine block-row maps to ONE aggregate, so the triple product collapses to a segmented
  scatter-add: fine block A_ij contributes T[i]ᵀ A_ij T[j] (m×m) to coarse block (aggId[i],aggId[j]).
  Uses fProxyBSRBuilder duplicate-summation; deterministic because the builder's counting+insertion
  sort is stable and fine blocks are added in ascending (i,j) order → fixed sum order. Full-storage
  numAgg×numAgg output with m×m blocks. Contract: full-storage square-block A. Tested MATRIX-FREE via
  the Galerkin identity <v,A_c u>==<Tv,A(Tu)> (scalar m=1 and block BR=2/m=2), coarse symmetry, and
  determinism — no dense TᵀAT oracle needed. BSR.spMV confirmed to handle T's rectangular BR×m blocks.

## AMG.tentativeProlongator
- 2026-07-18 | Tentative (unsmoothed) prolongator T, AMG-3. Per aggregate: gather B's local rows,
  modified Gram–Schmidt (in place) → Q (T's BR x m blocks) + R (m x m) with B_local = Q·R, so
  T·Bcoarse == B exactly (the defining identity). T is a BSR, one BR x m block per fine block-row
  (col = aggId[i]), built via fProxyBSRBuilder. Bcoarse = block-stacked R. Rank-deficient aggregate
  column (norm collapses under MGS) → zero Q column + R=0. Default overload: B = ones (m=1).
  🔴 BUG CAUGHT BY TESTS (uninit memory): `new fProxyMxN(m, m, Allocator.Temp, true)` — the ctor's
  4th bool is `uninit`, NOT `clear`. R is upper-triangular; its never-written lower triangle was
  garbage and got copied into Bcoarse → reconstruction failed for m>=2 AND was NONDETERMINISTIC
  (m=1 has no lower triangle, so it passed — the tell). Fix: allocate R with uninit=false. Fully
  written buffers (L, blk) keep uninit=true. Lesson: fProxyMxN(...,bool) = uninit; pass false for
  any partially-written matrix.

## AMG.aggregate
- 2026-07-18 | Deterministic greedy nodal aggregation (Vaněk–Mandel–Brezina), AMG-2. Strength
  ‖A_ij‖_F > θ·sqrt(‖A_ii‖_F·‖A_jj‖_F), θ=0 keeps all stored off-diagonals (PyAMG SA default).
  3 passes, all ascending block-row order, lowest-index tie-break, zero RNG. Contract: FULL storage
  input (asserts !Symmetric) with structurally-symmetric pattern — the hierarchy mirrors the user's
  A once at setup and every RAP coarse operator is assembled full, so iterating A's stored row-i
  neighbors covers the graph (no transpose/mirror needed inside aggregate). Pass-2 attachment reads
  a pass-1 membership snapshot (inP1) so attachments don't chain within the pass. Missing/zero
  diagonal block → diagNormF=0 → incident edges weak under θ>0 → node falls to a pass-3 singleton.
