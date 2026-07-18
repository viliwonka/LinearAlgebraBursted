# DEVLOG — MG (multigrid / AMG)

Code comments state contracts only; history lives here (see CLAUDE.md).

Building the unsmoothed nodal-aggregation AMG MVP (spec docs/dev/spec-multigrid-solver.md, the
post-research REVISED recommendation; user chose it 2026-07-18 over the geometric-first plan).
Path: elasticity-capable (rigid-body near-nullspace per aggregate), dodges general spGEMM (the
unsmoothed Galerkin RAP collapses to a deterministic segmented assembly), Chebyshev smoother
(already shipped), Flexible-CG outer solver. SA + deterministic spGEMM is the later Tier-2 end-state.

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
