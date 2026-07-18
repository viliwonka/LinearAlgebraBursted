# DEVLOG — MG (multigrid / AMG)

Code comments state contracts only; history lives here (see CLAUDE.md).

Building the unsmoothed nodal-aggregation AMG MVP (spec docs/dev/spec-multigrid-solver.md, the
post-research REVISED recommendation; user chose it 2026-07-18 over the geometric-first plan).
Path: elasticity-capable (rigid-body near-nullspace per aggregate), dodges general spGEMM (the
unsmoothed Galerkin RAP collapses to a deterministic segmented assembly), Chebyshev smoother
(already shipped), Flexible-CG outer solver. SA + deterministic spGEMM is the later Tier-2 end-state.

## AMG.aggregate
- 2026-07-18 | Deterministic greedy nodal aggregation (Vaněk–Mandel–Brezina), AMG-2. Strength
  ‖A_ij‖_F > θ·sqrt(‖A_ii‖_F·‖A_jj‖_F), θ=0 keeps all stored off-diagonals (PyAMG SA default).
  3 passes, all ascending block-row order, lowest-index tie-break, zero RNG. Contract: FULL storage
  input (asserts !Symmetric) with structurally-symmetric pattern — the hierarchy mirrors the user's
  A once at setup and every RAP coarse operator is assembled full, so iterating A's stored row-i
  neighbors covers the graph (no transpose/mirror needed inside aggregate). Pass-2 attachment reads
  a pass-1 membership snapshot (inP1) so attachments don't chain within the pass. Missing/zero
  diagonal block → diagNormF=0 → incident edges weak under θ>0 → node falls to a pass-3 singleton.
