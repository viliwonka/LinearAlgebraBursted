# DEVLOG — TemplateSourceBenchmarks
Code comments state contracts only; history lives here (see CLAUDE.md).

## KMeansBenchmark
- 2026-07-12 | KMeansJobFProxy.Execute() called KMeans.fit with a hardcoded literal 16 while
  BenchFProxy sizes the centroids/workspace buffers from parameter K -- harmless only because the
  sole caller always passes K=16 (audit release-scan-2026-07-12/22-benchmarks finding 8, UNCERTAIN).
  Added a `K` field to the job and passed it through instead of the literal, so the requested cluster
  count and the buffer sizes are driven by one value. (was KMeansBenchmark.fProxy.cs:26)

## SparseSolverBenchmark
- 2026-07-12 | Dropped "Milestone B"/"Milestone-A" internal phase tags from three comments (the
  transpose-optimized CGLS/LSQR job banner, the Section 0b symmetric-vs-full spMV banner, and the
  Section-with-transpose-rows comment); kept the contracts (Aᵀ materialized once outside timing so
  ApplyT runs as a forward spMV; symmetric lower-triangle storage vs full block-CSR on the identical
  matrix). (was SparseSolverBenchmark.fProxy.cs:184, 647, 843)

## DirectSolveBenchmark
- 2026-07-12 | LuSolveTransAJobFProxy's comment dropped a perf-verdict/expectation clause ("this
  should run at roughly the same speed as the forward LU row -- any large gap would mean the
  right-looking TransA formulation isn't vectorising as intended"); kept the contract identifying
  the job as the getrs(trans='T') counterpart of LuSolveJobFProxy. (was
  DirectSolveBenchmark.fProxy.cs:39-41)

## LargeSparseBenchmark
- 2026-07-12 | Stripped remaining ticket tags and budget-bookkeeping narration missed by the previous
  pass (audit release-scan-2026-07-12/22-benchmarks finding 5): "Krylov R3"/"Krylov R3b" round labels
  on the SpCgJobFProxy/SpPcgJobFProxy tol-field comment, the SpPcgSSORJobFProxy comment, the LOBPCG
  Bench.Time wiring comment, and the SSOR-preconditioner-axis comment (also dropped its
  LobpcgAcceptsSSORPreconditioner test-name reference and "hypothesis under test" narration);
  BenchKrylovFProxy's comment dropped the "spMV x50 row DELETED... Q7 budget ruling" bookkeeping and
  the stale BenchPrecondConvergenceFProxy reference (that method was inlined, never renamed back in
  the comment); BenchStencilFProxy's comment dropped the "spec: BR=4/1.5% fill..." citation and the
  "gained PCG-SSOR... PAID FOR by dropping N=5120... Q7 budget ruling" history; BenchLobpcgFProxy's
  comment dropped the "Krylov R3b budget trade (spec §3b, disclosed)... DROPPED... pay for" narration.
  All now state only what each row set measures. (was LargeSparseBenchmark.fProxy.cs:25, 42, 73-81,
  90-97, 131-140, 238-242, 322-328)
- 2026-07-11 | Dropped all `docs/draft-spec-krylov-optimization.md` citations (round labels R3/R3b)
  across the file's header and section comments: the "iters+status makes breakdown-guard early-exits
  visible" hygiene note, the SpCgJobFProxy/SpPcgJobFProxy `tol`-field-reuse comment, the LOBPCG
  wall-clock-column comment, and the PCG preconditioner-axis section banner. Also trimmed the SSOR-for-
  LOBPCG hypothesis comment: dropped the "(R3, verified by ...)" and "(R3 finding)" round-number tags
  and a vague "(spec §3b/task brief)" aside, keeping the actual hypothesis (LOBPCG's per-iteration cost
  is Rayleigh-Ritz-dominated, so SSOR's iteration-count win might beat block-Jacobi's cheaper apply)
  and the fact that fProxySSOR routes through the generic `lobpcg<TOp,TPre>` core via
  fProxyBSROperator rather than a dedicated overload.

## LQRBenchmark
- 2026-07-12 | BuildInstanceFProxy's construction comment dropped its development history (the former
  fixed +-0.05 off-diagonal magnitude was only stable to n~12; at n=128 it produced unstable, likely
  unstabilizable instances; at n=4 the current 0.2/n scaling reproduces the original +-0.05 exactly).
  Kept the current contract (diagonal range, off-diagonal scaled 0.2/n, Gershgorin bound stays under
  ~0.6 at every n). (was LQRBenchmark.fProxy.cs:95-99)

## LPBenchmark
- 2026-07-12 | Trimmed three more comments to contract-only (audit release-scan-2026-07-12/22-benchmarks
  findings 1-3): the LadJobFProxy residual-recompute comment dropped its bug-postmortem clause (a
  not-quite-converged solve can report LPInfo.objective BELOW the true L1 residual, observed once as
  m=192 float revised printing 4.37 vs a true residual of 104.08, silently misleading the table); the
  LpRhsMatVecJobFProxy comment dropped the "coordinator's sanity-scan" workflow reference and the
  73728-multiply-add Mono-interpreted rationale; the SectionInfeasibleFProxy comment dropped the
  "(the same robust recipe the review that requested this section specified)" reviewer-provenance
  parenthetical. (was LPBenchmark.fProxy.cs:59-65, 174-179, 608-610)
- 2026-07-11 | Trimmed the killed-run anecdote from the per-job reporting-outputs comment. Full story:
  the report used to harvest objective/iters/status via a SEPARATE plain managed call to LP.solve/LP.lad
  before ever timing the Burst job -- i.e. every row solved the SAME problem TWICE, once fully
  Mono-interpreted. That was fine at n=24 but catastrophic at n=384 (seconds per solve): an extended
  benchmark run measured minutes and was killed because of it. The fix moved reporting into the timed
  job itself (objOut/itersOut/statusOut, length-1 arrays, written from inside Execute()), so
  Bench.Time's existing warmup + 4 timed .Run() calls populate them as a side effect of the SAME
  Burst-native call already being timed -- no second solve. Kept in-source: the current contract
  (self-reporting from inside Execute(), no second solve) and the enum-to-int cast note for
  LPBenchmarkFmt.InfeasRow. Also dropped dangling `docs/spec-revised-simplex.md stages 1+2` (Section 1
  banner) and `docs/spec-lpbasis-persistence.md acceptance item 5` (Section 6 banner) citations.
  (was LPBenchmark.fProxy.cs:24-35, 253, 310)

## KalmanBenchmark
- 2026-07-12 | Section 4's drift-safety comment (EkfCycleFProxy) was arithmetically wrong and
  cited the wrong regime: it claimed NonlinearSteps kept the cumulative step count "well inside"
  KalmanTests.fProxy.cs's 80-step EKF acceptance test, but NonlinearSteps=100 already exceeds 80
  per Execute() and Bench.Time's 1 warmup + 4 timed calls on the same persistent state push the
  true cumulative to ~500 -- and that acceptance test tracks a real noisy trajectory with seeded
  P, not this zero-measurement self-simulation, so it never validated this regime anyway. Rewrote
  the comment with the actual bound (~500 cumulative) and the real reason boundedness holds here:
  Smeas = H P Hᵀ + R stays positive as long as R > 0, independent of how far the self-simulated
  state/covariance drift, so the Cholesky-based innovation solve does not trip
  InnovationSolveFailed on this model. Noted it applies identically to UkfCycleFProxy, which had
  no drift note at all.
