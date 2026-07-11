# DEVLOG — TemplateSourceBenchmarks
Code comments state contracts only; history lives here (see CLAUDE.md).

## LargeSparseBenchmark
- 2026-07-11 | Dropped all `docs/draft-spec-krylov-optimization.md` citations (round labels R3/R3b)
  across the file's header and section comments: the "iters+status makes breakdown-guard early-exits
  visible" hygiene note, the SpCgJobFProxy/SpPcgJobFProxy `tol`-field-reuse comment, the LOBPCG
  wall-clock-column comment, and the PCG preconditioner-axis section banner. Also trimmed the SSOR-for-
  LOBPCG hypothesis comment: dropped the "(R3, verified by ...)" and "(R3 finding)" round-number tags
  and a vague "(spec §3b/task brief)" aside, keeping the actual hypothesis (LOBPCG's per-iteration cost
  is Rayleigh-Ritz-dominated, so SSOR's iteration-count win might beat block-Jacobi's cheaper apply)
  and the fact that fProxySSOR routes through the generic `lobpcg<TOp,TPre>` core via
  fProxyBSROperator rather than a dedicated overload.

## LPBenchmark
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
