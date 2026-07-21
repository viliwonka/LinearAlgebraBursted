# DEVLOG — TemplateSource
Code comments state contracts only; history lives here (see CLAUDE.md).

## BurstProbe.cs
- 2026-07-21 | New singular (non-fProxy) file, `internal` so it's reachable from
  `BurstLinearAlgebra.Tests`/`BurstLinearAlgebra.Benchmarks` via the `InternalsVisibleTo` grants in
  AssemblyInfo.cs without adding public API surface to the shipped package. Placed here (flows
  through codegen as a verbatim copy to `Source/BurstProbe.cs`, same as `Assume.cs`) rather than
  hand-placed directly in `Source/`, because CLAUDE.md requires everything under `Source/` to be
  codegen output; this is the only location both the Tests and Benchmarks assemblies can reference
  without an asmdef change (neither references the other; both already reference the core `Source`
  assembly).
- 2026-07-21 | Verified end-to-end with a throwaway `SourceTests/BurstProbeSelfTest.cs` (deleted
  after use, per spec-test-burst-mono-hygiene.md's acceptance criterion): a `[BurstCompile]` job
  calling `RequireBursted()` passes clean (true no-op); a job with no `[BurstCompile]` attribute at
  all throws, and Unity reports it as `Unhandled log message: '[Exception]
  InvalidOperationException: Job ran under Mono...'`, which auto-fails the current NUnit test case --
  the same channel the project's existing in-job `Assert.IsTrue`-abort battery tests already rely on
  (no managed-side `Assert.Throws`/try-catch needed in a test).
- 2026-07-21 | Load-bearing finding from that same verification: a job's `Execute()` exception is
  NEVER rethrown synchronously to the `.Run()` caller in this Unity/Burst version -- confirmed by a
  plain `try/catch` around `.Run()` observing nothing, for both a `[BurstCompile]` job (with
  `DisableDirectCall = true`) and a plain non-attributed job, using both the
  `BurstCompiler.Options.EnableBurstCompilation` runtime toggle and the
  `--burst-disable-compilation` process-launch flag to force Mono. Unity only reports it via
  `Debug.LogException` on a later tick. This is fine for NUnit tests (which auto-fail on the stray
  log) but means `Bench.cs`'s benchmark harness cannot use try/catch at all -- see `RanUnderMono`
  (a plain static flag written just before the throw, unaffected by the logging quirk since it's an
  ordinary synchronous field write) and its use in `Assets/LinearAlgebra/Benchmarks/Bench.cs`.

## AssemblyInfo.cs
- 2026-07-11 | `InternalsVisibleTo("BurstLinearAlgebra.Tests")` exists so concrete (NOT codegen'd) test
  files like ChunkedRecordTableTests.cs can exercise internal-only building blocks (e.g.
  `LinearAlgebra.ChunkedRecordTable<TRecord>`, docs/dev/rfc-memory-model.md §4/§6.1/§7 step 2) directly,
  the same way ArenaLayoutTests.cs already exercises Arena's public surface. (was AssemblyInfo.cs:3-8)

## Consts.cs
- 2026-07-11 | LQ block gate (`floatLqBlockMinM`/`doubleLqBlockMinM`): measured on
  TallWideSolveBenchmark (A is k x 2k), blocked-vs-unblocked crossover. LQ's trailing-update fold is
  memory-reduction-bound and double streams 2x the bytes/element, so double stays bandwidth-starved
  (blocking pays off) only at a larger size: float wins from ~256 row-panels, double not until ~512.
  Gates are pinned conservatively (err high): below the gate the unblocked path is always correct, so
  a too-high gate only forgoes upside while a too-low gate can regress on a weaker cache; a worse CPU
  (smaller cache) crosses over earlier, so a high gate still captures its blocking win. (was Consts.cs:32-41)
- 2026-07-11 | Per-type level-3 blocking gates (QR/QRCP/Chol/LU): pinned from a same-session
  blocked-vs-unblocked sweep on the QR/Cholesky/LU/QRVariants benchmarks (each value = smallest swept
  size where blocked actually beat plain). float-vs-double ordering is NOT universal: QR/QRCP/Cholesky
  reconstruct-or-fold work is memory-reduction-bound, so double (2x bytes) crosses over LATER -> higher
  double gate; LU's trailing update is a proper GEMM where the UNBLOCKED path re-streams the trailing
  matrix, so double's 2x traffic hurts SOONER -> double crosses EARLIER (lower double gate than float)
  -- opposite ordering, caught only because we benched. Old shared gates (QR/QRCP 64, Cholesky/LU 256)
  were actively regressing double below its true crossover (double QR at N=64 was ~40% slower blocked;
  Cholesky double at 256 ~15% slower). Per-value measured results: floatQrBlockMinN=128 (float wins
  from 128, 64 ~neutral); doubleQrBlockMinN=512 (double loses <=256, wins from 512);
  floatQrcpBlockMinN=64 (float wins at every size); doubleQrcpBlockMinN=512 (double loses <=256, wins
  from 512); floatCholBlockMinN=1024 (float loses <=512, wins from 1024); doubleCholBlockMinN=512
  (double loses at 256, wins from 512); floatLuBlockMinN=256 (float loses at 128, wins from 256);
  doubleLuBlockMinN=128 (double wins from 128, GEMM update crosses earlier). (was Consts.cs:45-68)
- 2026-07-11 | CHOP (pivoted Cholesky, xPSTRF) blocked-path gate: measured on CholeskyBenchmark's
  face-off section (CHO vs CHOP vs LU, all decompInPlace) at N=256/512/1024, both dtypes, 1 warmup + 4
  timed runs. N=256 was noise-level (blocked vs unblocked ranges overlapped, no clear winner either
  dtype); N=512 showed a clear, non-overlapping win for both (float ~1.4%, double ~6.4%); N=1024
  widened further (float ~5.4%, double ~10.4%) -- win grows with N as expected once the level-3 SYRK
  trailing update dominates. Gate set at the first size with a clearly-real win, erring high per the
  size-gate convention used elsewhere. The win here is smaller than plain CHO's (~1.3-1.4x): the panel
  phase is intrinsically heavier than CHO's -- LAPACK's blocked PSTRF panel step is a left-looking
  full-row correction per column (not narrowed to the panel width), needed so pivot selection can see
  an up-to-date Schur-complement diagonal for the WHOLE trailing matrix -- see CHOP.decomp's
  blocked-path comment. (was Consts.cs:70-81)
- 2026-07-11 | `sweepBudget`'s scaling rationale cited docs/dev/spec-svd-eigen-convergence.md; citation
  dropped from the code comment, contract text kept in place. (was Consts.cs:85-96)
