# Release scan 2026-07-13 — N11: TemplateSourceTests, first half

Narrow scan of the FIRST HALF (68 of 136 files, sorted by full path) of
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests`, all dimensions at once, plus the
narrow-pass addendum patterns (pattern 7 — test-template comment-policy debt — applied heavily).

## Files covered

Partition = files 1-68 of the path-sorted list, i.e. everything from
`BoolAnalysisTests.cs` through `fProxy/QRCacheWorkspaceTests.fProxy.cs`:

- Root: BoolAnalysisTests, BoolBridgeTests, BoolDebugExportTests, BoolHashTests,
  BoolIndexingTests, BoolOperationsTest, BoolRandomTests, DebugInfoTests, RandomSharedTests
- ML/: PCATests.fProxy
- bool/: ArenaWiringTests.bool
- fProxy/: AccuracySweepTests, AnalysisTests, ArenaConversionsTests, ArenaHandleTests,
  ArenaWiringTests, BidiagTests, BidiagWorkspaceTests, BridgeFillTests, CHOPTests,
  CHOPWorkspaceTests, CHOTests, ChooseMarkerTests, CompMathTests, CompareTests,
  ConjugateGradientTests, ControlLQRTests, ControlTests, ConvergenceBudgetTests,
  DebugPrintTests, DotOperationTests, DotRefGuardTests, DotRefTests, EigenQRTests,
  EigenSymWorkspaceTests, EigenTests, FFTTests, FullStatsTests, GalleryPhase2Tests,
  GalleryTests, GeneratorTests, HashTests, HistogramTests, InPlaceOpTests, IndexingTests,
  InitTest, JacobiPrecondTests, KMeansTests, KalmanTests, KrylovFusedKernelTests,
  KrylovRound2Tests, KrylovVerifyAtExitTests, LOBPCGSmokeTests, LPTests, LQRPTests,
  LQWorkspaceTests, LUTests, LiteratureTests, MIPTests, MPCTests, MatrixMetricsTests,
  MultiRHSSolveTests, NLSTests, OperationsTest, OptimizeTests, QRCPDowndateTests,
  QRCPTests, QRCacheWorkspaceTests

Depth disclosure: 59 of the 68 files were read line-by-line in full. For the nine largest
remainder files (LQRPTests, LUTests, QRCPTests, QRCPDowndateTests, MPCTests, NLSTests,
MultiRHSSolveTests, OperationsTest, QRCacheWorkspaceTests) coverage was: full header/enum/
structure read + full-file keyword sweeps for every addendum pattern + enum-vs-driver
completeness check (all nine use `[TestCaseSource]`, so no dead-enum risk) + targeted reads
around every sweep hit. Logic-level line-by-line depth in those nine bodies is partial; a
follow-up could finish them, but all pattern hits found there are reported below.

Context read first: `TemplateConverter.cs` + `GenUtils.cs` (fProxy -> float,double; choose
markers index that order), and `TemplateSourceTests/DEVLOG.md` (used to verify which
comment-policy items were already relocated — several findings below are precisely the
copies that the recorded cleanups missed).

---

## HIGH

None. No test in the partition asserts something mathematically wrong, and no template
would generate incorrectly for the double variant. The two findings closest to HIGH are
M1/M2 (tests that silently never execute).

---

## MEDIUM

**M1 — CHOTests: `NotSPDStatus` test never runs (dead test).**
`fProxy/CHOTests.fProxy.cs:384-405` defines `NotSPDStatus()` and the enum case (line 76),
and `Execute` dispatches it — but this file drives cases via individual `[Test]` methods
(not `TestCaseSource`), and there is NO `[Test]` for `NotSPDStatus` (drivers list, lines
625-744, covers every other case). The only test pinning
`DirectSolveStatus.NotPositiveDefinite` from `CHO.decomp` is silently skipped.
Fix direction: add the missing driver (or switch the file to the `[TestCaseSource]` idiom
its siblings use).

**M2 — Dead `AssemblyTestJob` smoke structs (4 files in partition).**
`fProxy/CHOPTests.fProxy.cs:19-51`, `fProxy/EigenQRTests.fProxy.cs:27-39`,
`fProxy/LQRPTests.fProxy.cs:26-45`, `fProxy/QRCPTests.fProxy.cs:27-45` each define a
`[BurstCompile] AssemblyTestJob : IJob` that is never instantiated or `.Run()` anywhere
(also true of QRTests in N12's half). With `CompileSynchronously` these only compile when
run, so they provide zero coverage — dead code in every generated test assembly.
Fix direction: give each a one-line `[Test]` driver or delete them.

**M3 — "Solver API rework (commit 2)" / "Commit 2.5 (2a)" commit-ticket refs survive in
method-body comments.** The DEVLOG (2026-07-12, CHOTests entry) records dropping these from
the enum-case comments, but the method-level copies remain:
`fProxy/CHOTests.fProxy.cs:213, 249`; `fProxy/CHOPTests.fProxy.cs:70, 73, 471, 522`;
`fProxy/LUTests.fProxy.cs:45, 49, 1019`; `fProxy/QRCPTests.fProxy.cs:60, 374`.
Proposed DEVLOG entry:
`## CHOTests / CHOPTests / LUTests / QRCPTests`
`- 2026-07-13 | Dropped the remaining "Solver API rework (commit 2)" / "Commit 2.5 (2a)"
commit-ticket refs from method comments (the 07-12 pass only caught the enum-comment
copies). Contracts kept in place. (was CHOTests.fProxy.cs:213,249;
CHOPTests.fProxy.cs:70,73,471,522; LUTests.fProxy.cs:45,49,1019; QRCPTests.fProxy.cs:60,374)`

**M4 — "Stage-3 direct-solve-status coverage" internal stage label, 4 files.**
`fProxy/CHOTests.fProxy.cs:381`, `fProxy/CHOPTests.fProxy.cs:310`,
`fProxy/LUTests.fProxy.cs:381`, `fProxy/QRCPTests.fProxy.cs:530, 934`. Pattern-7 ticket
code; the sentence works without the label ("direct-solve-status coverage: ...").
DEVLOG relocation as in M3.

**M5 — R6a ticket code throughout KrylovVerifyAtExitTests (+ R1 in KrylovFusedKernelTests).**
`fProxy/KrylovVerifyAtExitTests.fProxy.cs:16, 62, 96, 169, 177, 223, 225` ("pre-R6a",
"the R6a contract", "every R6a-covered solver") and
`fProxy/KrylovFusedKernelTests.fProxy.cs:147-148` ("the shared-loop-bound bug the R1
review caught"). R6a is one of the exact codes the addendum names. The engineering content
(verify-at-exit recomputes the true residual at claimed convergence; the rectangular
updateXR loop-bound regression) is contract and stays; the round/review labels go to a
DEVLOG entry with `(was file:line)`. Also in the same file, lines 113-122:
debugging-methodology narration ("prototyping the recurrence in a throwaway dotnet console
app ... retuned ... via a throwaway diagnostic sweep run through Unity") — relocate to
DEVLOG, keep only the resulting contract (n/alpha/tol are tuned to this library's kernels;
float lies at this size, double does not, hence the `requireLie` choose-gate).

**M6 — FM2 ticket tags in ArenaHandleTests.**
`fProxy/ArenaHandleTests.fProxy.cs:7-8` ("labeled FM2 in project history, hence the 'FM2'
tags"), plus tags at lines 32, 129, 134, 139. FM2 is an addendum-named code; the DEVLOG
already holds the full postmortem (2026-07-11 entry), so the in-file label adds nothing a
stranger can use. Fix: drop the tags, state the defensive-copy contract plainly, add a
one-line DEVLOG note `(was ArenaHandleTests.fProxy.cs:7-8,32,129,134,139)`.

**M7 — "Stage E" label removed from fProxy ArenaWiringTests but not from the bool and
iProxy siblings (accidental sibling drift).**
`bool/ArenaWiringTests.bool.cs:341-342` and `iProxy/ArenaWiringTests.iProxy.cs:366-367`
still carry "Generational-overlay guard tests (Stage E; ...)" / "Stage E added a
checks-gated..."; the fProxy copy (fProxy/ArenaWiringTests.fProxy.cs:370-371) had exactly
this label dropped per the DEVLOG entry of 2026-07-12. Apply the same edit to both siblings.
(The iProxy file is N12's partition; reported here because the drift is only visible
against the fProxy sibling.)

**M8 — Agent-workflow references still present (4 spots).**
- `ML/PCATests.fProxy.cs:26` and `:206` — "the fable-caught trap" (agent name).
- `fProxy/ControlLQRTests.fProxy.cs:11` — "The coder's smoke tests live in
  ControlTests.fProxy.cs" (the mirrored reference in ControlTests' own header was cleaned
  on 07-12; this side was missed).
- `fProxy/QRCPDowndateTests.fProxy.cs:1034` — "reported out via Counts for the
  orchestrator" (residue of the ORCHESTRATOR-narration cleanup; if it means the managed
  driver, say "for the managed driver").
- `fProxy/LiteratureTests.fProxy.cs:13` — "See memory note literature-test-vectors."
  points at the agent's private memory file, meaningless to any human reader of the repo.
Fix direction: neutral wording, and delete the memory-note pointer.

**M9 — "per the spec" / spec-item labels (pattern 7), grouped.**
- `ML/PCATests.fProxy.cs:46, 531` ("per the spec", x2)
- `fProxy/LOBPCGSmokeTests.fProxy.cs:284` ("Oracle per the spec's suggested recipe")
- `fProxy/ControlLQRTests.fProxy.cs:30, 110-111, 131, 134-135` ("per spec", "the spec
  allows", "the spec's grid", "per the task's stabilizability guard")
- `fProxy/LPTests.fProxy.cs:911` ("Flagged as a stage-1 test gap in the original spec"),
  `:1420, 1422` ("the spec's 1e-6 rel"), `:1526` ("per the spec's item-2 wording"),
  `:1542` ("conditional per the spec"), `:1568` ("the spec-verified value")
- `fProxy/MIPTests.fProxy.cs:14` ("per the draft spec ...")
- `fProxy/AccuracySweepTests.fProxy.cs:166` ("exactly as the spec warned")
- `fProxy/ArenaWiringTests.fProxy.cs:129` ("the stronger ... check wanted by the spec")
In every case the surrounding sentence already states the actual rule; the spec citation is
removable without loss. One grouped DEVLOG entry per file, newest-first, per house format.

**M10 — Measured baselines / dated observations woven into comments (pattern 7), grouped.**
- `fProxy/LPTests.fProxy.cs:1123-1124` — "Observed margins are wide (2026-07-09): double
  warm 1 / cold 19, float warm 2 / cold 16" (a dated measurement; the assertion is only
  `warm < cold`).
- `fProxy/ControlLQRTests.fProxy.cs:232` — "(measured warm ~2 float / ~8 double,
  cold-recursion ~7/~13 ...)" (assertion is `<= 30`).
- `fProxy/MIPTests.fProxy.cs:663-671, 890, 957-961, 986-991` — stage-2/3/4 node-count
  measurement histories ("stage 2 = 267 nodes ... stage 3 = 241", "stage3 241 -> stage4
  (pre-cache) 218 -> stage4 (fProxyLPCache) 199", "double explores ~447 nodes") plus the
  float `nodes=0` anomaly postmortem. The single asserted number per test is contract and
  stays; the measurement lineage belongs in DEVLOG (partially already there in the
  07-11/07-12 MIPTests entries — extend those rather than duplicate).

**M11 — Bug-postmortem / iteration-narrative comments (pattern 7), grouped.**
- `fProxy/LOBPCGSmokeTests.fProxy.cs:293-300` — "a k=3 version of this exact setup was
  found, while iterating, to hit a rare numerical edge case ... worth a dedicated hardening
  follow-up, out of scope here" (debugging narrative + latent TODO; keep the k=2 contract,
  move the history and the follow-up idea to DEVLOG/tracker).
- `fProxy/LOBPCGSmokeTests.fProxy.cs:179-183` — "an earlier default X seed `(i+c*3+1)&3`
  repeats with period 4 ..." (history; the contract half — "a non-periodic deterministic
  fill is required to span all 6" — stays).
- `fProxy/LPTests.fProxy.cs:1015-1022` — "Reproduces a bug the benchmark caught:
  RevisedSimplex returned Optimal with 0 iterations ... a silent phase-1 bail" (postmortem;
  keep the regression contract, move the discovery story).
- `fProxy/LPTests.fProxy.cs:1648-1654` — "verified via instrumentation -- at m=1000 float
  ... only LP.lad(RevisedSimplex) returns MaxIterations" (methodology narration).
- `fProxy/FullStatsTests.fProxy.cs:10-12` — "previously-untested ... the n==2 case that
  used to read out of bounds (copy[-1])" — history, and the comment names the retired
  `StatsOP.` prefix while the code calls `Stats.` (double straggler).

**M12 — Rename-history comments (pattern 2 adjacent), grouped.**
- `fProxy/CompMathTests.fProxy.cs:461-464` — "(renamed from the kernel's old
  distance/distancesq names ...)".
- `fProxy/CompMathTests.fProxy.cs:508-509` — "Renamed from sincosInPlace - review flagged
  the InPlace suffix as misleading ..." (rename history + reviewer reference; the kept
  contract is simply "x is NOT mutated; s/c receive sin/cos").
- `fProxy/CHOTests.fProxy.cs:175-176` and `fProxy/ConjugateGradientTests.fProxy.cs:188-189`
  — "(choleskySolve(in A, ref L, ref b) was deleted -- it was a 2-line composition in
  disguise)" x2.
All to DEVLOG with `(was file:line)`.

**M13 — MatrixMetricsTests is named after a retired API.**
`fProxy/MatrixMetricsTests.fProxy.cs:12` — class `fProxyMatrixMetricsTests` (and the file
name) reference `MatrixMetrics`, which no longer exists as a class; every call in the file
is `Analysis.*` / `Norms.*`. Pattern-2 rename straggler (generated test-class names don't
ship in the UPM package, so MEDIUM not HIGH). Fix direction: rename file+class (e.g.
`AnalysisMetricsTests`) at a convenient regen point. Cross-partition note: the retired name
also survives in production XML docs (`TemplateSource/OP/QRCP.fProxy.cs`,
`TemplateSource/OP/LQRP.fProxy.cs` — "matching SVD.pinvSolve / MatrixMetrics.rank") and
`StatsOP.` survives in production exception messages
(`TemplateSource/Statistics/StatsCore.fProxy.cs:266+`) — those belong to the production
scanners (N1-N5/N10/W6) but are noted here since this partition's tests led to them.

**M14 — DotOperationTests: three "dot" tests assert only output SHAPE, not values.**
`fProxy/DotOperationTests.fProxy.cs:94-109 (MatVecDot), 183-198 (MatVecDotNonSquare),
200-215 (VecMatDotNonSquare)` — each computes `Blas.dot` on a random matrix with an
all-ones vector and asserts nothing but `b.N`. The names claim a dot-product test; a kernel
returning zeros (of the right length) would pass. Value coverage for mat-vec exists
indirectly (DotRefTests compares ref-vs-allocating — same kernel both sides — and solver
suites), but these three should assert the row-sum values they trivially have available.

---

## LOW

**L1 — Float-scale fixed tolerances in fProxy templates that also generate double.**
`BidiagTests` (1E-4f throughout), `EigenQRTests` (1E-2..1E-5), `FFTTests` (1E-3/1E-4
absolute), `MatrixMetricsTests` (1E-4/1E-5), `AnalysisTests` (1E-3..1E-6), `OperationsTest`
family. These pass in double but test double at float-grade slack, so a double-only
precision regression of several orders of magnitude would go unnoticed. The newer suites
show the house fix (per-type `Consts.fProxySqrtEps` scaling or `/*+choose[a|b]*/`).
Not a defect (loose, never wrong), but a systematic weakening of the double variant's
regression power. Fix direction: opportunistic migration to the sqrtEps/choose idiom.

**L2 — AnalysisTests degenerate random range.**
`fProxy/AnalysisTests.fProxy.cs:180` — `fProxyRandomDiagonalMat(dim, -1f, -1f)` (min ==
max == -1) produces a constant diagonal; almost certainly `(-1f, 1f)` intended. Test still
asserts correctly, just weaker than the written intent.

**L3 — Unused test-helper parameters.**
`fProxy/BidiagTests.fProxy.cs:63` (`AssertNearZero`'s `string context` never used) and
`:101` (`AssertBidiag`'s `ref Arena arena` never used).

**L4 — Small history/dev-speak leftovers (one-line fixes), grouped:**
- `BoolIndexingTests.cs:172-174` — "previously dereferenced a null arena core (the old
  _arenaPtr field, now the _arena handle's _core)".
- `fProxy/HistogramTests.fProxy.cs:22-23` — "the post-review fix under test"; `:307`
  "post-fix pins it".
- `fProxy/GalleryTests.fProxy.cs:191` — "(post-fix integer power)".
- `fProxy/InPlaceOpTests.fProxy.cs:27` — "(pre-fix, b would have been mutated instead)".
- `fProxy/GeneratorTests.fProxy.cs:140` — "(was entirely untested)".
- `fProxy/CHOTests.fProxy.cs:30-34` — "a measured crossover, not the naive 2*CHOL_BLOCK —
  see the size-gate comment at the call site" (perf-history in a test comment).
- `fProxy/AccuracySweepTests.fProxy.cs:32` — "so a reviewer can see the input really was
  ill-cond"; `:39` — "Landmine honored" (project jargon).
- `fProxy/KrylovFusedKernelTests.fProxy.cs:15-16` — "(user ruling: determinism required,
  bit-exactness not required)".
- `fProxy/LQWorkspaceTests.fProxy.cs:9` and `fProxy/QRCacheWorkspaceTests.fProxy.cs:9` —
  "Phase-2 solver-workspace tests" stage label.
- `fProxy/MPCTests.fProxy.cs:282` — "post-ship audit finding (2026-07-12, OP/DEVLOG.md)"
  (the DEVLOG pointer is right; drop the "post-ship audit" framing/date).

**L5 — MIPTests test-method names carry internal stage taxonomy.**
`Stage3NodesKnapsack6`, `Stage4NodesGomoryWolsey`, etc. (enum members become generated
NUnit test names). Tests don't ship in the package, so LOW; if the stage history moves to
DEVLOG (M10), feature-descriptive names (Pseudocost*/Propagation*) would replace the
dev-stage taxonomy.

**L6 — CompareTests relies on the hidden shared default seed.**
`fProxy/CompareTests.fProxy.cs:563-567 (MatMatEquals)` builds two "random" matrices with NO
seed argument and asserts them equal — this only holds because `fProxyRandomMat`'s default
seed is the fixed constant 121312 (ArenaExtensions.fProxy.cs:147). Deterministic, but a
future default-seed change breaks it mysteriously. Pass an explicit shared seed.

**L7 — FFTTests comments discuss float literally in the double variant.**
`fProxy/FFTTests.fProxy.cs:751-762, 809` — "at float32 the recurrence last-stage drift ...
eps_f ~ 1.2e-7" is generated verbatim into the double file, where the rationale text is
wrong even though the (loose) tolerances still pass. Reword per-type-neutrally.

**L8 — LiteratureTests `AssertBelow` arg order reads backwards at one call site.**
`fProxy/LiteratureTests.fProxy.cs:165` — `AssertBelow((fProxy)1E5, Analysis.cond(in H5))`
asserts `1E5 < cond` (correct: "cond is huge") but reads as "cond below 1e5". An
`AssertGreater` helper (as MatrixMetricsTests has) would read correctly.

**L9 — QRCPDowndateTests open-coverage note in a comment.**
`fProxy/QRCPDowndateTests.fProxy.cs:306-325` — "KNOWN, DISCLOSED LIMITATION ... A dedicated
Tier-E regression test for this ONE guard remains open coverage." The limitation disclosure
is fine; the open-coverage TODO belongs in DEVLOG/tracker, not a code comment.

**L10 — Mixed enum-member casing in the oldest files.**
`BoolAnalysisTests.cs` (`isDiagonal` vs `IsAllSame`), `fProxy/AnalysisTests.fProxy.cs`
(`isIdentity` vs `IsIdentityEpsilon`), plus `InPlaceOpTests`'s `DB_PoolChecks_...` method
name. Cosmetic, test-only.

---

## Areas confirmed clean

- Type gating: every `/*+choose[a|b]*/` block inspected resolves float|double in the
  correct order; DOUBLE-ONLY gating in MIPTests/LPTests uses `choose[0|1]` loop counts
  consistently; `//+deleteThis` in ArenaConversionsTests is well-formed; HashTests'
  uint-result funneling through `(int)` casts is deliberate and documented;
  ChooseMarkerTests pins the mechanism itself.
- Tests assert what their names claim in all files read in full, with the sole exceptions
  reported (M1, M14); oracles are genuinely independent (dense expansions, brute force,
  scipy/R/MIPLIB literature constants, transcribed unblocked sweeps) rather than
  self-comparisons — notably CHOP/LU blocked-vs-oracle, Kalman DARE fixed-point oracle
  with orientation discrimination, and MIP exhaustive enumeration.
- Numerical safety of the tests themselves: residual/oracle accumulation is done in double
  where it matters (AccuracySweep, ControlLQR Frobenius helpers, Kalman OracleGain); no
  unguarded divisions or sqrt-of-negative in test helpers; fixed seeds throughout (no
  flaky randomness found).
- Arena/memory discipline: every arena/Pivot/NativeArray allocated is disposed on all
  paths inspected (try/finally wherever Assert.Throws is involved); no operator-inside-job
  temp-alloc traps.
- Error handling: managed Assert.Throws guard tests consistently live on the main thread
  with the documented Burst rationale; no stale exception-type assertions found.
- Addendum patterns 1/2/4/5: no role-swapped InPlace usage, no maxIter/tolerance API-name
  stragglers at test call sites, no NoAlias duplicate-pointer misuse, no sibling-validation
  gaps in the guard-test files; no retired `Solvers.`/`BSM`/`_OP` usages in the partition.

## Summary

| Severity | Count |
|---|---|
| HIGH | 0 |
| MEDIUM | 14 (M1-M14; M3-M12 are grouped comment-policy findings with proposed DEVLOG relocations) |
| LOW | 10 (L1-L10, several grouped) |

Cross-partition handoffs: production `MatrixMetrics`/`StatsOP` doc stragglers and
`PCA.fitSvd(..., int maxIter, ...)` parameter name (TemplateSource/ML/PCA.fProxy.cs:316+)
to the production scanners; `iProxy/ArenaWiringTests.iProxy.cs:366-367` "Stage E" (M7) and
QRTests' dead `AssemblyTestJob` (M2) to N12.
