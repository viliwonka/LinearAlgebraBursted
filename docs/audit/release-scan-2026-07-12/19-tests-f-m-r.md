# Release scan 2026-07-12 — area: tests-f-m-r

Scanned 18 template files (tests). Findings: total 5 — 5 confirmed, 0 uncertain, 0 unverified, 0 refuted; severity: 0 high, 2 medium, 3 low.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MIPTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MatrixMetricsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MultiRHSSolveTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/OperationsTest.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/OptimizeTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRCPDowndateTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRCPTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRCacheWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRLeastSquaresResidualTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QueryPredicateTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QueryTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/RandomMatrixTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/RandomTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/RandomWeightedTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ResampleTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/RollingWindowTests.fProxy.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRTests.fProxy.cs:318 — The 'blocked non-aligned' QR tests do not reach the blocked kernel: the real gate is floatQrBlockMinN=128 / doubleQrBlockMinN=512, not 64, so these tests exercise only the unblocked fallback (entirely so for the double build).

**Evidence**

```
Line 318: "Exercises the BLOCKED QR path (engaged when N_Cols >= 2*QR_BLOCK = 64)"
```

The cases feed n=65,70,96,100,127,150 (QRDecompBlockedNonAligned_130x65 ... _256x150) and n=64 (QRDecompPreservesABlocked, line 438: "engages the level-3 blocked path"). But QR.fProxy.cs:415 gates on `A_to_Q.N_Cols < Consts.fProxyQrBlockMinN`, and Consts.cs:42-43 set floatQrBlockMinN=128 / doubleQrBlockMinN=512. So for DOUBLE none of these n<512 reach the blocked path, and for FLOAT only n>=128 (just _256x150) does; the blocked non-aligned last-panel branch is never covered for double, and QRDecompPreservesABlocked(96x64) is unblocked for both types despite its name/comment.

**Verifier**

The generated Consts.cs (mirroring template intent per DEVLOG) sets floatQrBlockMinN=128 and doubleQrBlockMinN=512, and QR.float.cs / QR.double.cs both gate on those per-type constants (lines 419, 469). The QRDecompBlockedNonAligned cases feed n in {65,70,96,100,127,150}: for double NONE reach the blocked kernel, for float only n=150 does. QRDecompPreservesABlocked at n=64 is below both gates. Comments at lines 46, 318-319, 438 asserting "engaged when N_Cols >= 2*QR_BLOCK = 64" contradict the actual per-type gate, so the tests do not exercise the blocked short-last-panel branch they claim to, particularly for the double codegen expansion.

**Suggested fix**

Correct the comments to the actual per-type gate, and add blocked non-aligned shapes at n>=512 (double) / n>=128 (float) with n not a multiple of 32 so the blocked short-last-panel branch is genuinely exercised for both generated types.

### 2. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRCacheWorkspaceTests.fProxy.cs:13 — The cache-equivalence tests claimed to exercise the blocked compact-WY buffers never reach the blocked path, so the blocked cache overload's numeric equivalence is untested; the header comment also self-contradicts the guard-test comment.

**Evidence**

```
Line 13: "engage the level-3 BLOCKED (compact-WY) kernel once N_Cols >= 2*QR_BLOCK (= 64)"
```

DecompCacheGate(64), DecompCacheAbove(96), DecompCacheTall(72), DecompPreservingCache(80) and WorkspaceReuse(96, "Size >= 64 columns so those buffers are actually exercised") all use n<128. But lines 263-265 in the SAME file correctly state the gate is "Consts.floatQrBlockMinN=128 / doubleQrBlockMinN=512". Since the blocked path needs 128 (float) / 512 (double), no cache-equivalence test (max n=96) ever exercises the blocked WY buffers; only the mis-sized 512x512 guard tests touch that branch, and they check throwing, not numeric equivalence.

**Verifier**

Consts.cs shows floatQrBlockMinN=128 and doubleQrBlockMinN=512 after codegen substitution (template's fProxyQrBlockMinN=64 is a //+deleteThis stub). QR.fProxy.cs:415/465 gate on Consts.fProxyQrBlockMinN, so the generated float/double kernels require N_Cols>=128 / >=512 to enter the blocked WY path. Every DecompCacheEquiv/DecompPreservingCacheEquiv/WorkspaceReuse case (n in 64, 72, 80, 96) falls below both gates, so the blocked-WY buffers' numeric-equivalence path is never exercised; the 512x512 guard tests only assert throws for mis-sized buffers, not bit-identity. The header comment (line 13) also self-contradicts lines 263-265 in the same file, which correctly cite the per-type gate.

**Suggested fix**

Fix the header comment to the real gate and add a cache-equivalence case at n>=512 (double) / n>=128 (float) so the blocked WY-buffer path is validated bit-for-bit against the allocating overload for both types.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QRCPDowndateTests.fProxy.cs:102 — Code comments carry extensive dev-history / reviewer / mutation-testing narration and internal ticket refs, violating the project's contracts-only comment policy.

**Evidence**

```
line ~102 "ORCHESTRATOR DIAGNOSIS (mutation-testing found no test pinned...)"
line ~319 "ORCHESTRATOR VERIFICATION (this session, temporary instrumentation, since removed per OQ-D2 ...)"
line ~325 "KNOWN, DISCLOSED LIMITATION (found by an adversarial mutation-testing review pass)"
```

Plus OQ-D1/OQ-D2 ticket references. CLAUDE.md: code comments state contracts only; dev history, bug postmortems, reviewer/agent notes and spec/ticket refs belong in DEVLOG.md.

**Verifier**

Lines 102-122, 319-323, and 325-345 contain "ORCHESTRATOR DIAGNOSIS", "ORCHESTRATOR VERIFICATION", "KNOWN, DISCLOSED LIMITATION", explicit "mutation-testing" / "adversarial review pass" narration, an OQ-D2 ticket reference, and session-scoped instrumentation history. CLAUDE.md's strict comment policy states code comments carry contracts only and explicitly bans development history, bug postmortems, debugging narration, internal ticket refs (OQ-7 style), and reviewer/agent notes — every banned category is present here verbatim. The material belongs in a per-folder DEVLOG.md, not in the .cs file. Severity low is appropriate: no runtime defect, but a genuine, provable policy violation.

**Suggested fix**

Move the debugging narrative, mutation-testing verdicts and OQ-D1/OQ-D2 references into the folder DEVLOG.md; leave only the contract of what each case asserts.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MIPTests.fProxy.cs:96 — Enum/method comments contain dev history and reviewer/brief narration, violating the contracts-only comment policy.

**Evidence**

```
Line 96-99: "integrality classification at large magnitude (third-review regression)... the former RELATIVE integrality tol..."
Recurring "(The brief reported ... only double reproduced it.)" (e.g. lines 946, 833)
"Baselines were measured on both stages directly (by reverting to the stage-2 commit ...)" (lines 56-58)
```

These are development history / reviewer notes that the comment policy routes to DEVLOG.md.

**Verifier**

Verified all cited passages exist at the stated lines. Line 96 tags the enum entry "(third-review regression)" (reviewer-workflow narration); lines 97-99 give a bug postmortem ("the former RELATIVE integrality tol ... declared ... Optimal; fixed to HiGHS's absolute tolerance"); line 946 and line 832-833 both carry "(The brief reported ...)" reviewer-brief narration; lines 56-57 describe dev-debugging methodology ("Baselines were measured ... by reverting to the stage-2 commit, running a throwaway diagnostic, then restoring"). All four categories (dev history, bug postmortem, reviewer narration, STAGE-n internal references pervasive throughout) are explicitly enumerated in CLAUDE.md as content that must live in DEVLOG.md, not code comments. None of it is a contract statement, so the policy contradiction is direct.

**Suggested fix**

Trim to the assertion contract (instance, expected optimum/nodes/status) and relocate the review/brief/stage-commit history to DEVLOG.md.

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/RandomWeightedTests.fProxy.cs:14 — Stale codegen-artifact comment: 'One template expands to Rand / Rand' names the same type twice and no longer conveys the float/double split it intended.

**Evidence**

```
Line 14: "One template expands to Rand / Rand, so statistics use loose tolerances that hold for both precisions"
```

The two expansions are the float and double builds (both using the non-proxy Rand class); the duplicated 'Rand / Rand' reads as a substitution leftover.

**Verifier**

Line 14 of the fProxy template literally reads "expands to Rand / Rand" — the same identifier duplicated with a slash, which by itself carries no information about the float/double split (both expansions target the same shared `Rand` class). The float/double semantics only survive via the follow-on phrase "both precisions". This is a genuine low-severity naming/wording defect, not a functional bug. Note: it is not truly a substitution artifact — the identical convention is used deliberately in `iProxy/RandomTests.iProxy.cs:17` ("Rand / Rand / Rand"), so the reviewer's "substitution leftover" framing is inaccurate, but the underlying awkwardness they flag is real.

**Suggested fix**

Reword to 'expands to a float and a double build' (or drop the fragment) to avoid the duplicated type name.

## Scanner notes

Verified the blocked-QR gate against production: Consts.cs (floatQrBlockMinN=128 / doubleQrBlockMinN=512; the separate fProxyQrBlockMinN=64 literal is not what the generated QR code references) and QR.fProxy.cs:415/465 which gate on Consts.fProxyQrBlockMinN. The two blocked-path coverage findings are pass-but-mis-scoped: the equivalence assertions still hold (cache-vs-allocating and vs QR-LS both take the unblocked path), so no test goes red today; the defect is false coverage of the blocked compact-WY kernel, which is exactly the path the memory notes flag as having a T/Tᵀ landmine. The same stale '2*QR_BLOCK = 64' claim also appears in out-of-scope generated files (AccuracySweepTests.double.cs) and SmallSizeBenchmark.cs, indicating the constant change was not propagated to test comments library-wide. All other files in scope (MatrixMetrics, MultiRHSSolve, Operations, Optimize, QRCP, QRCPTests, QRLeastSquaresResidual, QRWorkspace, QueryPredicate, Query, RandomMatrix, Random, Resample, RollingWindow) read clean numerically; tolerances are per-precision (fProxySqrtEps/fProxyEpsilon-scaled), Arena/Pivot/Temp allocations are disposed on all paths including the Fail0 early-return NaN guards, and the exact-value oracles (Strang LS, Wolsey IP, MIPLIB stein/p0033, Moore-Penrose rank-1/rank-2) check out.
