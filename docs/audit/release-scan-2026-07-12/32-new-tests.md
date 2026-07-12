# Release scan 2026-07-12 — area: new-tests (post-scan code)

{"total":3,"confirmed":1,"uncertain":0,"unverified":0,"refuted":2,"high":0,"medium":0,"low":1}

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KalmanTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/UKFTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MPCTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/NLSTests.fProxy.cs

## Findings

### 1. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/NLSTests.fProxy.cs:425 — RepeatedCallsNoLeak comment claims a guarantee (Unity catches the leak) that does not hold for the Allocator.Temp the code under test uses

**Evidence**

```
Lines 424-426: `// exercises the internal Allocator.Temp scratch's allocate/dispose cycle
repeatedly (a leaked buffer would be caught by Unity's collections checks well before
iteration 30).` nlsSolve/curveFit allocate their scratch from Allocator.Temp
(NLS.fProxy.cs lines 224-239), and Temp is a rewindable bump allocator whose un-disposed
handles are NOT reliably surfaced by the safety checks the way Persistent/TempJob leaks
are. The test still catches cross-call state corruption via the per-rep AssertClose, but
its stated leak-detection rationale is false.
```

**Verifier**

Traced the claim end-to-end. NLSTests.fProxy.cs:424-426 explicitly attributes the test's leak-catching power to "Unity's collections checks", but the code under test (NLS.fProxy.cs:224-239) allocates every scratch buffer from Allocator.Temp, and the test runs inside a `[BurstCompile(CompileSynchronously=true)]` IJob (NLSTests.fProxy.cs:101-102, invoked at line 540 via `.Run()`). Two concrete reasons the stated rationale does not hold:

1. Under Burst, DisposeSentinel is stripped (Burst cannot execute managed finalizers), so the leak-warning mechanism that would normally surface an un-disposed NativeContainer never fires for allocations made inside the Burst job body.
2. Even non-Burst Allocator.Temp leak detection is documented as weaker than Persistent/TempJob: Temp is a thread-local rewindable bump allocator whose leaks are surfaced (if at all) at frame boundaries / TempMemoryScope disposal, not mid-loop. All 30 iterations execute inside one synchronous IJob.Run() on one thread with no frame boundary between them, so a hypothetical missed Dispose inside curveFit has no opportunity to trigger a warning during the loop.

The per-iteration AssertClose does still verify state stability across repeated calls, but leak surfacing is not the actual mechanism. No DEVLOG entry (OP/DEVLOG.md, TemplateSourceTests/DEVLOG.md) documents this wording as a considered decision. Severity is correctly low: comment-accuracy only, no code defect. Corroborated against `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NLS.fProxy.cs` lines 224-239.

**Suggested fix**

Either drop the leak claim from the comment (call it a repeated-call state-stability test) or reproduce it with a TempJob/Persistent scratch path where leak detection genuinely fires.

## Refuted

| # | file:line | category | summary | why refuted |
|---|-----------|----------|---------|-------------|
| 1 | Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/UKFTests.fProxy.cs:58 | numerical | UKF-vs-linear-KF agreement tolerances (1e-2f\|1e-6) are 4-9 orders looser than the file's own measured agreement (~1.9e-6 float / ~3.6e-15 double), so a materially degraded UKF would still pass the "strongest correctness oracle" | By-design and documented in-code. Lines 50-56 carry an explicit rationale: wide safety margins chosen deliberately after the steadyStateGain episode where a too-tight tolerance was calibrated against a since-fixed bug. The 1e-2f\|1e-6 choose-marker matches the sibling KalmanTests.fProxy.cs JacTol at line 71. The same code path has tighter oracles elsewhere (AssertExactSymPosDiag bit-exact per step in NegativeW0StaysSymPSD, 300 steps, and UKFTracksPendulum, 80 steps), and the 60-step accumulation means a per-step degradation compounds past 1e-6 rather than sitting statically below it. Intentional design decision, not a defect. |
| 2 | Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MPCTests.fProxy.cs:125 | logical | The "scipy oracle" saturation tests only verify u0 equals its own input bound (-2.0 == uLo), which any clamping solver satisfies, so the assertion does not discriminate the condensed-QP interior solve | The narrow observation is factually true for that single assertion (scipy oracle -1.999999998939275 rounds to the bound), but the claim that "the interior QP is never exercised" is refuted by three other tests in the same file: UnconstrainedMatchesLQR (lines 76-109) checks u0 against an independent LQR closed form across 10 random x0 at LqrU0Tol()=5e-3f/1e-6 — a clamp-only or broken-Γ/H solver fails there; SoftRowUnavoidableMinimalViolation (lines 189-206) asserts maxSlackViolation ≈ 0.6, a non-bound numerical answer; SoftRowInactiveMatchesNoSoftBuild (lines 133-158) asserts plain and soft builds match away from any bound. DEVLOG (TemplateSource/OP/DEVLOG.md:161-165) documents the battery as deliberate — the oracle coinciding with the bound is a property of the problem (the true optimum is at the bound), not a testing mistake. The saturation test additionally still catches sign errors, wrong statuses, NaN/Inf, and out-of-bound outputs. |

## Scanner notes

Read all 4 test files in full plus the implementations they exercise (Kalman.fProxy.cs, Kalman.UKF.fProxy.cs, MPC.fProxy.cs, NLS.fProxy.cs) and the Info structs (Kalman.Info.cs, MPC.Info.cs, NLS.Info.cs) to verify contracts, oracles, and dispose paths. These four files are, on the whole, carefully written; no high/medium defect could be substantiated. Specifically verified as CORRECT (not bugs):

1. KalmanTests OracleGain is a genuinely independent primal fixed-point DARE iteration (implementation uses dual SDA doubling), and its A-vs-Aᵀ discrimination is self-checking via AssertGEd(relOraclePair, OracleFloorWrong).
2. MPC SoftRowUnavoidableMinimalViolation's hand-derived 0.6 worst-case slack (pos_2 = x0[0]+2*x0[1]+u0 at u0=-2) matches the double-integrator algebra.
3. NLS FlatParameterNoBlowup's tolerance-0 assertion is valid — the flat column is exactly zero in the finite-difference Jacobian and orthogonal to the damping block, so h[3]=0 exactly (d is default-zeroed, others use the skip-init flag).
4. UKF UpdateFailureLeavesState is valid despite calling ukfUpdate without ukfPredict — ukfUpdate regenerates sigma points from current x/P and leaves x/P untouched on the CHOP-indefinite path.
5. All Dispose paths and argument-validation-before-allocation orderings check out (no leaks found).

The three reported items are all low-severity oracle/tolerance/comment quality issues, not wrong results.
