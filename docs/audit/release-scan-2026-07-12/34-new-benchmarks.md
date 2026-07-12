# Release scan 2026-07-12 — area: new-benchmarks (post-scan code)

{"total":3,"confirmed":2,"uncertain":0,"unverified":0,"refuted":1,"high":0,"medium":0,"low":2}

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KalmanBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/MPCBenchmark.fProxy.cs
- Assets/LinearAlgebra/Benchmarks/KalmanBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/MPCBenchmark.cs

## Findings

### 1. [low/numerical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KalmanBenchmark.fProxy.cs:371 — EKF drift-safety justification cites the 80-step acceptance test, but the benchmark drives the pendulum far past that range and in a different regime

**Evidence**

```
Comment (365-371): 'NonlinearSteps keeps the cumulative step count (across Bench.Time's
own repeated job.Run() calls on this persistent state) well inside the range
KalmanTests.fProxy.cs's own 80-step EKF acceptance test already exercises.'

But NonlinearSteps=100 per Execute (already > 80), and Bench.Time runs
Warmup(1)+Runs(4)=5 calls on the SAME persistent state => ~500 cumulative steps,
~6x the cited 80. Moreover the acceptance test (KalmanTests.fProxy.cs:323) tracks a
TRUE trajectory with real noisy measurements (z=sin(theta_true)+noise) and seeds
P[0,0]=P[1,1]=0.5, whereas EkfCycleFProxy uses a ZERO measurement (line 379) and
unseeded P=0 self-simulation -- so the 80-step test does not actually validate
boundedness for the benchmark's regime. Risk: forward-Euler amplitude drift over
500 steps can push status to InnovationSolveFailed, which would print as 'Failed'
and misrepresent a timing row as a solver failure. UkfCycleFProxy has the identical
persistent-accumulation exposure with no drift note at all.
```

**Verifier**: Traced concretely: NonlinearSteps=100 (KalmanBenchmark.cs:37); Bench.Time is Warmup=1 + Runs=4 = 5 job.Run() calls on the same job value (Bench.cs:27-46); the file's own top comment (KalmanBenchmark.fProxy.cs:16-22) states that fProxyKFState is NativeArray-backed and mutations survive across those calls; the referenced EKF test (KalmanTests.fProxy.cs:323) is `for (int k = 0; k < 80; k++)` with real-truth propagation + noisy z + seeded P=0.5*I. The benchmark's EkfCycleFProxy uses z=zeros (line 379) and P=0 (fProxyKFState ctor contract "start zeroed"). Thus the comment at line 365-371 is factually wrong on two counts: (a) 100 per Execute already exceeds 80, and cumulative ~500 across the 5 persistent Runs is ~6x over, not "well inside"; (b) the cited 80-step test's regime (real tracking + seeded covariance) is materially different from the benchmark's (zero-z self-simulation + P=0), so it does not validate boundedness for this regime. The UkfCycleFProxy path (lines 394-415) has the same persistent-accumulation exposure with no drift-safety note. The escalation to "may print Failed via InnovationSolveFailed" is speculative — for m=1 the innovation solve is scalar division by H\*P\*H^T + R with R=1e-3 > 0, and cos/sin keep Jacobians bounded, so I cannot construct a concrete failure. But the documentation defect (bad arithmetic + wrong-regime justification) is directly verifiable, matching the reported low severity.

**Suggested fix**: Either reset the KF state between Bench.Time calls (or reduce NonlinearSteps so cumulative*5 stays under the validated bound), or drive the filter with a real tracking measurement so boundedness holds; and correct the comment's arithmetic (100/call, ~500 cumulative, not "well inside 80").

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/Benchmarks/KalmanBenchmark.cs:99 — Literal '[fProxy]' codegen placeholder left in user-facing section headers that actually contain both float and double rows

**Evidence**

```
Section headers print the literal token '[fProxy]': KalmanBenchmark.cs:99/113/127/141
and MPCBenchmark.cs:105/121/137 (e.g. '--- 1. Linear predict+update, full covariance
path [fProxy] ---'). These are hand-written harness files (not templates), so
'[fProxy]' is emitted verbatim into benchmark-kalman.txt/benchmark-mpc.txt. Yet each
such section then appends BOTH float rows (KfCycleFloat...) AND double rows
(KfCycleDouble...), so the '[fProxy]' tag misrepresents the content. No other
hand-written benchmark harness (e.g. LQRBenchmark.cs) uses this literal in its headers.
```

**Verifier**: Verified: KalmanBenchmark.cs lines 99, 113, 127, 141 and MPCBenchmark.cs lines 105, 121, 137 emit the literal string "[fProxy]" (a codegen placeholder token) into user-facing section headers in benchmark-kalman.txt/benchmark-mpc.txt. These files are the hand-written harness half (per the file's own comment at KalmanBenchmark.cs:78-79 pointing at TemplateSourceBenchmarks/KalmanBenchmark.fProxy.cs for the templated half), so the string is not substituted by codegen. Each section actually appends BOTH \*Float and \*Double build methods immediately after, so "[fProxy]" mislabels the content (dtype is already a column). LQRBenchmark.cs has no such literal (Grep returned zero matches), and TemplateSourceBenchmarks/DEVLOG.md contains no entry defending this as intentional. Real, cosmetic, correctly rated low severity — the printed report reads e.g. "--- 1. Linear predict+update, full covariance path [fProxy] ---" which is misleading.

**Suggested fix**: Drop the '[fProxy]' token from these seven section-header strings (the dtype column already distinguishes float vs double), or replace with '[float & double]'.

## Refuted

| # | File:Line | Severity/Category | Summary | Why refuted |
|---|-----------|-------------------|---------|-------------|
| 1 | Assets/LinearAlgebra/Benchmarks/MPCBenchmark.cs:33 | low/performance | WarmPrewarmFrames=5 is described as a 'conservative margin' over the test's steadyFrame=8, but 5<8 is less margin, not more | The finding misreads the referenced test. In MPCTests.fProxy.cs:222-228, `steadyFrame = 8` is an assertion checkpoint (`if (f >= steadyFrame) AssertLE(info.activeSetChanges, 3)`), not the number of frames it takes for churn to collapse. The comment explicitly distinguishes these: churn "shows collapsing within a handful of frames" (the actual collapse — fast) vs "that test's own steadyFrame constant is 8" (the test's safe checkpoint). "5 is a conservative margin given this benchmark's smaller, better-conditioned random plants" is scoped to the benchmark's plants: they collapse even faster than the test's harder plant (which already collapses well before frame 8), so 5 comfortably exceeds their actual collapse point. The comment is not claiming 5 > 8 nor margin over the test's 8; the a-fortiori argument is "the test proved 8 is safe on a harder plant; on easier plants 5 is still ample". Dense wording, but not self-contradictory and not a defect. The reviewer also concedes the value is empirically fine. No traced concrete incorrectness. |

## Scanner notes

Scanned the 4 in-scope benchmark files in full plus supporting sources (Kalman.fProxy.cs steadyStateGain/predictFixed/updateFixed, MPC.fProxy.cs solve + warm-start path, MPC.State.fProxy.cs full struct/ctor, Bench.cs, LQRBenchmark.cs/LQRBenchmarkFmt, KalmanTests EKF fixture, TemplateSourceBenchmarks/DEVLOG.md). No high/medium defects found: no DCE (all timed jobs sink into persistent/Temp NativeArrays consumed downstream), no setup inside timed regions, no leaks (every Persistent alloc and every Temp alloc in steadyStateGain is disposed), cold-reset is complete and faithful to fresh construction, Section-3 reps are genuinely cold (SDACore never reads Kss), all 10 jobs carry CompileSynchronously=true, us/step and per-solve arithmetic and all formatter arg/column counts are consistent, and warm-frame state genuinely persists across Run() with adequate prewarm for these fast-contracting plants. Only three low-severity text/methodology nits remain.
