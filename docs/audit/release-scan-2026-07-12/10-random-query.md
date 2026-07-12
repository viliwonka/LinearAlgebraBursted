# Release scan 2026-07-12 — area: random-query

Scanned 17 template files (core). Findings: total 4 — confirmed 4, uncertain 0, unverified 0, refuted 0; high 0, medium 0, low 4.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomMatrixOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Query.Shared.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryEnums.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryCore.Metric.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryCore.Metric.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryCore.Predicate.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryCore.Predicate.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.Predicate.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.Predicate.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SolveInfo.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SolveStatus.cs

## Findings

### 1. [low/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.fProxy.cs:395 — Cosine RowScore/ColScore recompute the invariant query norm ||q||^2 on every row/column call, so a per-row search loop recomputes it M times.

**Evidence**

```
for (int c = 0; c < nCols; c++) { dot += A[r,c]*q[c]; normA += A[r,c]*A[r,c]; normQ += q[c]*q[c]; }
```

In RowScore Cosine branch: normQ depends only on q, but distancesToRow/nearestRow/kNearestRows call RowScore once per row, recomputing ||q||^2 A.M_Rows times (ColScore line 447 has the identical pattern for columns).

**Verifier**

RowScore (lines 388-399) and ColScore (lines 439-450) both accumulate normQ (= q[c]*q[c] / q[r]*q[r]) inside the inner loop, and every per-row/col driver in the file (distancesToRow 478, nearestRow 515, farthestRow 563, countWithinRadius 609, kNearestRows 658, kFarthestRows 740, rowsWithinRadius 819, and their column twins at 494, 537, 585, 628, 699, 781, 841) calls the score kernel once per candidate. So for M candidates against N-dim query q, ||q||^2 is recomputed M times when it depends only on q — exactly the reviewer's claim. Burst cannot hoist across calls because q and A share the same pointer type and could alias from the kernel's view; there is no guard, alternate code path, or contract that mitigates this. Category and low severity are accurate; the suggested fix (hoist ||q||^2 out or special-case Cosine in the drivers) is the right remediation.

**Suggested fix**

Hoist ||q||^2 out of the per-row/col driver loops (compute once and pass in, or special-case Cosine in the callers) so the query norm is summed a single time per query rather than once per candidate vector.

### 2. [low/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.fProxy.cs:171 — weightedPickInPlace XML doc claims O(N + k) but the implementation is O(N*k): each of the k draws re-scans all N cumulative weights.

**Evidence**

```
Zero-alloc; O(N + k) where k = dest.N.
```

Doc claims the above, but the body is `for (int i = 0; i < k; i++) dest[i] = weightedPickFromTotal(in weights, total, ref rng);` and weightedPickFromTotal loops `for (int i = 0; i < n; i++)` over all weights per call — total cost is O(N) validation + O(N*k) picks.

**Verifier**

weightedPickInPlace (RandomOP.fProxy.cs:174-180) validates once (O(N)) then calls weightedPickFromTotal k times. weightedPickFromTotal (line 117-129) scans all n = weights.N cumulative weights per call, so k draws cost O(N*k), not the documented O(N + k). The doc at line 171 is inaccurate. Zero-alloc contract precludes the CDF-plus-binary-search structure that would justify O(N + k) (or O(N + k log N)). Low-severity documentation defect.

**Suggested fix**

Correct the complexity note to O(N*k) (validation O(N) + k linear scans), or state that O(N+k) would require a precomputed CDF that the zero-alloc contract disallows.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.cs:22 — Comment contains reviewer-facing design rationale (why two loops exist) rather than a contract, violating the contracts-only comment policy.

**Evidence**

```
A separate loop from <see cref="shuffleInPlace"/> is intentional: Pivot.Swap tracks the permutation parity via its swap counter, which plain index swapping cannot do.
```

This justifies an implementation choice/anticipates a reviewer question; the contract (what the method does) is already stated above it.

**Verifier**

Lines 17-21 already provide the full contract (reset + Fisher-Yates via Pivot.Swap, with the parity/Sign side effect explicitly noted). Lines 22-23 add "A separate loop from shuffleInPlace is intentional: ..." — the word "intentional" plus the comparison to a sibling method is reviewer-facing rationale justifying a code-organisation choice, exactly the class of comment CLAUDE.md tells authors to move to DEVLOG.md. The fix (delete the two lines, migrate to Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md) is correct.

**Suggested fix**

Move the rationale to the folder DEVLOG.md; keep only the contract (that randomPermutationInPlace resets to identity then shuffles via Pivot.Swap, preserving parity).

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/RandomOP.fProxy.cs:511 — Comment records a rejected-alternative / codegen-limitation note (why math.sincos is not used) rather than a contract, per the contracts-only policy.

**Evidence**

```
<c>math.sincos</c> is not used here because its <c>out</c>-parameter overload is not available via the type-proxy template mechanism; <c>math.sin</c> and <c>math.cos</c> are called separately instead.
```

**Verifier**

Lines 511-513 explicitly explain why math.sincos was rejected — this is a codegen-limitation / rejected-alternative note, not a contract. CLAUDE.md's comment policy states code/XML docs are contracts-only and lists "rejected alternatives" and "development history" as things that belong exclusively in DEVLOG.md. The first sentence (returns one variate; cached-spare path does not advance rng) is a legitimate contract; the sincos explanation is a codegen-mechanism narration that should be relocated to the folder's DEVLOG.md with a "(was RandomOP.fProxy.cs:511)" tag.

**Suggested fix**

Relocate the sincos-availability note to DEVLOG.md; the XML doc should only state Next's contract (returns one Gaussian variate; cached-spare path does not advance rng).

## Scanner notes

Verified codegen targets from Assets/LinearAlgebra/Source/OP/QueryOP.{float,double,int,long,short}.cs: fProxy->float/double, iProxy->int/short/long (all signed), so iAbs's `v<0`/MinValue/MaxValue logic and the WorstScore sentinels are safe (no uint expansion). Numerically the samplers are sound: all log-based ICDFs use uc=1-u to keep the argument in (0,1]; Cauchy/Logistic clamp with per-precision Consts.fProxyEpsilon; Triangular guards the b==a point-mass; Box-Muller draws u1=1-NextFProxy() to avoid log(0). Memory: every Temp-allocating generator (sampleKWithoutReplacementInPlace, multivariateNormal*, orthogonalInPlace, spdInPlace, conditionedInPlace, withRankInPlace) validates before allocating and disposes on the normal return path; no leaks on documented throw paths. The insertion-sort kNearest/kFarthest/topK kernels were checked in all four sim/distance x nearest/farthest quadrants and are correct. The iProxy nextUniformInPlace int-range guard running before the min==max constant-fill is consistent with its documented contract (int-range throw is unconditional), so not reported. No high/medium functional, aliasing, or overflow defects found beyond the documented integer-overflow contracts.
