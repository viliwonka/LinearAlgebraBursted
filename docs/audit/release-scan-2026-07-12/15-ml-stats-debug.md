# Release scan 2026-07-12 — area: ml-stats-debug

Scanned 27 template files (core). Findings: total 4 — confirmed 4, uncertain 0, unverified 0, refuted 0; severity: high 0, medium 0, low 4.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.Workspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeansEnums.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/PCA.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/PCA.Model.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/PCA.Shared.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ML/PCAEnums.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/HistogramCore.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/HistogramOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/Stats.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsCore.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsCore.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/Structs.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Analysis/Analysis.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Analysis/Analysis.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Analysis/Analysis.Metrics.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Analysis/BoolAnalysis.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.Histogram.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.PCAModel.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Export.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Export.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Export.iProxy.cs

## Findings

### 1. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.fProxy.cs:36 — Print.Log(fProxyN) puts a spurious trailing comma before ']' when printing a sub-range (end < a.N).

**Evidence**

```
for (int i = start; i < end; i++) { ... if (i == a.N - 1) elementString = $"{element:G3}"; else elementString = $"{element:G3}, "; }  // the 'last element' test compares against a.N-1, not end-1
```

The "last element" test compares against `a.N - 1` rather than `end - 1`, so when printing a partial range the final element gets a trailing `", "` before the closing bracket.

**Verifier**

In Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.fProxy.cs lines 28-42, the loop runs `for (int i = start; i < end; i++)` but the last-element test is `if (i == a.N - 1)`. When a caller supplies `end < a.N` (a partial range), the loop exits before i reaches a.N-1, so the last iteration falls into the else branch and appends `", "`. The endStr is then appended immediately after (line 43), producing a spurious trailing comma before the closing bracket. Fix is exactly as suggested: compare against `end - 1`.

**Suggested fix**

Test the loop end: use `i == end - 1` (not `i == a.N - 1`) so the final element in a partial range prints without a trailing comma.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/HistogramCore.fProxy.cs:138 — densityInto/cdfInto XML docs advertise a half-open range '[lo, hi)' but the implementation (via histogramInto) is inclusive on hi (closed upper edge).

**Evidence**

```
densityInto doc: "Computes the probability density estimate over [lo, hi)"; cdfInto doc (line 179): "over [lo, hi)" — yet histogramInto keeps `lo <= x <= hi` and maps x==hi to the last bin (line 55-57).
```

The advertised half-open range contradicts the actual closed-upper binning contract of the underlying `histogramInto`.

**Verifier**

densityInto (line 138) and cdfInto (line 179) XML docs advertise range "[lo, hi)" (half-open), but both delegate unconditionally to histogramInto (lines 164, 201), which is contractually and behaviorally closed-upper: line 26 documents "[lo, hi]", line 30-33 states "lo <= x <= hi" with "closed upper edge x == hi maps to the last bin K−1", and the implementation at lines 55-57 confirms this. The very same docstrings then say "Same bin rule as histogramInto" (lines 145, 185), internally contradicting their own summary line. Fix by changing ")" to "]" in the two summaries.

**Suggested fix**

Change the two docs to '[lo, hi]' (or 'closed upper edge') to match the actual inclusive binning contract stated on histogramInto.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsCore.iProxy.cs:28 — Code comment references a test-class/method by name, which the project's contracts-only comment policy forbids (belongs in DEVLOG).

**Evidence**

```
"...see StatsTests.iProxy.cs's SumAccumulatorOwnOverflow, which pins this contrast: the same 2-element/MaxValue-filled input is correct-and-widened for int/short but silently wraps for long)."
```

A test-class + test-method name is embedded in a code comment; the contracts-only comment policy routes such references to DEVLOG.md.

**Verifier**

Lines 27-29 of Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsCore.iProxy.cs contain the exact quoted "(see StatsTests.iProxy.cs's SumAccumulatorOwnOverflow, which pins this contrast: ...)" — a test-class + test-method name embedded in a code comment. CLAUDE.md's comment policy is explicit: code comments state contracts only; notes to reviewers, dev/spec references, and (per the public-docs bar) test-class names belong in DEVLOG.md. The genuine contract (long-variant wraps; int/short safe by construction) is already fully stated on lines 17-26, so the parenthetical adds no contract information — it is a pointer to a test that demonstrates the behavior, which is exactly what the policy routes to DEVLOG. No DEVLOG.md exists in the Statistics folder yet, confirming the reference was left in-code rather than moved. Low severity as claimed (policy violation, not a numerical or behavioral defect); suggested fix (drop the test reference from the comment, add the note to a new Statistics/DEVLOG.md) is appropriate.

**Suggested fix**

Drop the test-class reference from the code comment; keep only the contract (long accumulator can wrap for the long variant; int/short are safe). Move the test-name note to Statistics/DEVLOG.md.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.fProxy.cs:350 — Comment records a rejected/prior algorithm alternative (development history), which the contracts-only policy routes to DEVLOG rather than code.

**Evidence**

```
"...update D2Weights[n] = min(...) instead of recomputing from scratch (which was O(k²·N·D))."
```

The past-tense parenthetical describes a prior/rejected implementation — development history that belongs in the folder's DEVLOG.md.

**Verifier**

Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.fProxy.cs lines 348-350 contain the exact quoted text, including the past-tense parenthetical "(which was O(k²·N·D))" describing a prior/rejected implementation. Per CLAUDE.md's contracts-only comment policy, development history and rejected alternatives belong in the folder's DEVLOG.md, not in code comments. The current-complexity phrase "Incremental D2Weights (O(k·N·D))" is contract; only the "instead of recomputing from scratch (which was ...)" clause is history. Severity "low" is appropriate — cosmetic policy violation, no runtime impact.

**Suggested fix**

State only the current contract/complexity (O(k·N·D) incremental update). Move the 'was O(k²·N·D)' history to ML/DEVLOG.md.

## Scanner notes

Scanned all 27 listed template files in full. Verified the one non-local risk: isOrthogonal's matMatDotTransA(A,A,B, A.N_Cols, A.M_Rows, B.N_Cols) correctly computes B = AᵀA (kernel semantics m=out-rows, n=inner, k=out-cols; confirmed against UnsafeOP.fProxy.cs), so no defect there. Numerical cores are sound: KMeans GEMM assignment (‖c‖²-2x·c argmin, ‖x‖² added back for inertia) is correct; PCA denominator conventions are consistent across the four fit routes and the covariance/SVD totalVariance definitions agree; histogram density/cdf normalization formulas check out; int/uint expansions avoid unsigned-subtraction traps (StatsCore.iProxy is signed-only by design; Analysis.iProxy uint path uses only equality tests). Memory: all Allocator.Temp scratch is disposed on all reachable paths, arena temps are documented ClearTemp-reclaimed, and PCA guard-before-alloc prevents orphaning. No high/medium functional defects found. Additional borderline comment-policy notes not filed as findings: StatsCore.fProxy.cs:21 ('2x width-4 accumulators, frozen fold' impl/perf detail) and covarianceInto's dispatch-detail comment could also move to DEVLOG under the strict policy.
