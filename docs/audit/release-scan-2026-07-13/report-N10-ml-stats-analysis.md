# Release scan 2026-07-13 — N10 narrow pass: ML (7) + Statistics (7) + Analysis (4)

Scanner: N10. Every line of all 18 template .cs files read; sibling pairs (fProxy vs iProxy,
row vs col, exact vs epsilon, facade vs core) diffed; XML contracts verified against bodies;
addendum patterns 1-7 swept. DEVLOG.md present in ML and Statistics (both consistent with
code); Analysis has none (nothing in it currently needs one, but see L-3/L-4 relocations).

Files: KMeans.fProxy.cs, KMeans.Workspace.fProxy.cs, KMeansEnums.cs, PCA.fProxy.cs,
PCA.Model.fProxy.cs, PCA.Shared.cs, PCAEnums.cs | StatsOP.fProxy.cs, StatsCore.fProxy.cs,
Stats.iProxy.cs, StatsCore.iProxy.cs, Structs.fProxy.cs, HistogramOP.fProxy.cs,
HistogramCore.fProxy.cs | Analysis.fProxy.cs, Analysis.iProxy.cs, Analysis.Metrics.fProxy.cs,
BoolAnalysis.cs

---

## HIGH

### H-1 — Analysis.isDiagonal(in boolMxN, bool compare = true) tests identity, not diagonality; no squareness check; undocumented inverting parameter
- File: Assets/LinearAlgebra/CodeGen/TemplateSource/Analysis/BoolAnalysis.cs:9-22
- Defect (three-part, same method):
  1. The predicate returns false unless bm[i,j] == (i==j) for EVERY cell — i.e. it requires
     the diagonal to be ALL TRUE. A genuinely diagonal bool matrix with any false diagonal
     entry is rejected: [[true,false],[false,false]] -> isDiagonal returns FALSE, while every
     same-named numeric sibling (Analysis.isDiagonal over fProxyMxN / iProxyMxN) allows
     arbitrary diagonal values. Same name, different semantics -> wrong result / API lie.
     It is actually isIdentity for bools.
  2. No M_Rows != N_Cols early-false: a non-square identity-pattern matrix
     (2x3 [[T,F,F],[F,T,F]]) returns TRUE, where all numeric isDiagonal/isIdentity overloads
     return false for non-square input.
  3. bool compare = true is undocumented, is immediately overwritten (compare = !compare;),
     and when passed false inverts the predicate into "no cell matches the identity pattern"
     — a parameter that lies about its role; reads like a debug leftover.
  - Quoted: public static bool isDiagonal(in boolMxN bm, bool compare = true) /
    compare = !compare; / if ((bm[i, j] == (i == j)) == compare) return false;
- Fix direction: reimplement as off-diagonal-all-false + squareness check to match the
  numeric siblings; if identity semantics are wanted, add isIdentity(boolMxN) instead; drop
  (or document and un-invert) compare.

---

## MEDIUM

### M-1 — 12 exception messages name the retired class StatsOP (facade is Stats)
- File: TemplateSource/Statistics/StatsCore.fProxy.cs:266, 287, 335, 358, 381, 403, 427, 462, 524, 545, 566, 586
- Users of Stats.rowSum(...) etc. see "StatsOP.rowSum: dest.N must equal A.M_Rows" — a class
  that no longer exists (public surface is Stats, StatsOP.fProxy.cs:10). Addendum pattern 2
  (rename straggler in exception messages).
- Fix direction: StatsOP. -> Stats. in all 12 message literals.

### M-2 — PCA XML doc + comments reference retired StatsOP
- File: TemplateSource/ML/PCA.fProxy.cs:14 (class summary: "not StatsOP's population (n)
  one"), :208 ("reusing StatsOP.correlation()"), :254 ("StatsOP.correlation's convention").
- The class-level XML summary is user-visible IntelliSense; the API is Stats.correlation.
- Fix direction: StatsOP -> Stats in the doc and both comments.

### M-3 — HistogramCore header comment uses retired HistogramOP / StatsOP
- File: TemplateSource/Statistics/HistogramCore.fProxy.cs:7 ("// HistogramOP: count-based
  distribution estimation..."), :18 ("identical to StatsOP reductions").
- Public facade is Histogram (HistogramOP.fProxy.cs:12) and Stats.
- Fix direction: update both names in the comment.

### M-4 — PCA public parameter maxIter vs settled canon maxIterations
- File: TemplateSource/ML/PCA.fProxy.cs:316, 344, 347-359 (docs), 401, 431, 437 (doc), 493,
  523, 526-527 (doc) — fitSvd/fitSvdTruncated/fitRandomized take int maxIter.
- The wrapped kernels all use maxIterations (SVD.fProxy.cs:167, SVD.LowRank.fProxy.cs:61,
  SVD.Randomized.fProxy.cs:51), KMeans.fit next door uses maxIterations, and maxIterations
  was one of the breaking canon renames. Addendum pattern 2.
- Fix direction: rename the parameter (and the "maxIter = SVD.thin's default" doc phrases)
  to maxIterations pre-release.

### M-5 — Analysis.rank parameter relTol vs canon relativeTolerance
- File: TemplateSource/Analysis/Analysis.Metrics.fProxy.cs:53 —
  public static int rank(in fProxyMxN A, fProxy relTol) (+ docs at :49-50, :78).
- QRCP/LQRP/SVD.Solvers/SVD.Subspace all use relativeTolerance. Addendum pattern 2 verbatim
  ("tol/relTol vs tolerance/relativeTolerance").
- Fix direction: rename to relativeTolerance.

### M-6 — Analysis.MaxZeroError — Pascal-case, opaque name on a camelCase public surface
- File: TemplateSource/Analysis/Analysis.fProxy.cs:68, 77
- Every sibling on Analysis is camelCase (trace, cond, rank, isSymmetric, isAnyNan); this is
  MaxZeroError and computes max|x| (an L-inf magnitude) — the name only makes sense from
  inside the test suite, where it is heavily used (10+ test templates).
- Fix direction: camelCase + descriptive rename (it is "max absolute entry"), or relocate to
  test-support if it isn't meant as public API.

### M-7 — Pascal-case predicates in BoolAnalysis (IsAllSame/IsAllEqualTo/IsAnyEqualTo, Arena.WhichTrue)
- File: TemplateSource/Analysis/BoolAnalysis.cs:24, 34, 47 and ArenaExtensions :142, :156
  (WhichTrue).
- Canon (naming-style-guide.md): predicates are lowercase camelCase; the SAME file defines
  Analysis.whichTrue (:85) next to Arena.WhichTrue — one concept, two casings.
- NOTE: Pascal-case predicates are a known OPEN ruling (coherence-audit s3-4); recorded here
  with both locations so the ruling can close it either way.

### M-8 — rowMean/colMean silently return NaN for an empty-axis matrix where all statistical siblings throw
- File: TemplateSource/Statistics/StatsCore.fProxy.cs:304-328 (via rowSum/colSum +
  fProxyComp.divInPlace(dest, 0)).
- Stats.colMean on a 0x3 matrix: colSum writes zeros, then divide by M_Rows == 0 ->
  dest = NaN,NaN,NaN with no error; Stats.colMin on the same input throws
  "Cannot compute statistics of an empty matrix." (:379). Same for rowMean with N_Cols == 0.
  Sibling-validation gap (addendum pattern 5) producing silent NaN.
- Fix direction: add the same empty-matrix guard to rowMean/colMean (or to rowSum/colSum).

### M-9 — Stats.covarianceInto validates nothing about C's shape
- File: TemplateSource/Statistics/StatsCore.fProxy.cs:612 (public via StatsOP.fProxy.cs:91).
- Doc says "fills caller-provided NxN matrix C (already allocated)" but no
  C.M_Rows != N || C.N_Cols != N check exists; every sibling ref-dest primitive in the same
  file validates dest.N. A mis-shaped C indexes out of bounds — in a release/Burst build
  (collections checks compiled out) that is silent memory corruption, not an exception.
- Fix direction: add the shape guard before the M<2 zero-fill (which also writes NxN).

### M-10 — Same pointer passed to two [NoAlias] kernel parameters (Gram-matrix calls)
- Files: TemplateSource/Analysis/Analysis.fProxy.cs:242 —
  UnsafeOP.matMatDotTransA(A.Data.Ptr, A.Data.Ptr, B.Data.Ptr, ...) (isOrthogonal), and
  TemplateSource/Statistics/StatsCore.fProxy.cs:648 —
  Blas.dot(in centered, in centered, ref C, transposeA: true) (covarianceInto), which
  forwards both pointers into matMatDotTransA([NoAlias] matA, [NoAlias] matB, ...)
  (OP/UnsafeOP.fProxy.cs:365).
- Addendum pattern 4 verbatim. Benign today (matA/matB are only read; only matC is written),
  but it formally breaks the no-alias promise — any future kernel revision that writes
  through or caches across matA/matB can miscompile these two call sites.
- Fix direction: either document A==B as supported and drop [NoAlias] from the two read
  pointers of matMatDotTransA, or route Gram products through a dedicated AtA kernel.

### M-11 — Analysis.cond / Analysis.rank silently ignore SVD convergence failure
- File: TemplateSource/Analysis/Analysis.Metrics.fProxy.cs:39, 62 (root cause in OP
  partition: SVD.Metrics.fProxy.cs:31,36 discards the SVDInfo returned by values).
- House pattern is diagnostics via status structs; here a non-converged bidiagonal QR yields
  garbage kappa/rank with no signal at all (not even NaN in many cases).
- Fix direction: have singularValues surface the SVDInfo (out param) and let cond/rank
  propagate it (or document that the sweepBudget backstop makes failure practically
  unreachable).

### M-12 — fProxyFullStats.count typed fProxy; float variant misrepresents large counts
- File: TemplateSource/Statistics/Structs.fProxy.cs:24 (public fProxy count;), filled with
  x.Data.Length at StatsCore.fProxy.cs:215/250.
- In the float variant, counts above 2^24 round (a 20,000,001-element vector reports count
  20,000,000). An element count is an int; nothing in the struct needs it as fProxy. Also
  count is the one field omitted from ToString() (:53).
- Fix direction: change field to int count (and add it to ToString).

### M-13 — Stats in-place transforms mutate without any in-place marker; flat forms mutate through in parameters — OPEN QUESTION
- Files: TemplateSource/Statistics/StatsCore.fProxy.cs:741-1022
  (standardize/rescale/center/maxAbs/softmax + *Rows/*Columns variants), exposed at
  StatsOP.fProxy.cs:43-54, 96-107.
- Sibling elementwise mutators carry the marker (fProxyComp.divInPlace, fillInPlace,
  Rand.weightedPickInPlace); these don't. The flat forms additionally take in T x and mutate
  through x.Data.Ptr, so the signature actively signals non-mutation; the per-axis forms take
  ref, giving the same operation two different mutation signals. XML docs do say "in-place"
  clearly, so this may be a deliberate transform-verb convention — but it is not recorded in
  the naming guide or a DEVLOG. Needs a maintainer ruling (rename vs record the convention);
  addendum pattern 3.

---

## LOW

### L-1 — Float-suffixed literals in fProxy templates (benign in double, style drift)
- StatsCore.fProxy.cs:34, 37, 60, 160 (/ 2f), 179, 183, 223, 233, 270, 290, 431, 436, 472,
  528, 549, 568, 588, 652 (1f /), 705 (R[i,i]=1f), 710-711 (0f, clamp -1f/1f);
  Analysis.fProxy.cs:70, 79, 96, 114, 159, 187, 215.
- Addendum pattern 6 hits: 0f/1f/2f generate verbatim into the double variant. All are
  exactly representable so values are correct after implicit widening — consistency-only
  (same files elsewhere use (fProxy)0). Fix: normalize to (fProxy)... casts.

### L-2 — Exception-message and exception-type drift
- StatsCore.fProxy.cs:15, 30, 54, ... — empty-input messages ("Cannot compute sum of an
  empty array.") lack the MethodName: prefix the style guide's exception canon requires; and
  KMeans.fProxy.cs:51 throws InvalidOperationException for empty X while PCA.fProxy.cs:24-27
  throws ArgumentException for the equivalent shape guard.
- Fix: prefix messages; pick one exception type for input-shape violations.

### L-3 — Implementation narration comment (contract-only policy)
- StatsCore.fProxy.cs:20-21 — "The SIMD reduction lives in UnsafeOP.sum (2x width-4
  accumulators, frozen fold)... See UnsafeOP reductions / matVecDot." — narrates another
  file's implementation. Proposed DEVLOG entry (Statistics/DEVLOG.md):
  "## StatsCore.fProxy" / "- 2026-07-13 | sum forwards to UnsafeOP.sum's SIMD reduction
  (2x width-4 accumulators, frozen fold). (was StatsCore.fProxy.cs:20-21)"

### L-4 — Rejected-alternative postmortem in KMeans comment
- ML/KMeans.fProxy.cs:103-109 — first sentence is contract ("assignment seeded to -1 so all
  N points register as changed on iter 0"); the rest ("Initialising PrevAssignment instead
  read from an uninitialized assignment buffer ... causing k=1 to return the seeded point
  rather than the global mean") is a bug postmortem. Proposed DEVLOG entry (ML/DEVLOG.md):
  "## KMeans.fProxy.cs" / "- 2026-07-13 | -1-seed goes into assignment, NOT PrevAssignment:
  the swapped variant read uninit assignment (often all-zeros) -> zero changes on iter 0 ->
  k=1 returned the seeded point instead of the global mean. (was KMeans.fProxy.cs:103-109)"

### L-5 — Dev-speak / historical phrasing in PCA comments
- ML/PCA.fProxy.cs:104 — "keeping the cross-route oracle honest" (test-strategy speak);
  :252 — "which is only ~1 now that sampleStd is a direct row-sum, not sqrt(C[j,j])"
  (historical "now that"). Both relocatable to ML/DEVLOG.md; the surrounding numeric contract
  sentences can stay.

### L-6 — HistogramCore header contradicts its own edge rule
- Statistics/HistogramCore.fProxy.cs:9-12 — header says "K equal-width bins over [lo, hi)"
  two lines above "Closed upper edge: x == hi maps to the last bin". The effective range is
  [lo, hi]. Fix: phrase the header as [lo, hi] with hi folded into bin K-1.

### L-7 — Analysis.cond returns 0 for an empty (k==0) matrix
- Analysis/Analysis.Metrics.fProxy.cs:35-36 — kappa2 = 0 is an impossible condition-number
  value used as an undocumented sentinel (reads as "perfectly conditioned"). Fix: document
  it, or return NaN/throw for degenerate shape.

### L-8 — Missing exact isZero on the float/double Analysis surface
- Analysis/Analysis.fProxy.cs:48-66 — every other structural predicate has both a bare
  (exact) and an epsilon form; isZero has only the epsilon form, while the integer sibling
  (Analysis.iProxy.cs:12) has the bare form. Asymmetric coverage that looks accidental
  (exact zero is meaningful for structural zeros). Fix: add isZero(in fProxyN/fProxyMxN).

### L-9 — _OP survives in template FILE names only
- Statistics/StatsOP.fProxy.cs contains class Stats; Statistics/HistogramOP.fProxy.cs
  contains class Histogram. Generated package ships StatsOP.float.cs etc. — the purged token
  remains visible in shipped filenames. Fix: rename templates (Histogram.fProxy.cs and e.g.
  StatsFacade.fProxy.cs; note Stats.iProxy.cs already exists, so plain Stats.fProxy.cs is
  also free and symmetric).

### L-10 — fProxy median can overflow where the iProxy sibling deliberately guards
- Statistics/StatsCore.fProxy.cs:160 — (copy[n-1] + copy[n]) / 2f overflows to +Inf for two
  near-MaxValue middle elements; StatsCore.iProxy.cs:159-165 documents and avoids exactly
  this by widening. Sibling drift; edge case. Fix: a + (b - a) / 2 form.

### L-11 — Attribute oddities on data structs / partials
- Statistics/Structs.fProxy.cs:21 — [BurstCompile] on plain data struct fProxyFullStats (no
  jobs/function pointers; meaningless) while sibling fProxyMeanMinMaxRangeStats has none;
  Analysis/BoolAnalysis.cs:6 — [BurstCompile] on only one partial declaration of Analysis.
  Harmless; remove for consistency.

### L-12 — PCA.transform doesn't reject an unconverged model
- ML/PCA.fProxy.cs:574-593 — guards feature count and stale k ("defends a hand-assembled or
  stale model") but not model.converged == false, where components are documented-undefined;
  projection then silently emits garbage scores. Fix: add a converged guard or a one-line
  doc note on transform.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 1     |
| MEDIUM   | 13    |
| LOW      | 12    |

Areas confirmed clean (verified, no findings):
- KMeans algorithm logic: GEMM-assignment score patch, convergence/final-sync split,
  empty-cluster reseed bookkeeping (-1 exclusion scratch), k-means++ incremental D2 update,
  reservoir uniform seeding, k-clamp propagation between overloads, all workspace shape
  guards, guard-before-alloc in the arena wrappers, NextFProxy shim resolution
  (NextFloat/NextDouble both exist) and Rand.weightedPick contract compatibility
  (non-negative finite weights, total>0 pre-checked).
- PCA numerics: sample (n-1) denominator consistency across all four routes, totalVariance
  computed before C is destroyed, correlation-mode degenerate-feature convention matching
  between fitCov and BuildWorkingCopy, sigma^2->variance conversion, ratio-vs-full-
  totalVariance for top-k, sign-convention skip on !converged, eigen/singular value
  DESCENDING order verified against the Eigen/SVD templates, forwarding-overload default
  formulas match the kernels' documented defaults verbatim (oversample/seed/sweepBudget
  algebra checked), temp-pool discipline (Xc/U separate; symmetricInPlace destroys only
  PCA-built scratch), doc crefs resolve against real SVD/Eigen overloads.
- Statistics numerics: two-pass variances, Bessel-corrected covariance via exactly-symmetric
  Gram formulation, temp means zero-init confirmed (fProxyTempVec defaults to ClearMemory),
  numpy-linear percentile bounds-safe for n>=1, NaN-safe !(x > 0) guards on all zero-range /
  zero-std transforms, softmax max-shift stability, correlation clamp to [-1,1] and
  zero-variance conventions, cdf last-bin pin to exact 1, density/CDF scratch validated
  before Temp allocation, iProxy widened-return contracts (long sum / double mean/median)
  correct for all of int/short/long, and the long-wrap limitation properly DEVLOG-pinned.
- Histogram binning: in-range test drops NaN, closed upper edge, rounding clamps make the
  b in [0, K-1] invariant airtight even when w underflows; 2D variant applies the identical
  rule per axis; all four facade shape-combination overloads forward correctly.
- Analysis.iProxy alsoExpand[uint]: every predicate is exact-equality with no subtraction/
  negation — safe for uint; exact/epsilon sibling pairs in Analysis.fProxy are logically
  consistent (strict-triangle bounds correct, squareness pre-checks present); isOrthogonal's
  NaN-reject before the epsilon identity test is correct, and its Temp Gram matrix is
  zero-initialized before the accumulating kernel (fProxyMxN ctor ClearMemory path).
- Determinant/logDeterminant: pivot-sign handling, empty-matrix conventions (det=1, log 0),
  singular -> (sign 0, -Inf) verified against Pivot/LU contracts; Consts.fProxyZeroThreshold
  substitutes to per-type values (1e-6 float / 1e-14 double) — no cross-precision epsilon
  leak anywhere in the partition.
