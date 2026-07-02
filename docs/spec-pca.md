# Spec: PCA (Principal Component Analysis) — `LinearAlgebra.ML`

Status: coder-ready (design locked after a fable design review — verdicts folded in). Mirrors the
k-means precedent (`ML/KMeans*.cs`, class `fProxyKMeans_OP`, namespace `LinearAlgebra.ML`). New files
under `Assets/LinearAlgebra/CodeGen/TemplateSource/ML/`. fProxy-only (float/double).

## Goal
PCA over a data matrix, with a **fast** route (covariance/correlation matrix → symmetric eigensolve)
and **accurate + fast SVD** routes (full / exact-top-k / randomized-top-k SVD of the centered data),
plus a projection helper. Every fit returns a **model** the caller keeps.

## Choosing a route (for the public docs) — a 2×2 fast/accurate × full/partial
- **`pcaCovariance`** — full, fast. Forms the p×p covariance/correlation matrix (Gram of the centered
  data) and eigendecomposes it. Best when p (features) is small/modest; a p×p eigensolve is tiny. Also
  the only route that handles **wide** data (p > n). Squares the condition number (κ²) → prefer `pcaSVD`
  for near-degenerate data.
- **`pcaSVD`** — full, accurate. SVD of the centered data directly (no Gram, no κ²). Numerically safe
  default for moderate sizes when you want all components. Requires n ≥ p.
- **`pcaSVDTruncated`** — partial (top-k), exact. GKL top-k without the full SVD — the right tool when
  you only need the leading few PCs. Exact.
- **`pcaRandomized`** — partial (top-k), fast-approximate. Pays off only when p is LARGE, k ≪ p, and an
  approximate answer is acceptable (images/embeddings/gene expression). Requires n ≥ p. At small p the
  covariance route dominates; for exact top-k prefer truncated.

## Data orientation (matches StatsOP)
`X` is a `fProxyMxN`, **rows = samples (n), columns = features (p)** → `X.M_Rows == n`, `X.N_Cols == p`.
All PCA math is in feature space (p-dimensional).

## The PCA model — `fProxyPCAModel` (`ML/PCA.Model.fProxy.cs`)
Every fit returns a fitted **model** the caller keeps: to project new data (`pcaTransform`) or to read
axes/variances for reduction work. It bundles `fProxy` buffers so it is a per-precision
(fProxy-templated) struct (generates `floatPCAModel`/`doublePCAModel`) in `LinearAlgebra.ML`. This is
NOT a new pattern: every `_WS` (`fProxyKMeans_WS`, `fProxySVDThin_WS`) is already a struct of arena
buffer handles + an Arena factory — the model is the same thing pointed at OUTPUTS instead of scratch.
The house line to hold: unprefixed **scalar-only** structs = diagnostics (the `SolveInfo`/`EigenSolveInfo`
family); an fProxy-prefixed **buffer-carrying** struct = a model, justified here only because PCA has a
`transform` stage that consumes the outputs as a unit. Do NOT let anyone widen the scalar info structs
to carry buffers off the back of this.
```
namespace LinearAlgebra.ML {
  public struct fProxyPCAModel {
      public fProxyMxN components;             // p×k, each COLUMN a unit-norm principal axis (sign-fixed)
      public fProxyN   explainedVariance;      // length k, variance per component, DESCENDING
      public fProxyN   explainedVarianceRatio; // length k, explainedVariance[i] / totalVariance
      public fProxyN   mean;                    // length p, per-feature mean (to center new data)
      public fProxyN   scale;                   // length p, per-feature divisor before projecting:
                                                //   all ones (Covariance) or sample std-dev (Correlation)
      public int       k;                       // number of components (== p for full routes; top-k otherwise)
      public bool      converged;               // underlying eigensolve/SVD convergence; outputs undefined if false
      public bool Solved => converged;
      public static implicit operator bool(fProxyPCAModel m) => m.converged;  // `if (model)` reads naturally
  }
  // Arena factory, same pattern as the _WS factories (second `namespace LinearAlgebra { ArenaExtensions }`
  // block in the same file):
  //   public static ML.fProxyPCAModel fProxyPCAModel(this ref Arena arena, int p, int k);
  // sizes components p×k, explainedVariance/explainedVarianceRatio length k, mean/scale length p.
}
```
`k == p` for `pcaCovariance`/`pcaSVD`; `k ==` the requested top-k otherwise. Model fields are lowercase
(API surface, like the info structs — not `_WS` PascalCase). `converged` is a plain bool, NOT a status
enum: the wrapped kernels expose only bool, and inventing a richer status would violate the
"only-already-computed-values" rule.

## `PCAScaling` enum (`ML/PCAEnums.cs`, `//singularFile//` line 1, namespace `LinearAlgebra.ML`)
Two values, mirrors `KMeansInit`. Default (forwarding overloads) = `Covariance` (= sklearn's center-only default).
- `Covariance` — center only (PCA on the covariance matrix). `scale` output = all ones.
- `Correlation` — center **and** divide by per-feature **sample** std-dev (PCA on the correlation
  matrix). `scale` output = the sample std-devs; a **zero-variance feature gets `scale = 1`** so it maps
  to an all-zero standardized column (never a divide-by-zero) and contributes a zero component.

## THE denominator trap (makes the two routes agree — this is the headline correctness point)
`StatsOP.covariance` uses **sample** ÷(n−1); `StatsOP.standardizeColumns`/`colVariance` use **population**
÷n. For `pcaCovariance` and `pcaSVD` to return the SAME `explainedVariance` (the cross-route oracle
test), the SVD route MUST standardize with the **sample** convention:
- **Covariance mode:** center the working copy `Xc = X − mean`. Then `explainedVariance[i] = S[i]²/(n−1)`
  = the covariance-matrix eigenvalues.
- **Correlation mode:** `Xs[:,j] = (X[:,j] − mean[j]) / sampleStd[j]`,
  `sampleStd[j] = sqrt(Σ_i (X[i,j]−mean[j])² / (n−1))`. Then `Xsᵀ Xs / (n−1)` is exactly the correlation
  matrix, so `S[i]²/(n−1)` = the correlation-matrix eigenvalues. Do NOT reuse `standardizeColumns` (÷n).
Requires `n ≥ 2` (variance undefined for n<2) → throw `ArgumentException`.

## THE correlation degenerate-feature trap (fable catch — do NOT reuse `StatsOP.correlation()`)
`StatsOP.correlation()` puts **1** on the diagonal of a zero-variance feature by convention. If
`pcaCovariance(Correlation)` used that matrix it would emit a spurious **unit** eigenvalue on the
degenerate axis — but the SVD route's all-zero standardized column emits **0** there, so the two routes
would disagree on `explainedVariance` itself and break the cross-route oracle. Therefore
**`pcaCovariance` must build its own correlation matrix R inline** with the PCA-consistent convention:
`R = Cov ./ (sampleStd ⊗ sampleStd)`, and where `sampleStd[j] == 0` set the **entire row/column j to 0,
including the diagonal** (not 1). (Cheap, and PCA must build R itself anyway since there is no
`correlationInto` ref-dest form — do NOT add a public `correlationInto` with PCA-specific degenerate
semantics; inline it.)

## `totalVariance` (for `explainedVarianceRatio`) — computed directly, NOT as Σ explainedVariance
- Covariance mode: `totalVariance = Σ_j sampleVar(feature j) = trace(covariance)`.
- Correlation mode: each non-degenerate standardized feature has sample variance 1 ⇒
  `totalVariance = Σ_j (sampleVar(j) == 0 ? 0 : 1)` = number of non-degenerate features.
For the full routes `Σ explainedVariance == totalVariance` (roundoff); for top-k it is strictly less
(that's the ratio's point) — so compute `totalVariance` from the data in one pass, never as the sum of
the returned (possibly truncated) `explainedVariance`.

## Sign convention (determinism — this project cares)
Singular/eigen vectors have arbitrary sign. Fix it: for each component column, find the largest-|entry|
(first index wins ties); if negative, negate the whole column. Apply once, AFTER the solve, to
`model.components` — but **skip it entirely when `!converged`** (outputs are undefined and NaNs make the
abs-scan meaningless). Scores from `pcaTransform` inherit the fixed sign for free. Document three limits:
(1) it does NOT fix **degenerate/repeated-eigenvalue rotation** ambiguity — near-equal `explainedVariance`
values leave the eigenspace basis route-/precision-dependent; keep cross-route tests on well-separated
spectra and say so publicly. (2) A component with two near-equal-magnitude opposite-sign entries can
still flip between routes (roundoff picks a different pivot) — the "same up to sign" test tolerance
covers it. (3) Never rely on it across a degenerate spectrum.

## Methods (`fProxyPCA_OP`, namespace `LinearAlgebra.ML`)
**No `fProxyPCA_WS`** (see Scratch policy). Overload ladder per route (fable's shape): a zero-alloc `ref`
primary carrying ALL expert knobs; a scaling-only rung; a default rung; the allocating `ref Arena`
wrappers mirror the two shallow rungs (expert knobs stay on the ref primary — anyone tuning
`oversample`/`powerIters` is already in zero-alloc territory). Guards validate X and every `model.*`
shape against (n, p, k) **before** touching the temp pool (k-means guard-before-alloc rule); allocating
wrappers guard before allocating the model.

Shared model fill (all routes): `model.mean = colMean(X)`; `model.scale` = ones (Covariance) or
per-feature sampleStd (Correlation); `model.explainedVarianceRatio[i] = explainedVariance[i]/totalVariance`;
set `model.k`, `model.converged`; apply the sign convention (unless `!converged`).

```csharp
// ── 1. full, fast (covariance/correlation eigensolve; handles wide p>n) ──
bool pcaCovariance(in fProxyMxN X, ref fProxyPCAModel model, PCAScaling scaling);
bool pcaCovariance(in fProxyMxN X, ref fProxyPCAModel model);                       // => Covariance
fProxyPCAModel pcaCovariance(ref Arena arena, in fProxyMxN X, PCAScaling scaling);
fProxyPCAModel pcaCovariance(ref Arena arena, in fProxyMxN X);
//   Covariance: C = covarianceInto(X). Correlation: build R inline (degenerate-feature trap above).
//   eigenSymmetric(ref C, ref model.explainedVariance, ref model.components) — it DESTROYS C (fine, C is
//   temp scratch) and does an O(p²) symmetry check (passes: the Gram/ R build is exactly symmetric under
//   IEEE754). C must be PCA-built each call, never a caller matrix. k == p. Wide (p>n) is fine here.

// ── 2. full, accurate (svdThin on centered copy; REQUIRES n >= p) ──
bool pcaSVD(in fProxyMxN X, ref fProxyPCAModel model, PCAScaling scaling, int maxIter);
bool pcaSVD(in fProxyMxN X, ref fProxyPCAModel model, PCAScaling scaling);          // maxIter = svdThin default (75)
bool pcaSVD(in fProxyMxN X, ref fProxyPCAModel model);
fProxyPCAModel pcaSVD(ref Arena arena, in fProxyMxN X, PCAScaling scaling);
fProxyPCAModel pcaSVD(ref Arena arena, in fProxyMxN X);
//   Center/standardize a temp copy Xc per the denominator trap. svdThin(in Xc, ref U, ref S, ref V, …):
//   model.components = V (p×p), model.explainedVariance[i] = S[i]²/(n−1). NOTE svdThin demands an m×n
//   ref U that PCA discards → a SEPARATE n×p temp (do NOT alias Xc as U: svdThin reads A while writing U).
//   Throw ArgumentException if n < p ("PCA.pcaSVD requires samples>=features; use pcaCovariance for wide data").

// ── 3. exact top-k (GKL) ──
bool pcaSVDTruncated(in fProxyMxN X, ref fProxyPCAModel model, int k,
                     PCAScaling scaling, int oversample, uint seed, int maxIter);
bool pcaSVDTruncated(in fProxyMxN X, ref fProxyPCAModel model, int k, PCAScaling scaling); // svdTruncated defaults, verbatim
bool pcaSVDTruncated(in fProxyMxN X, ref fProxyPCAModel model, int k);
fProxyPCAModel pcaSVDTruncated(ref Arena arena, in fProxyMxN X, int k, PCAScaling scaling);
fProxyPCAModel pcaSVDTruncated(ref Arena arena, in fProxyMxN X, int k);
//   svdTruncated(in Xc, ref Uk, ref Sk, ref Vk, k, …, out converged). model.components = Vk (p×k),
//   explainedVariance = Sk²/(n−1) (length k), ratio vs the FULL totalVariance (sums to <1). 0 < k ≤ min(n,p).
//   svdTruncated ALSO requires n ≥ p (verified: its core throws for m<n) → pcaSVDTruncated throws the same
//   n<p guard + "use pcaCovariance for wide data" message, exactly like pcaSVD/pcaRandomized. NOT shape-free.

// ── 4. randomized top-k (HMT; inherits svdRandomized's n >= p) ──
bool pcaRandomized(in fProxyMxN X, ref fProxyPCAModel model, int k,
                   PCAScaling scaling, int oversample, int powerIters, uint seed, int maxIter);
bool pcaRandomized(in fProxyMxN X, ref fProxyPCAModel model, int k, PCAScaling scaling);   // svdRandomized defaults, verbatim
bool pcaRandomized(in fProxyMxN X, ref fProxyPCAModel model, int k);
fProxyPCAModel pcaRandomized(ref Arena arena, in fProxyMxN X, int k, PCAScaling scaling);
fProxyPCAModel pcaRandomized(ref Arena arena, in fProxyMxN X, int k);
//   svdRandomized(in Xc, …). 1 ≤ k ≤ min(n,p). Throw the same n<p guard + "use pcaCovariance for wide data".

// ── projection ──
void      pcaTransform(in fProxyMxN X, in fProxyPCAModel model, ref fProxyMxN scores); // zero-alloc, n_new×k
fProxyMxN pcaTransform(ref Arena arena, in fProxyMxN X, in fProxyPCAModel model);      // allocating
//   scores[i,:] = ((X[i,:] − model.mean) / model.scale) · model.components. Guards:
//   model.mean.N == model.scale.N == model.components.M_Rows == X.N_Cols; model.k == model.components.N_Cols
//   (defends a hand-assembled/stale model); scores is X.M_Rows × model.k.
```

## Determinism
`pcaRandomized` (and `pcaSVDTruncated`'s sketch) default `seed = 0x9E3779B1u` — the exact constant
`svdRandomized`/`svdTruncated` already default to (and `svdRandomized` maps `seed==0` to it internally):
just forward via the default overloads, so `pcaRandomized(arena, X, k)` is bitwise-reproducible and a
caller opts INTO variation with an explicit seed. Combined with the sign convention, default PCA output
is deterministic across runs (modulo the documented degenerate-spectrum rotation ambiguity).

## Scratch policy (why NO `fProxyPCA_WS`)
Temp-pool scratch + `ClearTemp` IS this library's realtime story (verified: the SVD kernels' non-WS
overloads build their `_WS` from `tempfProxyMat`/`tempfProxyVec`; `svdThin`/`eigenSymmetric` use
self-disposing `Allocator.Temp` locals; `covarianceInto`'s doc names the rolling window as its temp-pool
consumer). PCA fits ONCE (k-means earned its `_WS` via a per-frame restart loop; PCA has no such loop),
so a caller workspace is ceremony, and a union WS across four heterogeneous routes would charge every
covariance-route caller for n×p + kernel workspaces they never touch. Instead: each method allocates the
buffers it owns (p×p C / inline R, n×p centered copy Xc, the n×p U dump for svdThin) from **X's arena
temp pool**, and calls the kernels' existing **non-WS** overloads (which temp-alloc their own scratch).
The reuse that matters — the OUTPUTS — is covered by the `ref fProxyPCAModel` form. **Realtime pattern
(document it):** allocate the model once via `arena.fProxyPCAModel(p, k)`, call the `ref` fit each frame,
`ClearTemp()` at end of frame reclaims all internal scratch. Reserve the name `fProxyPCA_WS` for a future
true-zero-alloc pass only if profiling ever demands killing the per-frame temp bump.

## Guards (managed throws, before any alloc)
`n < 2` → throw (variance undefined). `X.N_Cols < 1` → throw. `pcaSVD`/`pcaSVDTruncated`/`pcaRandomized`
with `n < p` → throw with the "use pcaCovariance for wide data" message (all three inherit the SVD
kernels' m≥n constraint; only `pcaCovariance` handles wide data). Top-k `k` out of `(0, min(n,p)]` → throw.
`ref`-form model/scores shape mismatches → throw. `pcaTransform` asserts `model.k == model.components.N_Cols`.

## Tests (the cross-route oracle is the important one)
- **Route agreement:** same random data (n=50, p=6, well-conditioned, well-SEPARATED spectrum),
  `pcaCovariance` and `pcaSVD` give the SAME `explainedVariance` (tight tol) and SAME `components` up to
  the fixed sign — for BOTH `Covariance` and `Correlation`. Proves the denominator + degenerate handling.
- **Known spectrum:** data with known covariance diag(9,4,1) recovers those variances (descending),
  axis-aligned components.
- **explainedVarianceRatio:** sums to 1 (±tol) for full routes; `explainedVariance[0]` is the max.
- **Correlation degenerate feature:** a constant (zero-variance) column yields `scale=1`, an all-zero
  component/zero eigenvalue on that axis, and `totalVariance == #non-degenerate` — and BOTH routes agree
  (this is the fable-caught trap; without the inline-R fix pcaCovariance would emit a spurious unit λ).
- **Top-k routes:** `pcaSVDTruncated` top-k == first k of `pcaSVD` (tight tol); `pcaRandomized` matches
  within a loose tol; both ratios sum to <1.
- **pcaTransform:** projecting training data → scores whose columns have variance == `explainedVariance`
  (covariance mode); scores == U·S from the SVD route (up to sign); a manual `((X−mean)/scale)·components`
  matches; a stale-`k` model throws.
- **Sign determinism:** negating an input column / re-running yields identical `components`; assert on a
  well-separated spectrum only.
- **Determinism:** `pcaRandomized(arena, X, k)` twice → bitwise-identical (fixed default seed).
- **Guards:** n<2 throws; `pcaSVD`/`pcaSVDTruncated`/`pcaRandomized` all THROW for p>n (same n≥p
  constraint — svdTruncated requires m≥n too); mis-sized model/scores throws (managed, before alloc);
  `pcaTransform` stale-k throws.
- **Wide data:** `pcaCovariance` works for p>n; trailing (p−rank) eigenvalues ≈ 0.

## Burst / codegen notes
- `PCAEnums.cs` needs `//singularFile//` as literal line 1 (enum has no fProxy → else CS0102). The model
  file `PCA.Model.fProxy.cs` IS fProxy-templated (it holds fProxy buffers) — normal template, NOT singular.
- In `[BurstCompile]` test jobs assert with `Assert.IsTrue(a == b)`, NEVER `Assert.AreEqual(enum, …)`
  (BC1330 aborts the assembly's batch Burst compile → silent managed fallback → spurious failures).
- `const fProxy` is illegal in templates (proxy struct); use plain `fProxy` locals.
- Reuse existing ops, do NOT re-implement: `StatsOP.covarianceInto`/`colMean`, `Eigen.eigenSymmetric`,
  `SVD.svdThin`/`svdTruncated`/`svdRandomized`, `Linear_OP.dot`. (Correlation matrix R is built inline —
  see the degenerate-feature trap — NOT via `StatsOP.correlation()`.)

## Deliberately deferred (note, do not build)
`pcaFitTransform` — the SVD routes get training scores free as U·S, but adding a third output path to
four routes for one saved GEMM isn't worth the pre-release surface; `pcaTransform` after fit covers it.
`fProxyPCA_WS` true-zero-alloc forms — only if profiling later demands it.
