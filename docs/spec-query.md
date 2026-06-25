# Spec: QueryOP — search & selection inside vectors/matrices

Status: **CONVERGED, coder-ready** (2026-06-25). Supersedes the Query half of
`docs/spec-views-and-query.md`. **Views are DROPPED** (see that doc's banner). Games-oriented Burst
library — favour small, zero-alloc, Burst-compatible designs; do **not** chase numpy/matlab completeness.

## The reframe (why this is worth building)
- `StatsOP` is the **reduction** layer: collapse a vec/matrix to scalars or per-axis vectors
  (mean/min/max/argmin/variance/norms). Mostly done.
- `QueryOP` is a **different** layer: **search & selection over a matrix treated as a set of vectors**
  (rows = points/entities, cols = features). The stub comment says it outright: *"search closest or
  farthest vector inside matrix compared to a given vector."* This layer (nearest/farthest/top-k/
  within-radius, best-by-score) is **not built at all** and is the genuinely games-shaped, high-value part.

## Burst ground rules (unchanged — see old doc for detail)
- No managed types in jobs: no lambdas/delegates/LINQ/`List<T>`/`int[]`. Unmanaged structs + pointers only.
- `fProxyN`/`fProxyMxN` are contiguous row-major (`A[i,j]=Data[i*N_Cols+j]`). Rows are contiguous;
  **columns are strided** (stride = `N_Cols`).
- The Burst-native "lambda" = the struct-functor pattern (same as the optimizer `where F:struct,I...`).
- **Codegen gotcha:** `const fProxy` is illegal in templates (`fProxy` is a proxy struct). Use plain
  `fProxy` locals (Burst constant-folds).

---

## Cross-cutting policies

### P0. Naming — camelCase (matches existing StatsOP/NormsOP compounds)
**lowerCamelCase** function names, consistent with the library's existing `rowMin`/`colMax`/`rowNormL1`/
`colStdDev`: `argMaxAbs`, `rowArgMin`, `argMaxRowNorm`, `nearestRow`, `kNearestRows`, `findValue`,
`countNonzero`, `distancesToRow`, `rowsWithinRadius`. (The short primitives `argmin`/`argmax` already in
StatsOP stay as-is; new compound names use camelCase.)

### P1. Symmetry — every row op has a column twin
`nearestRow`↔`nearestColumn`, `rowArgMin`↔`colArgMin`, `argMaxRowNorm`↔`argMaxColNorm`,
`distancesToRow`↔`distancesToColumn`, `rowsWithinRadius`↔`columnsWithinRadius`, etc. Column kernels loop
with stride `N_Cols` (slower, non-contiguous) — documented, no view type needed. (This is the one place
the dropped "views" idea would have helped; we just write the strided loop directly.)

### P2. Types — generate the integer-exact subset for `iProxy` too
- **fProxy + iProxy (int/short/long):** `argMaxAbs`/`argMinAbs`, `rowArgMin`/`rowArgMax`/`colArgMin`/
  `colArgMax`, `findValue`, `nonzero`/`countNonzero`, `argMaxRowNorm` by **L1/Linf**, and search by
  **Manhattan / Chebyshev / SqEuclidean / Dot**.
- **fProxy only (need sqrt / division):** norm selection by L2, metrics **Euclidean / Cosine**,
  `normalizeColumns`, `argMaxScore`/`kBestScored`, `pairwiseDistances`.
- New `QueryOP.iProxy.cs` for the integer subset; `QueryOP.fProxy.cs` for the full surface.

### P3. Integer overflow — REAL BOUNDS (NOT "overflow-safe")
Integer **SqEuclidean** and **Dot** accumulate products (`diff*diff`, `a*b`) → can overflow for large
coords (esp. `short`). Decision:
- **No silent widening** (cross-type accumulation breaks the one-type-per-expansion codegen). Accumulate
  in the proxy type.
- **L1 (Manhattan) and Linf (Chebyshev) are the recommended integer metrics** — they sum/max `|diff|`,
  never square. They're also the canonical grid distances games want (taxicab / king-move).
  **However** they are NOT fully overflow-safe: the subtraction `A[r,c] - q[c]` itself can overflow
  at the type boundary (e.g. `30000 - (-30000) = 60000` wraps for `short`). The real precondition is:
  **each element AND each element-wise difference must fit the proxy type** (for `short`: coordinates
  roughly within ±16383 so differences fit ±32767). Values at the type extreme (MinValue) are mapped to
  MaxValue in abs — off-by-one in magnitude, but correct sign/ordering and nonzero classification.
- **SqEuclidean / Dot for integers carry a documented caveat:** caller must ensure
  `maxAbsValue² × dimension` fits the type; otherwise use the float/double variant. Document the bound;
  do not guard in the hot loop.
- **For larger coordinate ranges:** use the float/double variant. Document this at each API entry point.

### P4. Variable-length results — caller `intN` buffer + returned count, NEVER `int[]`
`int op(... ref intN indices)` fills `indices[0..count)` and returns `count`; caller sizes
`indices.N ≥ worst-case`. Optional allocating arena convenience (`-> intN`) does a count pass +
exact alloc. value+index singletons use `out` params (not ValueTuple — safest under Burst).

### P5. Top-k — bounded insertion, NOT a heap
Games' k is tiny (≤ ~16). Maintain the k best-so-far in the caller's length-k `intN` (+ parallel
`fProxyN` scores), sorted; each candidate does one compare vs the current k-th best, insert-and-shift on
win. Contiguous, branch-predictable, zero-alloc, Burst/SIMD-friendly; O(n·k) but beats a heap at small k.
(= numpy `argpartition` / matlab `maxk`.) `nearestRow` = the k=1 degenerate (use a dedicated tight loop,
no buffer). If large-k ever matters, quickselect is the future add — not now.

### P6. Codegen-legality (red-team)
- **No proxy-typed default params** (CS1750 — see `[[template-default-params-cs1750]]`). With Lp/Minkowski
  dropped, no metric needs a `p`, so this is moot for v1 — but if a `fProxy`-typed param ever gets a
  default, use a forwarding overload, never `fProxy x = ...`. `int` defaults are fine.
- **Shared enum, subset kernels:** `iProxy` kernels implement only the integer-exact enum members (P2);
  passing a float-only member to an int kernel must **assert/reject**, not silently misbehave.
- **Score units:** `nearestRow`'s returned `score` is in the metric's own units — **SqEuclidean returns
  squared distance** (caller sqrts if needed). Don't sqrt in the hot loop.
- **Cosine zero-vector guard:** define `cosine(0, ·) = 0` (avoid div-by-zero).

### Enums (singular shared file `OP/QueryEnums.cs`, like `WindowType.cs`)
```
enum Norm   { L1, L2, Linf }                                       // single-vector magnitude (Lp dropped — add to NormsOP if ever needed)
enum Metric { Manhattan, Euclidean, SqEuclidean, Chebyshev,        // distance: nearest = MIN
              Cosine, Dot }                                        // Cosine/Dot similarity: nearest = MAX
enum NormalizeMode { MinMax, ZScore }
```
`distance(a,b) = norm(a−b)` unifies the distance metrics. **The enum carries direction:** `nearestRow`
returns the row optimizing closeness (min distance OR max similarity) so cosine "just works";
`farthestRow` returns the opposite. Metric dispatch + its min/max direction hoist outside the per-row loop.
*(Enum vs struct-functor metric: enum = simple API + one hoisted branch; a metric functor would specialize
with zero overhead but is more verbose. For game-sized sets the enum wins — keep it.)*

---

## The surface

Groups **1–4 are core**; group 5 + `pairwiseDistances` + `Lp` + Tier-C predicate functors are
**on-demand**. All signatures shown for `fProxy`; `iProxy` gets the P2 subset. `<T>` = generic over
vec+matrix flat data (`where T:unmanaged,IUnsafefProxyArray`, as `StatsOP.argmin<T>` already does).

### Group 1 — Extremes (value + index)
```
void argMaxAbs<T>(in T x, out fProxy val, out int flatIndex)       // + argMinAbs<T>  (the pivot primitive — CORE)
void decodeIndex(int flat, int nCols, out int row, out int col)    // flat -> (r,c) helper

// per-axis (matrix; row+col twins). value+index, and index-only convenience.
int  rowArgMin(in fProxyMxN A, ref intN colIndexPerRow, ref fProxyN valPerRow)   // returns M_Rows
int  rowArgMin(in fProxyMxN A, ref intN colIndexPerRow)
//   + rowArgMax, colArgMin, colArgMax   (col* fill a length-N_Cols buffer of row indices)

// on-demand: minWithIndex / maxWithIndex<T> (value+index together) — convenience over existing argmin/argmax+[]
```

### Group 2 — Norm-selection   [DEDUP: reuse existing norm code — do NOT add a `norm` fn here]
`NormsOP` already has `L1<T>`/`L2<T>`/`LInf<T>` (generic over vec+matrix); `StatsOP` has per-row/col
`rowNormL1/L2`, `colNormL1/L2`. QueryOP adds only **selection**, reusing those:
```
int argMaxRowNorm(in fProxyMxN A, Norm n)                          // README example — CORE; reuses StatsOP.rowNormL2 etc.; + argMaxColNorm
//  on-demand: argMinRowNorm / argMinColNorm, value+index out variants
```
The `Norm` enum just indexes which existing norm to select on. (`Lp` dropped from v1; if ever wanted, add
`Lp<T>(in T x, fProxy p)` to **`NormsOP`** — its home — not here.)

### Group 3 — Search over a set of vectors (the centerpiece)
```
void distancesToRow   (in fProxyMxN A, in fProxyN q, Metric m, ref fProxyN dest)                // + ...Column
void nearestRow       (in fProxyMxN A, in fProxyN q, Metric m, out int index, out fProxy score) // + farthestRow, + Column twins
int  kNearestRows     (in fProxyMxN A, in fProxyN q, int k, Metric m, ref intN idx, ref fProxyN score) // returns min(k,M_Rows); + kFarthest, + Column
int  rowsWithinRadius (in fProxyMxN A, in fProxyN q, fProxy r, Metric m, ref intN idx)          // returns count; + Column
int  countWithinRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m)                        // + Column
// on-demand (pdist2): void pairwiseDistances(in fProxyMxN A, in fProxyMxN B, Metric m, ref fProxyMxN dest)
```
`nearestRow` is zero-alloc (tight loop, no `dest`); `distancesToRow` is for when you want every distance
(then `argmin` it yourself). Guard `q.N == A.N_Cols`. `score` is in the metric's units (SqEuclidean →
squared).

### Group 4 — Value / mask search
```
int findValue<T>   (in T x, fProxy target, fProxy tol)           // first flat index, else -1 (Excel MATCH)
int nonzero<T>     (in T x, fProxy tol, ref intN idx)            // returns count
int countNonzero<T>(in T x, fProxy tol)

// whichTrue / countTrue are BOOL-ONLY → live in singular BoolAnalysis, NOT per-type QueryOP (no ×5 dup):
int BoolAnalysis.whichTrue(in boolN mask, ref intN idx)          // bridges existing `C > A` masks; returns count
int BoolAnalysis.countTrue(in boolN mask)                        // + boolMxN; sits beside IsAny/IsAllEqualTo
```
**Tier C (on-demand, sketch only):** predicate functor `interface IfProxyPredicate { bool Test(fProxy x); }`
→ `findFirst/count/any/all/findAll<P>(... ref P pred)`; row-score `interface IfProxyRowScore { fProxy
Score(in fProxyN row); }` → `argMaxBy/argMinBy/topKBy<S>`. Build when a real use appears.

### Group 5 — Utility-AI convenience (fProxy only, optional)
```
void normalizeColumns(ref fProxyMxN A, NormalizeMode mode)       // reuse colMin/Max/Mean/StdDev; + normalizeRows
int  argMaxScore(in fProxyMxN A, in fProxyN weights)             // = argmax(A·w), FUSED zero-alloc (dot-per-row, track max; no temp)
int  kBestScored(in fProxyMxN A, in fProxyN weights, int k, ref intN idx, ref fProxyN score)
```

---

## Utility AI — what it actually needs (expanded)
Utility AI scores a set of options and picks the best/top-k. **Rows = options** (targets, cover points,
actions), **cols = considerations** (distance, health, threat, cost). Pipeline:
1. **Normalize considerations** so they're comparable → `normalizeColumns` (reuses col stats). *[Group 5]*
2. **Apply a response curve** (diminishing returns/thresholds) → the **easing functors already exist**
   (generators work). *[already in library]*
3. **Weighted sum** → score per row = `A·w` → `dot`. *[already in library]*
4. **Pick** → `argmax(score)` / top-k / softmax-sample → `argMaxScore`/`kBestScored`. *[Group 5 = thin]*

So utility AI is **not a subsystem** — the library owns 3 of 4 stages; Query adds only step 4 (+ optional
step 1). Document it as a recipe, not a black box.

## Call-site validation (wrote real game snippets to test ergonomics)
```
// nearest enemy:           queryOP.nearestRow(in enemies, in playerPos, Metric.SqEuclidean, out int idx, out fProxy d2);
// 3 best targets:          int n = queryOP.kNearestRows(in enemies, in playerPos, 3, Metric.SqEuclidean, ref idxBuf, ref distBuf);
// units in attack range:   int n = queryOP.rowsWithinRadius(in units, in self, range, Metric.Euclidean, ref idxBuf);
// utility-AI best action:  queryOP.normalizeColumns(ref C, NormalizeMode.MinMax);  int best = queryOP.argMaxScore(in C, in weights);
```
All read cleanly. Two gaps the snippets surfaced:
- **Masked / predicate-filtered nearest** ("closest *visible* enemy") has no direct API — it's the
  intersection of Group 3 (search) and Tier C (predicate). The one genuinely game-useful on-demand combo:
  `nearestRow(... ref P pred)` skipping rows where `pred` is false. Flag for Tier C, not core.
- **Buffer sizing** for `rowsWithinRadius` is worst-case `M_Rows`; document so callers don't undersize.

## Dogfooding (ops our own algorithms can reuse) — with one honest caveat
- `countNonzero` over R's diagonal (with tol) **is** numerical rank in QRCP / SVD. Clean reuse.
- **Partial-pivot caveat:** pivot search needs argmax-|·| over *column j, rows k..m-1* (a strided
  sub-range), which the flat generic `argMaxAbs<T>` does NOT express. So `argMaxAbs` is genuinely useful
  for whole-vec/whole-matrix abs-extremes, but the internal LU/QRCP/Cholesky pivots keep their tight
  inline loops — **don't contort the public API to match them.** (A column-range overload is possible
  on-demand, but not worth it for a game lib.)

## Explicitly SKIP (above-and-beyond for a game lib)
Full sort / `sortrows`, `unique`, histogram/`FREQUENCY`, `searchsorted` (needs sorted data), numpy
`where`-as-a-system / broadcast, k-d trees / spatial acceleration (brute-force O(n·d) is fine for
game-sized sets), negative indices / steps.

## Build phasing & acceptance
- **Phase 1 (core, fProxy):** Groups 1–4 for float/double. Enums file. Metric kernels.
- **Phase 2 (int):** `QueryOP.iProxy.cs` — the P2 integer-exact subset, with the P3 overflow doc.
- **Phase 3 (on-demand):** Group 5, `pairwiseDistances`, Tier-C functors, masked-nearest — only if wanted.
- **Tests** (`*Tests.fProxy.cs`): known-vector oracles for each metric; nearest/farthest/top-k vs a
  brute-force reference; `kNearest` ties & k>M_Rows clamping; within-radius boundary; row/col **symmetry**
  (column op on `A` == row op on `Aᵀ`); `intN` buffer + count contract; integer L1/Linf exactness + a
  documented-overflow case; mask-bridge `whichTrue`.

---
*Lean core to build first = `argMaxAbs` + per-axis `rowArgMin`/`rowArgMax` + `argMaxRowNorm` +
`nearestRow`/`farthestRow`/`kNearestRows`/`rowsWithinRadius` + `findValue`/`nonzero`/`countNonzero` + the
`BoolAnalysis` bool bridge. Everything else is on-demand.*
