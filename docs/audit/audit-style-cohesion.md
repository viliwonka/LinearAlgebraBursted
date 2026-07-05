# Style / Cohesion / Usability Audit — LinearAlgebra Burst (TemplateSource)

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

**Scope:** `Assets/LinearAlgebra/CodeGen/TemplateSource/**` (template source only — `Generated/` ignored), plus `docs/spec-*.md` cross-checks.
**Convention reminder:** `fProxy` → {float, double}; `iProxy` → {int, short, long}; `bool` is its own twin. The codegen expands each `*.fProxy.cs` / `*.iProxy.cs` / `*.bool.cs` into concrete per-type files. Findings on a template apply to every generated variant.

All line numbers are in the **template** files. Several HIGH correctness findings (acosh→acos, from-end indexing off-by-one, bool copy-ctor null deref) were independently re-verified by reading the source.

---

## 1. Executive Summary (top findings)

1. **`acosh` computes `acos` (HIGH, correctness).** `math/mathUnsafefProxy.cs:211` — the hyperbolic arc-cosine loop calls `math.acos`. Pure copy-paste from `acos`; returns numerically wrong results for every float/double consumer. One-line fix: `math.acosh`.
2. **`System.Index` from-end accessors are off-by-one vs C# (HIGH, silent wrong element).** `fProxyN.Indexing.cs:23` and `fProxyMxN.Indexing.cs:32,69,90,111-112` use `Length - 1 - index.Value`. `vec[^1]` (idiomatic "last") returns the **second-to-last** element; `vec[^0]` returns the last. No exception. Mirrored across all proxy/bool accessors, so it is *consistently* wrong. Bounds checks on 2D from-end indices also validate the unresolved value (`fProxyMxN.Indexing.cs:67,88,109`), so they give no protection.
3. **`bool` copy-constructors lack the null-arena guard that fProxy/iProxy already have (HIGH, null deref).** `boolN.cs:46` / `boolMxN.cs:51-52` dereference `_arenaPtr->Allocator` unconditionally. The fProxy twin (`fProxyN.cs:48-49`) was explicitly fixed (`_arenaPtr != null ? … : Allocator.Temp`); the fix was never propagated to bool. Copying a standalone bool vector/matrix crashes.
4. **The `OP`-suffix class-naming convention is applied inconsistently (HIGH, the headline cohesion issue).** Op-bag classes split into `*OP` (`fProxyGenOP`, `fProxyNormsOP`, `OrthoOP`, `SelectOP`, `fProxyStatsOP`, …) and bare names (`Cholesky`, `LU`, `SVD`, `Eigen`, `Solvers`, `fProxyFFT`, `Optimize`, `Analysis`, `BoolAnalysis`). `fProxyFFT` (`FFT.fProxy.cs:22`), `Optimize` (`Optimize.fProxy.cs:25`), and `Analysis`/`BoolAnalysis` are the clearest sore thumbs because they are plain function collections exactly like the `*OP` classes.
5. **`ref Random rng` parameter position is inconsistent — even inside one class (HIGH, usability).** Dominant pattern is rng-first, but `weightedPick`/`weightedPickInpl` (`RandomOP.fProxy.cs:164,179`) put it last *inside the otherwise rng-first `fProxyRandomOP`*, and the shared `RandomOP.cs` helpers (`:26,43,64`) are rng-last. Footgun for argument-order memory.
6. **`IfiProxyPredicate` ships with a copy-paste-mangled name (HIGH/MED, public API).** `Interfaces/PredicateQuery.iProxy.cs:9` — the integer predicate interface is `IfiProxyPredicate` (leftover `f` from `IfProxyPredicate`). Should be `IiProxyPredicate`. It is a visible public type.
7. **Integer Arena generators have real range bugs and an N==1 divide-by-zero (HIGH/MED, correctness).** `ArenaExtensions.iProxy.cs`: `iProxyLinVector` (`:63-73`) divides by `N-1` (N==1 → Inf→NaN→garbage int) even though the fProxy twin was migrated to the guarded `linspace`; `iProxyRandomVector` (`:54-57`) and `iProxyRandomDiagonalMatrix` (`:158-161`) pass an invalid `(min,max)` to `NextInt` in the `max<min` branch while `iProxyRandomMatrix` (`:180-181`) passes the valid `(max,min)` — three siblings, three behaviors.
8. **Matrix `operator *` is element-wise (Hadamard), not the matrix product (MED, major footgun).** `fProxyMxN.Operators.cs:149-159`. Every linear-algebra user expects `A * B` to be matmul; silent Hadamard (gated by `Assume.SameDim`) is a surprising contract for a *linear algebra* library.

Pervasive lower-severity themes: bare `System.Exception` vs `ArgumentException` thrown for the same class of error across files; the misspelled `Treshold` baked into the `Consts.fProxyZeroTreshold` public-ish name; class `<summary>` blocks that are just the glossary note "Inpl = inplace"; and several twin-parity gaps (bool missing `ToString`/`CopyTo`/two-form `select`; iProxy missing range-`dot`, generic `swap`, `IComparable`, `IMatrix`).

---

## 2. Naming

### Class naming (`OP` suffix + proxy prefix)
- **HIGH** — `OP` suffix split. With-suffix: `fProxyGenOP` (GenOP.fProxy.cs:21), `fProxyNormsOP` (NormsOP.fProxy.cs:9), `fProxyOP` (OP.Dot.fProxy.cs:13), `OrthoOP` (OrthoOP.fProxy.cs:15), `SelectOP`, `SwapOP`, `UtilityOP`, `BoolOP`, `fProxyStatsOP`, `fProxyHistogramOP`, `fProxyQueryOP`, `fProxyResampleOP`, `fProxyKMeansOP`, `fProxyRandomOP`. Without: `Cholesky`/`LU`/`SVD`/`Eigen` (decomposition algorithm names, defensible as a separate category), but also `Solvers` (Solvers.fProxy.cs:11), `fProxyFFT` (FFT.fProxy.cs:22), `Optimize` (Optimize.fProxy.cs:25), `Analysis`/`BoolAnalysis` (Analysis.fProxy.cs:8, BoolAnalysis.cs:8) — these are function collections and *should* carry `OP`. The worst pair: `Solvers` (no suffix) and `OrthoOP` (suffix) both hold solve routines. **Fix:** suffix `FFT`→`fProxyFFTOP`, `Optimize`→`OptimizeOP`, `Analysis`→`fProxyAnalysisOP`, `BoolAnalysis`→`BoolAnalysisOP`, and decide whether decomposition classes are an exempt category (document it) or also get suffixes.
- **LOW** — Arena-factory/extension class pattern is itself three-way inconsistent: `fProxyRealtimeArena` (RollingWindow.fProxy.cs:186, static extension), `ArenaExtensions` (BoolAnalysis.cs:119), and `partial struct Arena` instance methods (KMeans.Workspace.fProxy.cs:35). Pick one.

### Method casing & verb conventions
- **MED** — PascalCase vs camelCase mixed within single classes: `Solvers` has `SolveUpperTriangular` (Solvers.fProxy.cs:20) next to `conjugateGradient` (:148); `LU` has `luDecomposition` (camel) next to `LUSolve` (Pascal). `SwapOP` methods are PascalCase nouns `Vec`/`Rows`/`Columns` (SwapOP.cs:15,36,59) — should be camelCase verbs (`swapVec`/`swapRows`/`swapColumns`).
- **MED** — `Solve` word order: `LUSolve`/`choleskySolve`/`pinvSolve`/`qrDirectSolve` (verb last) vs `Solvers.SolveUpperTriangular`/`SolveQR` (verb first).
- **MED** — row/col argmin verb order differs by file: `rowArgMin`/`colArgMax` (QueryOP.fProxy.cs:90,231, axis-first) vs `argMaxRowBy`/`argMinColBy` (QueryOP.Predicate.fProxy.cs:333,438, arg-first).
- **MED** — three "writes into caller buffer" suffixes coexist: `Inpl` (most OPs), `Into` (ResampleOP `sampleAtInto`/`resampleInto`; Stats/Histogram `*Into`), and bare (`SelectOP.select`). Choose a house rule.
- **LOW** — `compModDiv` (OP.Component.fProxy.cs:123; fProxyN.Operators.cs:192; iProxyN.Operators.cs:246) drops the `Inpl` suffix its siblings (`compMulInpl`/`compDivInpl`) carry; should be `compModInpl`.
- **LOW** — `normalizeL2Inpl` (UnsafeOP.fProxy.cs:278) is the only `normalize*` with an `Inpl` suffix though `normalizeL1`/`normalizeLMax`/`normalizeLP` (:308,336,364) are equally in-place. Either suffix all four or none.
- **LOW** — `bitwiseLeftShift(int value, iProxy* TargetWShift, int n)` (UnsafeOP.iProxy.cs:295,301) — `TargetWShift` is PascalCase and cryptic; every other param is camelCase.

### Enum / norm naming
- **MED** — infinity-norm spelled three ways: `LInf` (NormsOP.fProxy.cs:33), `NormalizeLMax` (:105), `Norm.Linf`/`matrixLInf` (:166,270 — note `Linf` vs `LInf` capitalization drift). Settle on `LInf`.
- **LOW** — pervasive misspelling `Treshold` in the public-ish const `Consts.fProxyZeroTreshold`, propagated through Cholesky/SVD/Eigen/Ortho/Norms/MatrixMetrics/Optimize docs (e.g. OrthoOP.fProxy.cs:104, Optimize.fProxy.cs:84). Also `absTreshold`/"Treshold" in Debug.fProxy.cs:94,101,115. Rename to `Threshold`.

### GOOD
- BLAS-style `axpy`/`aypx`/`matMatDotTransA` (UnsafeOP.fProxy.cs:186,194) are clear and conventional.
- `row*`/`col*` symmetry in StatsOP and QueryOP search methods is uniform and predictable.
- iProxy correctly *adds* type-appropriate bitwise/shift operators rather than blindly mirroring the float template (iProxyN.Operators.cs:111-161).

---

## 3. Cohesion

- **MED** — `Solvers.SolveQR` (Solvers.fProxy.cs:134-138) is a thin forwarder to `OrthoOP.qrDirectSolve`: two discoverable entry points, different names/namespaces, same job. Drop or mark canonical-alias.
- **MED** — Matrix metrics split across classes: `trace`/`cond`/`rank` in `fProxyOP` (MatrixMetrics.fProxy.cs:14,30,51) vs `matrixL1`/`matrixL2`/`matrixLInf` in `fProxyNormsOP` (NormsOP.fProxy.cs:255,270,286), even though `cond`/`matrixL2`/`rank` all funnel through `SVD.singularValues`. Co-locate or cross-`<see>`.
- **MED** — Two Hilbert generators with different guards: `fProxyHilbertMatrix` (ArenaExtensions.fProxy.cs:266, requires `M>=2`) vs `fProxyHilbert` (Gallery.SPD.fProxy.cs:23, requires `n>=1`). Same matrix, two names, two namespaces, two contracts. Deprecate one.
- **MED** — Two-form (zero-alloc ref-dest primitive + allocating wrapper) convention is applied very consistently in OP.Dot/GenOP/SVD.Solvers/OrthoOP/StatsOP — **but** `SelectOP.bool.cs` (`:15,30,45,52`) ships only allocating forms (no `select(..., ref boolN dest)` primitive) while `SelectOP.fProxy.cs` has both. Bool can't go zero-alloc.
- **MED** — Workspace/factory entry points are inconsistent: `Arena.fProxyKMeansWorkspace(...)` is a plain instance method (KMeans.Workspace.fProxy.cs:43) but documented like an extension; `fProxyRollingWindow` IS a `this ref Arena` extension (RollingWindow.fProxy.cs:193). SVD/Ortho/KMeans/Realtime allocation entry points should follow one pattern.
- **MED** — `ArenaExtensions.Query` distance wrappers `fProxyDistancesToRow/Column` (ArenaExtensions.Query.fProxy.cs:23,34; iProxy:34,52) are plain `static`, not `this`-extensions, while every neighboring method is `this ref Arena arena`. Inconsistent call-site ergonomics within one feature.
- **MED** — `BoolOP` (BoolOP.cs:18-103) mutates its receiver in place but no method carries `Inpl` and the class summary literally says "Inpl = inplace". `a.or(b)` silently overwrites `a`.
- **LOW** — Component add/sub are overloads of `addInpl`/`subInpl` (OP.Component.fProxy.cs:53,63) but component mul/div/mod are distinct `compMulInpl`/`compDivInpl`/`compModDiv` (:107,115,123). A user can't guess whether "component add" is `addInpl` or `compAddInpl`.
- **LOW** — `IMatrix<T>` (Interfaces.cs:51-67) ends in `// Other necessary properties…` and is implemented only by `fProxyMxN` (as throwing `NotImplementedException` stubs, fProxyMxN.cs:112-118); `iProxyMxN` does not implement it. Orphaned/stub interface; generic `IMatrix<T>` code works for float, not int.
- **LOW** — `Arena.AllocationsCount`/`TempAllocationsCount` (Arena.cs:11-31) count only fProxy/iProxy vectors+matrices, silently excluding bool/Pivot/Indices buffers — `AllAllocationsCount` is misleading.
- **LOW** — `UtilityOP.zeroInpl` (UtilityOP.cs:12) has only a vector overload; no matrix variant.

### GOOD
- KMeans 4-overload matrix (workspace×{explicit/default init} + allocating×{…}) with forwarding, and a fully shape-documented workspace struct — exemplary.
- Query metric kernels (`RowScore`/`ColScore`) shared `internal` so Arena two-pass alloc reuses them rather than duplicating.

---

## 4. Usability / Ergonomics / Footguns

- **MED** — Range-`dot` (OP.Dot.fProxy.cs:27) validates only `a.N == b.N`, never `0 <= start < end <= N`, then calls the unguarded `vecDotRange` → OOB read. Compare `L2Range` (NormsOP.fProxy.cs:43-47) which guards. Also no XML doc and no iProxy twin.
- **MED** — `OrthoOP.householderInpl` (OrthoOP.fProxy.cs:19-23) guards `IsSquare` first, making the second guard `matrix.M_Rows < matrix.N_Cols` ("must be square or tall") unreachable — the method contradicts its own "tall is allowed" message.
- **MED** — `in` vs `ref` is dishonest for read-only matrices: `conjugateGradient(in A,…)`/`determinant(in LU,in P)` use `in`, but `choleskySolve(ref L,…)` (Cholesky.fProxy.cs:89), `LUSolve(ref LU,…)` (LU.fProxy.cs:221), `SolveQR(ref Q, ref R,…)` (Solvers.fProxy.cs:107) pass un-mutated matrices by `ref`, falsely signalling "this gets destroyed."
- **MED** — Extension-method (`this`) usage is non-uniform in OP.Component: `divInpl<T>(this T place,…)` (:35) is an extension but `addInpl<T>(T place,…)`/`mulInpl<T>(T place,…)` (:19,27) are not, so `v.divInpl(2)` chains but `v.addInpl(2)` doesn't compile.
- **MED** — `matVecDot(mat, x, y)` (UnsafeOP.fProxy.cs:86) has `x`=input,`y`=output; `vecMatDot(y, mat, x)` (:102) has `y`=input,`x`=output. Sibling routines invert the same two letters' roles → silent buffer mis-wiring. Rename to `xIn`/`yOut`.
- **MED** — No zero-norm guard in any `normalize*` (UnsafeOP.fProxy.cs:278-393): all-zero vector → divide-by-0 → Inf/NaN written back, undocumented.
- **MED** — `RollingWindow` indexer `this[i,f]` and `GetSample(i,…)` (RollingWindow.fProxy.cs:78,85) don't bounds-check `i` against `Count`; `RingRow(i)=(OldestRow+i)%_capacity` silently wraps, returning stale ring rows for `i>=Count`. Per-frame off-by-one footgun.
- **MED** — Static `Pivot.ApplyVecInpl`/`ApplyRowInpl`/`ApplyColumnInpl` (Pivot.Operations.cs:23,47,71) have no dimension guard while the instance forms (:95,112,128) do — the lower-level, more dangerous entry points are the unguarded ones (OOB).
- **MED** — `BoolOP.equals`/`notEquals` (BoolOP.cs:79,97) return `void` and overwrite `a` with the equality mask — named like a comparison, behaves destructively. High surprise.
- **LOW** — `iProxyRandomMatrix` default `min=-1,max=1` → `NextInt(-1,1)` only yields {-1,0} (never 1) (ArenaExtensions.iProxy.cs:143-146), surprising and undocumented.
- **LOW** — `Indices` buffer is `UninitializedMemory` (Indices.cs:40-41) but the summary calls it a "zero-alloc index buffer" without warning that entries `[count..N)` are garbage after a partial fill (e.g. `whichTrue`).

### GOOD
- Destructive ops are loudly flagged in XML ("A is DESTROYED": SVD.Solvers.fProxy.cs:20,191; Eigen.fProxy.cs:179,368). Result-aliasing and must-not-alias guards in OP.Dot/Eigen are explicit.
- Optimizer overload ladders (full → defaulted) with each default named in the doc; `bisection` sign-equality test to dodge underflow/NaN (Optimize.fProxy.cs:46-47).
- KMeans `seed==0u?1u` and `(int)(rng.NextFProxy()*N)→min(...,N-1)` clamp defend the two classic RNG footguns.

---

## 5. Docs (XML quality & accuracy)

- **MED** — `densityInto`/`cdfInto` docs say interval is half-open `[lo, hi)` (HistogramOP.fProxy.cs:142,184) but the inherited `histogramInto` behavior is inclusive `[lo, hi]` (`x==hi`→last bin, :58-60). Docs contradict behavior.
- **MED** — Stale complexity: `KMeansPlusPlus = … O(k²·N·D)` (KMeansEnums.cs:5) — the seeding was made incremental at `O(k·N·D)` (KMeans.fProxy.cs:352-355) and even the helper comment says so.
- **MED** — iProxy `argMaxRowNorm`/`argMaxColNorm` docs claim L1 is "overflow-safe" (QueryOP.iProxy.cs:289-291,337-339), contradicting the explicit Manhattan overflow warnings (:416,470). L1 is a running sum that overflows identically.
- **MED** — `SelectOP` class carries a method's `<summary>`/`<param>`/`<returns>` on the *class* declaration (SelectOP.fProxy.cs:9-14; SelectOP.bool.cs:8-13), duplicated on both partials; the individual `select` overloads have no docs.
- **MED** — `gradientDescent` (Optimize.fProxy.cs:191-192) uses plain `//`, not `///` — the one optimizer with a non-obvious workspace-sizing trap (`g.N==x.N`) is the one whose contract won't surface in IntelliSense.
- **MED** — Copy-paste/stale constructor docs: `fProxyN.cs:40` & `iProxyN.cs:64` say "Creates a copy of vector with new allocation" above the `(int n, Allocator, bool uninit)` *empty*-vector ctor; `fProxyMxN.cs:38-42` documents a non-existent `allocator` param and a single `N` dimension for a `(M_rows,N_cols,arena)` ctor.
- **LOW, pervasive** — Class `<summary>` is frequently just "Inpl = inplace" — including on classes with no `Inpl` methods (OP.Dot.fProxy.cs:10-12, MatrixMetrics, LU, SVD, Eigen, Solvers, OrthoOP). Not a description. (Cholesky.fProxy.cs:10-14 is the good model.)
- **LOW** — Many public methods undocumented: all of `NormsOP` L1/L2/LInf/Normalize* (NormsOP.fProxy.cs:12-150), most `OP.Component` ops, `SwapOP`, `UtilityOP`, `WindowType` (uses `//` not `///`). Reduces IntelliSense discoverability.
- **LOW** — `idft`/`ifft` throw error messages prefixed with the forward op name ("dft:"/"fft:") because they share `DftCore`/`FftCore` (FFT.fProxy.cs:178,180,186 reached via :169; :51,53 via :42).
- **LOW** — Guard messages name the wrong method (copy-paste): `luDecompositionNoPivot` throws "luDecomposition:" (LU.fProxy.cs:28,31,34); `L2Range` throws "NormsOP.L2:" (NormsOP.fProxy.cs:44,46); `SwapOP.Vec` two guards share one message though each checks only `i` or only `j` (SwapOP.cs:17,21).
- **LOW** — `fProxyFullStats.ToString()` omits `count` (Structs.fProxy.cs:51-54), the first field. `StatsOP.fProxy.cs:12` carries stale "just a prototype, needs matrices handling too" above a class with a full matrix region.

No token-leak artifacts (e.g. "float-only (float/double)") were found; the "fProxy-only"/"float-only" tags that exist are legitimate type constraints, not stale codegen text.

---

## 6. Mistakes (incidental)

- **HIGH** — `acosh` body calls `math.acos` (mathUnsafefProxy.cs:211). **Verified.** Fix: `math.acosh`.
- **HIGH** — From-end indexing off-by-one (`Length-1-Value`) in fProxyN.Indexing.cs:23 and fProxyMxN.Indexing.cs:32,69,90,111-112. **Verified.** `vec[^1]`→second-to-last; `vec[^0]`→last. Fix: `Length - Value` / `N_Cols - indexC.Value` / `M_Rows - indexR.Value`.
- **HIGH** — bool copy-ctor null-deref: `boolN.cs:46` / `boolMxN.cs:51-52` dereference `_arenaPtr->Allocator` without the `_arenaPtr != null` guard present in `fProxyN.cs:48-49`. **Verified** (fProxy guarded, bool not). Propagate the fix.
- **HIGH** — Non-square interop indexers (proxyStructs.math.cs) return the wrong vector width and use the wrong column bound: `fProxy2x3` returns `ref fProxy3`/checks `>=2` (:98,101), `fProxy3x2` returns `ref fProxy2`/checks `>=3` (:145,148); same for `2x4`,`4x2`,`3x4`,`4x3` (:115,118,193,196,162,165,209,212). Reads past column memory / wrong bound. Square structs are correct. (This is the *parked* interop stub — likely unused today — but it is latent memory corruption. Fix or delete.)
- **HIGH/MED** — Integer Arena range bugs: `iProxyLinVector` N==1 divide-by-zero (ArenaExtensions.iProxy.cs:63-73); inverted-range branches feed invalid `(min,max)` to `NextInt` in `iProxyRandomVector` (:54-57) and `iProxyRandomDiagonalMatrix` (:158-161) while `iProxyRandomMatrix` (:180-181) uses valid `(max,min)`. The "reverse iteration on max<min" device accomplishes nothing observable.
- **MED** — `float`-typed scalars leak into the `double` variant: `fProxyVec(int N, float s)` (Arena.fProxy.cs:23), `fProxyMat(…, float s)` (:71) hardcode `float` instead of `fProxy s` (the iProxy twin correctly uses `iProxy s`). Silent narrowing in the double build.
- **MED** — `Solvers`/`NormsOP`/`SwapOP.Vec`/Arena generators throw bare `System.Exception` (Solvers.fProxy.cs:23,45,…; NormsOP.fProxy.cs:44,46,…; SwapOP.cs:18,22; ArenaExtensions.fProxy.cs:36,192,269) while the rest of the library — and even sibling methods in the same class — throw `ArgumentException`. Some docs in those very classes *claim* `ArgumentException`. Callers can't write one `catch`.
- **MED** — `M_Rows`/`N_Cols` are mutable public fields but `Length` is `readonly` (fProxyMxN.cs:12-13,20; iProxyMxN.cs:13-14,21; boolMxN.cs:12-13,20). Assigning a dimension without resizing `Data` desyncs `Length` and indexing → silent corruption. Make `{ get; private set; }` or readonly.
- **MED** — Proxy struct equality/comparison parity gaps: `iProxy` lacks `IComparable<iProxy>` that `fProxy`/`anyProxy` have (proxyStructs.cs:54 vs 7,108) → sort/top-k generic constraints fail for int; `fProxy`/`anyProxy` define `operator ==` with `Equals` commented out (proxyStructs.cs:36-45,135-144) while `iProxy` overrides `Equals` (:83) — inconsistent equality, CS0660/CS0661-class smell.
- **LOW** — `UnsafeOP.bool.cs:31-34` `swapColumns` missing `[BurstCompile]` (sibling `swapRows` has it). `Analysis.fProxy.cs:4` `using Unity.Burst;` unused (no `[BurstCompile]`). `ArenaExtensions.iProxy.cs:3` stray `using UnityEngine.UIElements;`. Redundant `using LinearAlgebra;` inside `LinearAlgebra.*` namespaces (Gallery files, KMeans.Workspace.fProxy.cs:1, RollingWindow.fProxy.cs:8). Unused `System.Collections*` in Pivot files.
- **LOW** — Dead/leftover code: commented square-check in `OrthoOP.genHouseholderPete` (OrthoOP.fProxy.cs:82-83) and `Analysis.IsOrthogonal`/`IsDiagonal` (Analysis.fProxy.cs:240-241); double semicolon OrthoOP.fProxy.cs:110; dead `if (l < 0)` in `eigenvaluesQR` (Eigen.fProxy.cs:476); leftover dev TODOs (UnsafeOP.iProxy.cs:324 "do something about bitwise…", OP.Component.fProxy.cs:56-57 bug-narrative comment); commented `Equals`/`modf` in proxyStructs.cs / mathUnsafefProxy.cs; `mathUnsafefProxy.cs:264 mod` vs `:306 fmod` redundant for float.
- **LOW** — `[BurstCompile]` on POCO data structs is inert and applied inconsistently: `fProxyFullStats` has it (Structs.fProxy.cs:21), `fProxyMeanMinMaxRangeStats` (:5) doesn't; `BoolOP` (BoolOP.cs:11) is a plain static-method class. `fProxyFullStats.count` is stored as `fProxy` (float) though logically integer — precision loss past 2^24.

---

## 7. Ambiguity (contracts)

- **MED** — `dot` mat×mat supports `transposeA` but not `transposeB` (OP.Dot.fProxy.cs:128; iProxy:115): no way to compute `A·Bᵀ`. Document or add the twin.
- **LOW** — Empty/failure sentinels undocumented or surprising: `cond` returns `0` for an empty matrix (MatrixMetrics.fProxy.cs:33-34) — reads as "perfectly conditioned" — vs `+Inf` for singular; `SVD.singularValues` swallows `svdDecomposition`'s convergence bool (SVD.Metrics.fProxy.cs:36,42) so `cond`/`rank`/`matrixL2` silently use unconverged values; `fft` throws on N==0 while `dft` silently returns (FFT.fProxy.cs).
- **LOW** — Aliasing contracts are documented but with opposite defaults across families (`multivariateNormalInpl`/`ResampleOP` "caller responsible, no guard" vs `SelectOP` "aliasing dest is safe"). Individually fine; cognitive load when moving between OPs.
- **LOW** — Tolerance comparison asymmetry: `findValue` uses `<= tol` (QueryOP.fProxy.cs:916) while `nonzero`/`countNonzero` use `> tol` (:898,932). Both documented, but a subtle gotcha across siblings.
- **LOW** — Vector `/` and `%` scalar forms throw `DivideByZeroException` but vector÷vector / reverse-scalar forms fall to IEEE silently (fProxyN.Operators.cs:69-70,78,92-93,101); matrix file documents the choice, vector file doesn't. Component `==`/`!=` call `Assume.SameDim` and *throw* on length mismatch rather than returning a defined result (fProxyN.Comparators.cs:127,141) — undocumented.
- **LOW** — NaN policy is uneven within StatsOP: `argmin`/`argmax` documented "unspecified" (StatsOP.fProxy.cs:81,101), `min`/`max` use `math.min/max` with no note (:118,132), transforms use NaN-safe `!(x>0)`; Histogram explicitly drops NaN. A one-line module NaN policy would resolve it.

### GOOD
- Histogram NaN/out-of-range contract is precise and implemented exactly (`!(x>=lo && x<=hi)` drops NaN by construction). Failure/"read outputs only on true" contracts in Cholesky pivot, `eigenvaluesQR`, CG are explicit. The NaN-safe `!(x > 0)` rejection idiom is used uniformly across Cholesky/SVD/Ortho/Solvers/Norms.

---

## 8. Prioritized Findings Table

| Severity | File:Line | Issue | Suggested fix |
|---|---|---|---|
| HIGH | math/mathUnsafefProxy.cs:211 | `acosh` calls `math.acos` (wrong results) | `x[i] = math.acosh(x[i]);` |
| HIGH | fProxy/fProxyN.Indexing.cs:23; fProxyMxN.Indexing.cs:32,69,90,111-112 | `System.Index` from-end off-by-one; `^1` returns 2nd-to-last, silently | Use `Length - index.Value` (and `N_Cols/M_Rows - Value`) |
| HIGH | fProxyMxN.Indexing.cs:67,88,109 | 2D bounds check runs on unresolved from-end value | Resolve r/c first, then bounds-check |
| HIGH | bool/boolN.cs:46; bool/boolMxN.cs:51-52 | Copy-ctor derefs `_arenaPtr` without null guard (fProxy fix not propagated) | Mirror `_arenaPtr != null ? …->Allocator : Allocator.Temp` |
| HIGH | proxyStructs.math.cs:98-212 (non-square structs) | Indexers return wrong vector width + wrong bound → OOB | Return correct `fProxyK`, fix `>= cols` bound; or delete parked stub |
| HIGH | ArenaExtensions.iProxy.cs:63-73 | `iProxyLinVector` divides by `N-1` (N==1 → garbage int) | Route through guarded `linspace` / special-case N==1 |
| HIGH | ArenaExtensions.iProxy.cs:54-57,158-161 | Inverted-range branch passes invalid `(min,max)` to `NextInt` | Use valid `(max,min)` or clamp/swap; share one impl |
| HIGH | Interfaces/PredicateQuery.iProxy.cs:9 | Public interface misnamed `IfiProxyPredicate` | Rename to `IiProxyPredicate` |
| HIGH | FFT.fProxy.cs:22; Optimize.fProxy.cs:25; Analysis.fProxy.cs:8; BoolAnalysis.cs:8 | Op-bag classes lack the `OP` suffix every other op class uses | `fProxyFFTOP`/`OptimizeOP`/`fProxyAnalysisOP`/`BoolAnalysisOP` |
| HIGH | RandomOP.fProxy.cs:164,179; RandomOP.cs:26,43,64 | `ref Random rng` position inconsistent (rng-last inside rng-first class) | Standardize on rng-first |
| MED | fProxyMxN.Operators.cs:149-159 | `A * B` is element-wise, not matrix product | Document prominently or reserve `*` for matmul |
| MED | fProxyMxN.cs:12-13,20 (+iProxy,bool) | Mutable `M_Rows`/`N_Cols`, readonly `Length` → desync/corruption | `{ get; private set; }` or readonly |
| MED | OP.Dot.fProxy.cs:27 | Range-`dot` has no bounds guard → OOB; no doc; no iProxy twin | Add `0<=start<end<=N` guard; mirror to iProxy |
| MED | OrthoOP.fProxy.cs:19-23 | `householderInpl` square guard makes "tall" guard unreachable | Drop square check or the misleading tall message |
| MED | UnsafeOP.fProxy.cs:86,102 | `matVecDot`/`vecMatDot` invert x/y input-output roles | Rename to `xIn`/`yOut` consistently |
| MED | UnsafeOP.fProxy.cs:278-393 | No zero-norm guard in `normalize*` → Inf/NaN | Early-out / document contract |
| MED | RollingWindow.fProxy.cs:78,85 | Indexer/`GetSample` not bounds-checked vs `Count`, ring wraps | `if ((uint)i >= (uint)_count) throw` |
| MED | Pivot.Operations.cs:23,47,71 | Static `Apply*Inpl` primitives lack dimension guards | Add guards matching instance forms |
| MED | BoolOP.cs:79,97 | `equals`/`notEquals` destructive + named like comparisons | Rename (`eqInpl`/`xnorInpl`) + add `Inpl`; loud doc |
| MED | SelectOP.bool.cs:15-52 | No ref-dest `select` primitive (two-form parity gap vs fProxy) | Add `select(..., ref boolN dest)` |
| MED | Arena.fProxy.cs:23,71 | `float s` hardcoded → narrows in double build | Use `fProxy s` |
| MED | Solvers/NormsOP/SwapOP.Vec/Arena generators | Bare `System.Exception` for arg errors elsewhere `ArgumentException` | Use `ArgumentException` uniformly |
| MED | proxyStructs.cs:54 (iProxy) | No `IComparable<iProxy>`; fProxy/anyProxy have it | Add `IComparable<iProxy>`+`CompareTo` |
| MED | Cholesky/LU/Solvers ref params | Read-only matrices passed `ref` (falsely signals "destroyed") | Use `in` for non-mutated inputs |
| MED | HistogramOP.fProxy.cs:142,184 | density/cdf docs say `[lo,hi)` but behavior is `[lo,hi]` | Fix brackets / state closed upper edge |
| MED | KMeansEnums.cs:5 | Stale `O(k²·N·D)` (now incremental `O(k·N·D)`) | Update complexity |
| MED | QueryOP.iProxy.cs:289-291,337-339 | L1 "overflow-safe" claim contradicts Manhattan warnings | Qualify ("sum may overflow; Linf is safe") |
| MED | SelectOP.fProxy.cs:9-14; SelectOP.bool.cs:8-13 | Method `<summary>/<param>/<returns>` placed on the class | Move docs onto the overloads |
| MED | Optimize.fProxy.cs:191-192 | `gradientDescent` uses `//` not `///` (has workspace trap) | Promote to `///` with `<param>` |
| MED | fProxyN.cs:40; fProxyMxN.cs:38-42 (+iProxy) | Stale/wrong constructor XML docs | Rewrite to match actual ctor params |
| MED | QueryOP.fProxy.cs:90,231 vs Predicate:333,438 | row/col argmin verb order differs by file | Align to one order |
| MED | Solvers.fProxy.cs:134-138 | `SolveQR` duplicates `OrthoOP.qrDirectSolve` | Drop or mark canonical alias |
| MED | ArenaExtensions.fProxy.cs:266 vs Gallery.SPD.fProxy.cs:23 | Two Hilbert generators, different guards/namespaces | Deprecate one, align guard |
| LOW | OP.Component.fProxy.cs:123; fProxyN.Operators.cs:192 | `compModDiv` missing `Inpl` suffix | Rename `compModInpl` |
| LOW | SwapOP.cs:15,36,59 | PascalCase noun methods `Vec`/`Rows`/`Columns` | camelCase verbs `swapVec`/`swapRows`/`swapColumns` |
| LOW | NormsOP.fProxy.cs:33,105,166,270 | Infinity norm spelled `LInf`/`LMax`/`Linf` | Settle on `LInf` |
| LOW | Consts (fProxyZeroTreshold) + many docs | "Treshold" misspelling in public-ish name | Rename `Threshold` |
| LOW | OP.Dot/MatrixMetrics/LU/SVD/Eigen class summaries | `<summary>` is just "Inpl = inplace" (even where no Inpl) | Write real class summaries |
| LOW | NormsOP/Component/SwapOP/UtilityOP/WindowType | Public methods undocumented or `//` not `///` | Add `///` summaries |
| LOW | UnsafeOP.bool.cs:31-34 | `swapColumns` missing `[BurstCompile]` | Add the attribute |
| LOW | Analysis.fProxy.cs:4; ArenaExtensions.iProxy.cs:3; Pivot/Gallery usings | Unused/redundant usings | Remove |
| LOW | Multiple (see §6) | Dead/commented code, leftover TODOs, double `;` | Remove |
| LOW | Structs.fProxy.cs:51-54,21 | `ToString()` omits `count`; `count` stored as float; inert `[BurstCompile]` | Include count; store int; drop attr |
| LOW | OP.Dot.fProxy.cs:128 | mat×mat has `transposeA` but no `transposeB` | Document or add twin |
| LOW | MatrixMetrics.fProxy.cs:33-34; SVD.Metrics.fProxy.cs:36,42 | `cond` empty→0 (reads as well-conditioned); convergence swallowed | Document / return NaN; document swallow |
| LOW | bool/boolN.cs, boolMxN.cs | Missing `ToString`/`CopyTo`/`(int n,Allocator)` ctor (twin gap) | Add for parity |

---

*End of audit. Counts: ~10 HIGH, ~22 MED, ~25 LOW across 6 themes. The HIGH correctness items (acosh, from-end indexing, bool copy-ctor) are independently verified against source and are cheap, high-value fixes.*
