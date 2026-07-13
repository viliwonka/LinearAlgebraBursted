# Release scan 2026-07-13 — N4 narrow pass: TemplateSource/OP, QueryCore.Predicate.fProxy.cs → SVD.Metrics.fProxy.cs

Partition (24 files, case-insensitive alphabetical): QueryCore.Predicate.{fProxy,iProxy},
QueryEnums, QueryOP.{fProxy,iProxy}, QueryOP.Predicate.{fProxy,iProxy}, RandomMatrixOP.fProxy,
RandomOP.{bool,cs,fProxy,iProxy}, ResampleEnums, ResampleOP.fProxy, SelectOP.{bool,fProxy,iProxy},
SimdMath, SolveInfo, SolveStatus, SVD.fProxy, SVD.FullWorkspace.fProxy, SVD.LowRank.fProxy,
SVD.Metrics.fProxy.

Every line of every file read. Siblings diffed (fProxy vs iProxy Query/Select/Random; the four
QueryCore helper/predicate files; the three truncated-SVD allocating overloads). TemplateConverter.cs
and GenUtils.cs read first; //+choose does not appear in this partition — type-splits here are via
per-type Consts tokens, `alsoExpand[uint]` (SelectOP.iProxy only), and separate iProxy files.
Addendum patterns 1-7 swept explicitly (grep + read): no rename stragglers (maxIter/tol/BSM/Elem/_OP
all absent), no TODO/FIXME, no role-swapped InPlace wrappers, no same-pointer-two-NoAlias call sites
found in-library (but see M-3 for the contract-level problem).

---

## MEDIUM

### M-1. SimdMath.cs goes through the WRONG codegen path — generated file ships a mangled comment
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SimdMath.cs:7-17`
- **Defect:** The header comment claims *"it contains no fProxy/iProxy token, so codegen treats it
  as a singular file"* — but the comment itself contains the literal tokens `fProxyM`, `fProxy4`,
  and `fProxy/iProxy`, and `TemplateConverter.Execute` decides singular-vs-multiplied by
  `sourceCode.Contains(GenUtils.fProxy)` over the WHOLE file text, comments included. The file has
  no `//singularFile//` marker, its filename contains no proxy, so it falls into the **iProxy
  multiplying branch** and is emitted three times (int/short/long) to the SAME output path, last
  writer wins. Verified in the generated output — `Assets/LinearAlgebra/Source/OP/SimdMath.cs:17`
  ships: `"it contains no fProxy/long token, so codegen treats it as a singular file"` (the
  iProxy-to-long substitution mangled the comment). Code is unaffected (classes are hand-written
  `floatM`/`doubleM`), but the shipped package contains a nonsense auto-generated comment, and the
  triple same-path `context.AddCode` is latent codegen fragility.
- **Fix direction:** add `//singularFile//` as line 1 of the template (like QueryEnums.cs), or
  reword the comment so it doesn't spell the raw tokens.

### M-2. SolveInfo.cs (singular file): `fProxy*` type names inside crefs survive verbatim into the generated package
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SolveInfo.cs:229-236` (RankInfo summary)
- **Defect:** The file is `//singularFile//` (correct — it defines shared structs once), but the
  RankInfo doc contains crefs with proxy-typed parameter lists:
  `<see cref="QRCP.solveInPlace(ref fProxyMxN, ref fProxyN, ref fProxyN, ref fProxyMxN, ref Pivot, ref fProxyN, fProxy)"/>`,
  `CHOP.decomp(in fProxyMxN, ..., ref fProxyCHOPCache)`, `SVD.pinvSolve(ref fProxyMxN, ...)`. Singular
  files receive NO proxy substitution, and the generated `Source/OP/SolveInfo.cs:233-236` (verified)
  ships these crefs referencing types (`fProxyMxN`, `fProxyCHOPCache`) that do not exist in the
  generated assembly — unresolvable crefs (CS1574-class doc warnings) and confusing doc text in the
  released source.
- **Fix direction:** drop the parameter lists from the crefs (`<see cref="QRCP"/>` /
  `<c>QRCP.solveInPlace</c>` prose), as the rest of this same file already does.

### M-3. SelectOP: `[NoAlias]` annotations contradict the wrapper's own documented aliasing guarantee
- **Files:** `SelectOP.fProxy.cs:14-15 + 103`, `SelectOP.iProxy.cs:16-17 + 105`,
  `SelectOP.bool.cs:60` (all three siblings identical)
- **Defect:** The public wrappers state *"the destination may alias a or b safely"* and pass
  through to `UnsafeSelectOP.selectfProxy([NoAlias] fProxy* a, [NoAlias] fProxy* b, [NoAlias] bool* c, fProxy* target, ...)`.
  Burst's `[NoAlias]` on a parameter asserts it aliases NO other parameter — including `target` —
  so a caller who takes the comment's invitation (`select(in x, in y, in c, ref x)`) violates the
  kernel's declared contract, and so does the perfectly legal `select(in x, in x, in c, ref d)`
  (same pointer into the two `[NoAlias]` params a and b). Today's scalar loop happens to be
  same-index elementwise so miscompilation is unlikely, but the annotation licenses Burst to
  reorder/vectorize on a promise the public API explicitly does not enforce or even keep.
  (Addendum pattern 4.)
- **Fix direction:** remove `[NoAlias]` from a/b (keep it on c only if c-overlap is truly
  forbidden), or keep the annotations and change the wrapper contract/comment to forbid aliasing.

### M-4. SVD.Metrics.singularValues silently swallows the inner SVDInfo — non-convergence is undetectable by matrixL2 / cond / rank
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Metrics.fProxy.cs:29-37`
- **Defect:** `values(in A, ref S);` / `values(in At, ref S);` — the returned `SVDInfo` is
  discarded and `singularValues` returns `k` unconditionally. Per `values`'s own contract, on
  MaxIterations **S is unwritten**; the three consumers (`Norms.matrixL2` reads `S[0]`,
  `Analysis.cond` reads `S[0]/S[k-1]`, `Analysis.rank` counts against `S[0]` — all verified to do
  no convergence check) then compute from a never-filled buffer with no way for any caller to
  detect it. This breaks the library's own diag-struct convention (every SVD entry point returns
  SVDInfo) exactly one call-layer up. Rare path (pathological spectra exhausting
  `Consts.sweepBudget`), which is why this is MEDIUM not HIGH — but it is a genuinely silent
  wrong-result path.
- **Fix direction:** return `SVDInfo` (or an `out` status) from `singularValues` and propagate a
  failure signal through matrixL2/cond/rank (e.g. NaN + documented), or at minimum document the
  behavior on non-convergence at all four sites.

---

## LOW

### L-1. SVD.thin XML doc still states the retired bool contract
- `SVD.fProxy.cs:161-163`: *"Returns true on convergence; false if the bidiagonal QR hit
  maxIterations"* — the method returns `SVDInfo`. The sibling `values` doc (line 22-24) was
  updated to the SVDInfo wording; thin's wasn't. Implicit-bool keeps it truthy, but the sentence
  misdescribes the return type. Fix: mirror values's "Returns an SVDInfo (implicit-bool ==
  Converged)" phrasing.

### L-2. SVDInfo.converged doc's "Equals n iff status is Converged" is violated by truncated
- `SolveInfo.cs:199-201` vs `SVD.LowRank.fProxy.cs:445-515`: `truncated` returns
  `converged = innerConverged` (count over the INNER pxp problem, p = k+oversample) and can return
  `status = MaxIterations` from the residual check or the `kOut < k` rank-deficiency path even
  when the inner QR fully converged (`converged == p`). So neither the "n" nor the "iff" holds for
  truncated-produced infos. truncated's own doc says the counters are the inner QR's — the struct
  doc just overclaims. Fix: soften the struct doc ("for thin/values equals n iff Converged;
  GKL/randomized routes report their inner problem's counters").

### L-3. QueryOP.fProxy duplicates the similarity-metric rule inline; iProxy sibling uses the helper
- `QueryOP.fProxy.cs:710,752,794,836,876,899` and `QueryOP.Predicate.fProxy.cs:194,269`:
  `bool sim = m == Metric.Cosine || m == Metric.Dot;` — while `countWithinRadius` (line 660) and
  all of QueryOP.iProxy consistently call `*QueryCore.IsSimilarityMetric(m)`. Same semantics
  today; a future metric addition has 8 extra places to miss. Fix: route the inline copies through
  `fProxyQueryCore.IsSimilarityMetric`.

### L-4. RandomOP.bool doc recommends a "double" Bernoulli path that does not exist
- `RandomOP.bool.cs:25-26`: *"Use double if exact fidelity near the boundaries is required"* —
  `nextBernoulliInPlace` only exists with `float p` (grep: no other Bernoulli API in the
  codebase). The advice points at a nonexistent overload. Fix: reword ("draw doubles yourself and
  threshold") or add the overload.

### L-5. RandomOP.iProxy range guard is an always-false comparison in the short variant
- `RandomOP.iProxy.cs:36,62`: `if (min < int.MinValue || max > int.MaxValue)` — for the `short`
  expansion both comparisons are always false (the file header even says so), and the constants
  lie outside short's range, which the C# compiler may flag as CS0652 in the generated `short`
  file. Harmless and self-documented; noting only because it ships in generated public source.
  Fix (optional): gate the guard per-type (skipFor/choose) instead of relying on the always-false
  branch.

### L-6. SVD.LowRank: float-suffixed literals and a float-resolution seed inside the double variant
- `SVD.LowRank.fProxy.cs:138,182,289`: `(fProxy)1.5f`, `(fProxy)1.01f` — in the double variant
  these become `(double)1.01f = 1.0099999904632568` etc. Both are order-of-magnitude fudge factors
  (omega-recurrence floor, anorm safety factor), so numerically harmless — but they are exactly
  addendum pattern 6 and trivial to write as `(fProxy)1.01`.
- `SVD.LowRank.fProxy.cs:113`: `v0[i] = (fProxy)(rng.NextFloat() * 2f - 1f);` — the double
  variant's Lanczos start vector has float-resolution entries. Harmless (any non-degenerate start
  vector works; may even be a deliberate cross-precision determinism choice) — if deliberate, a
  one-line comment or DEVLOG entry would stop future scanners re-flagging it.

### L-7. Comment-policy: perf/dev-history narration in SVD comments
- `SVD.fProxy.cs:201-203`: *"...(unit-stride, SIMD via UnsafeOP.jacobiRotate) instead of strided
  columns — the same trick that vectorized Eigen.symmetricInPlace"* and `SVD.fProxy.cs:378-380`:
  *"needed for FLOAT to converge on clustered/zero singular values (same lesson as the symmetric
  eigen QL)"* — perf-verdict / cross-file history references, not contracts.
  Proposed DEVLOG entries (OP/DEVLOG.md, under `## SVD`):
  - `2026-07-13 | thin transposes U/V so bidiagonal QR rotations hit contiguous rows (same
    vectorization approach as Eigen.symmetricInPlace). (was SVD.fProxy.cs:201)`
  - `2026-07-13 | bidiagonalQR deflation threshold is relative to the GLOBAL anorm, not local
    |d|+|e| — float needs this on clustered/zero singular values (same finding as the symmetric
    eigen QL). (was SVD.fProxy.cs:378)`
- `SVD.LowRank.fProxy.cs:50`: *"Existing call sites without this parameter receive partialReorth =
  true."* — call-site history in a public XML doc; the preceding sentence ("Default is true.")
  already states the contract. Fix: delete the sentence.

### L-8. SVD.Metrics doc: "Allocates SVD scratch from A's arena" is inaccurate
- `SVD.Metrics.fProxy.cs:14`: the tall path calls the allocating `values` overload, which uses
  `Allocator.Temp` internally (`SVD.fProxy.cs:43-44`), not A's arena; only the wide path's
  `Blas.trans(A)` uses the arena temp pool. Fix: "Allocates temporary scratch internally."

### L-9. lowRankApprox allocating overload builds its workspace before validating
- `SVD.LowRank.fProxy.cs:721-732`: allocates U/S/V from A's temp pool, then the ref-ws overload
  throws on bad m<n/k/Ak dims. The truncated allocating overloads (lines 553-560, 590-597,
  627-633) validate k/oversample BEFORE building the workspace, and RandomMatrixOP's class doc
  states validation-before-allocation as the house rule. Arena-pool temps are reclaimed en masse
  so nothing leaks — consistency nit only. Fix: hoist the k/shape checks above the ws builder.

### L-10. SelectOP.bool lacks the ref-dest primitives its fProxy/iProxy siblings have
- `SelectOP.bool.cs` offers only allocating `select` overloads; `SelectOP.fProxy.cs` /
  `SelectOP.iProxy.cs` each pair every allocating overload with a zero-alloc `ref dest` primitive.
  Possibly deliberate (bool masks are cheap) but there is no DEVLOG/policy note either way. Fix:
  add the two ref-dest overloads or record the scoping decision in DEVLOG.

### L-11. truncated's three allocating overloads duplicate a 14-field workspace builder verbatim
- `SVD.LowRank.fProxy.cs:561-581, 598-618, 634-654`: three byte-identical
  `fProxySVDTruncatedCache` initializers (only `p` differs, and the third's p provably equals the
  first's for its arguments — the code comment at line 655 even proves it). One private
  `BuildTruncatedTempWs(in A, int p)` would collapse them. Style-only.

---

## Areas confirmed clean (one line each)

- **QueryCore.Predicate fProxy/iProxy** — identical siblings, correct guards, no drift.
- **QueryCore.Metric fProxy/iProxy** (read as context for callers) — direction helpers correct; integer Euclidean/Cosine rejection correct.
- **QueryEnums / ResampleEnums** — singular markers present, docs accurate.
- **QueryOP.fProxy / iProxy** — all row/col loops verified against row-major M_Rows x N_Cols with M != N in mind; strided column loops correct; bounded-insertion k-selection logic verified in all 8 variants (nearest/farthest x rows/cols x masked); validation ladders consistent; integer overflow contracts thorough and honest (incl. iAbs MinValue saturation).
- **QueryOP.Predicate fProxy/iProxy** — empty-result sentinel contract consistent with QueryCore; bestIdx==-1 first-adopt guard correct; iProxy Groups B/C/D exclusion documented.
- **RandomOP.cs / RandomOP.fProxy** — Fisher-Yates bounds correct (NextInt max-exclusive), partial-FY sample correct; all 9 samplers' ICDFs verified incl. endpoint guards (log(0), tan at +-pi/2, triangular point-mass) with per-type `Consts.fProxyEpsilon`; Box-Muller spare fully-scaled and by-ref contract documented; weightedPick validation (NaN/Inf/negative/zero-total) complete.
- **RandomMatrixOP.fProxy** — MVN/Haar/SPD/conditioned/rank/stochastic all verified (Q-transpose captured before Lambda-scaling, Mezzadri sign fix, NaN-catching `!(cond >= 1)` guards, validation-before-Temp-alloc, symmetry enforcement); `NextFProxy` caps-token usage correct.
- **ResampleOP.fProxy** — Catmull-Rom stencils identical across the three copies; endpoint pinning correct incl. the pass-2 `scratch[srcM-1, c]` row; separable 2D pass order and scratch shape (srcM x dstN) correct; validation before the single Temp alloc.
- **SVD.fProxy** — faithful Golub-Reinsch structure; cancellation/shift/deflation index bookkeeping (l/nm/k) verified; Ut/Vt row-rotation pointer math all `(long)`-cast; workspace and allocating overload pairs semantically identical; values-only variant is a correct rotation-free reduction of the same recurrence; sweeps/converged counter plumbing consistent.
- **SVD.FullWorkspace** — sizes checked match the arena builder.
- **SVD.LowRank truncated** — omega-recurrence/ELR/DGKS logic checked against the stated lanbpro conventions; pDone/alpha/beta breakdown bookkeeping and the zero-padded inner bidiagonal are index-consistent (no negative-index path: kOut=0 whenever pDone=0); residual check uses the sorted inner U correctly; k=0/n=0 early-outs correct.
- **SolveStatus.cs** — enum docs and Burst-safe Name() switches complete and mutually consistent (all six DirectSolveStatus cases covered).
- **DEVLOG.md** — exists, already carries RandomOP/RandomMatrixOP/SVD.LowRank/SolveInfo history; no code comment in the partition duplicates an existing DEVLOG entry.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 4 (M-1 SimdMath codegen path, M-2 SolveInfo crefs, M-3 SelectOP NoAlias, M-4 singularValues swallowed status) |
| LOW      | 11    |
