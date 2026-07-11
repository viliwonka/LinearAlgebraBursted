# Doc/comment quality audit — Sparse, Arena, Analysis, Statistics, ML, Hash, Debug (2026-07-11)

**Summary**
- Files scanned: 62 (Sparse 13, Arena 18, Analysis 4, Statistics 7, ML 7, Hash 4, Debug 9). Every file read in full.
- Files with findings: 34. Clean: 28.
- Counts by category: WRONG/STALE 1, TOO-LONG 31, JARGON 4, HISTORY/DEV-SPEAK 36, NOISE 0. Total ~72 findings.
- Worst offenders: `Arena/Arena.cs` (10 findings; ~60-line struct doc), `Sparse/UnsafeOP.Sparse.fProxy.cs` (R2/R8 optimization-campaign banners), `Sparse/fProxyBSRBuilder.cs` (30-line bug-fix diary in a public struct doc), `Arena/ChunkedRecordTable.cs`, `ML/PCA.fProxy.cs` (~45-line class-doc essay).
- Systemic patterns (fix once, apply everywhere): (a) internal doc paths (`docs/dev/rfc-memory-model.md §4 Option A`, `docs/draft-spec-krylov-optimization.md`, `docs/dev/naming-style-guide.md`) cited in ~20 shipped comments; (b) ticket/round codes ("Krylov R2/R3/R5/R8", "Q4 ruling", "failure mode 1/FM2", "Milestone D", "5.4.x"); (c) A/B-benchmark narratives with percentages embedded in source; (d) the same dated "LINALG_DEBUG NaN-poison-on-dispose removed (2026-07-05)" sentence duplicated in two files; (e) multi-paragraph design-rationale essays on public struct/class docs.

All line numbers refer to the template source (the generated float/double/int copies inherit every finding).

---

## Sparse/Arena.Sparse.fProxy.cs
- `Sparse/Arena.Sparse.fProxy.cs:9-19` — HISTORY — "Pointer-stable allocation-record tables (docs/dev/rfc-memory-model.md §4 Option A)... DELIBERATELY... no divergence risk (RFC failure mode 1) left to fix..." — condense to one line: builders keep the value-copy model because State* is already pointer-stable; drop RFC path and failure-mode code.
- `Sparse/Arena.Sparse.fProxy.cs:102-126` — TOO-LONG — fProxyBSRTranspose doc: 3 paragraphs of spMVT-vs-spMV perf rationale plus concurrency-tripwire internals ("wrapping this method too would nest EnterMutation()...") — cut to: materializes Aᵀ (O(nnz)); returns A itself if Symmetric.
- `Sparse/Arena.Sparse.fProxy.cs:161-174` — HISTORY — fProxyBSRMirrorToFull doc cites "Krylov R3, Q4 ruling: v1 preconditioners are full-storage BSR only... MKL/Eigen practice matches" — keep contract (one-time O(nnzb·BR·BC) copy; no-op if already full), delete ruling narrative.

## Sparse/Debug.Sparse.fProxy.cs
- `Sparse/Debug.Sparse.fProxy.cs:33-36` — JARGON (minor) — "would emit into BOTH the float and double Print partials and collide (CS0102)" — fine to keep the reason, drop the compiler error code: "kept inline to avoid a duplicate const across the float/double partials".

## Sparse/fProxyBlockJacobi.cs
- `Sparse/fProxyBlockJacobi.cs:9-29` — TOO-LONG — struct doc is 4 paragraphs, ending in a defensive-copy essay about two *other* types ("undermining the zero-cost-dispatch claim...") — cut to: block-Jacobi preconditioner, built once (LU per diagonal block), Apply is zero-alloc.
- `Sparse/fProxyBlockJacobi.cs:185-195` — HISTORY — Apply doc: "mirroring bsrMatVecB{b}'s unroll -- Krylov R2, docs/draft-spec-krylov-optimization.md ... bit-identical to the general loop below" — keep the contract (z must not alias r); delete the dispatch/spec narrative.
- `Sparse/fProxyBlockJacobi.cs:238-249` — TOO-LONG — Dispose doc is a tradeoff essay ("This is a strictly-no-worse-than-before tradeoff: the pre-migration Dispose() had no double-dispose protection at all...") — keep one line: double-dispose throws (record-table guard); do not call twice.
- `Sparse/fProxyBlockJacobi.cs:252-258` — HISTORY — "LINALG_DEBUG NaN-poison-on-dispose removed (2026-07-05): the symbol was defined nowhere in the project, so that block was dead code..." — delete (comments about deleted code).

## Sparse/fProxyBSR.cs
- `Sparse/fProxyBSR.cs:20-31` — TOO-LONG/HISTORY — struct doc cites "spec-sparse-bsm.md §2.3" and adds a StructLayout paragraph pointing at "_gen's own doc comment for the padding-hole analysis" — trim; drop spec path.
- `Sparse/fProxyBSR.cs:43-52` — HISTORY — _gen comment walks exact padding-byte arithmetic and cites "ArenaLayoutTests.SparseStructsAreExpectedSize staying at 104" — replace with: generation stamp packed into existing struct padding (size unchanged); detects stale handles to recycled slots.
- `Sparse/fProxyBSR.cs:60-66` — HISTORY — "Replaces the old `Arena _arena` handle field: retiring it keeps this struct's size unchanged..." — delete the "replaces the old" narrative.
- `Sparse/fProxyBSR.cs:217-223` — HISTORY — same dated "LINALG_DEBUG NaN-poison-on-dispose removed (2026-07-05)" paragraph as fProxyBlockJacobi.cs — delete.

## Sparse/fProxyBSRBuilder.cs
- `Sparse/fProxyBSRBuilder.cs:19-41` — HISTORY (one of the worst in the audit) — "MUTABLE-STATE INDIRECTION (fixes a use-after-free)... reliably reproducible by adding more than capacityHint triplets... A NativeReference<State> wrapper would be an equivalent alternative; raw Malloc/Free was chosen..." — a ~25-line bug-fix diary with repro steps and rejected alternatives, inside a public struct doc. Replace with 2 sentences: triplet state lives behind a single heap-allocated State* so every struct copy (including the arena's tracked copy) sees list growth.
- `Sparse/fProxyBSRBuilder.cs:82-84` — HISTORY — "which fixes failure mode 1 (growable-list relocation), not FM2." — internal bug-taxonomy codes; delete.
- `Sparse/fProxyBSRBuilder.cs:306-314` — JARGON (minor) — comment leans on "CS1612" and "record-table migration" history — keep the useful part (locals share the same native buffer; cheaper than property dispatch in the loop), drop the migration narrative.
- `Sparse/fProxyBSRBuilder.cs:359-369` — TOO-LONG — Dispose doc over-explains idempotency across struct copies and the arena-owns-everything convention — one line: idempotent on the same copy; the owning arena disposes builders it created.

## Sparse/fProxyBSROperator.cs
- `Sparse/fProxyBSROperator.cs:75-80` — HISTORY (light) — ApplyBlock comment: "Krylov R5 (docs/draft-spec-krylov-optimization.md): a real BSR SpMM kernel... See BSR.spMM... for the per-kernel bit-identity argument" — reduce to: forwards to BSR.spMM (streams A once for all rows).

## Sparse/fProxySparseLP.fProxy.cs
- `Sparse/fProxySparseLP.fProxy.cs:70-77` — HISTORY — ApplyDot comment quotes the spec doc: "per Krylov R2's spec (docs/draft-spec-krylov-optimization.md): 'operators that delegate ... compose sensibly and document'" — keep "composes Apply + dot; no fused kernel here", delete the spec citation.

## Sparse/fProxySSOR.cs
- `Sparse/fProxySSOR.cs:16-34` — TOO-LONG/HISTORY — struct doc includes a "verified independently via 'one SSOR relaxation sweep from z=0'" process narrative, quotes spec wording ("Setup = block-Jacobi's setup (spec wording)"), and cites "Krylov R3, Q4 ruling" — keep the M formula, the three-step Apply factorization, the SPD condition, and the Saad/Young citations; delete the verification/ruling narrative.

## Sparse/SparseOP.fProxy.cs
- `Sparse/SparseOP.fProxy.cs:87-92` — HISTORY — spMM comment: "Krylov R5 (docs/draft-spec-krylov-optimization.md)... the old fProxyBSROperator.ApplyBlock looped `rows` scalar spMV calls through two Allocator.Temp vectors" — keep what it computes and the alias/zeroing contract; delete the old-implementation comparison.
- `Sparse/SparseOP.fProxy.cs:154-177` — HISTORY (top-5 finding) — spMVDot's comment is a full A/B benchmark writeup: "MEASURED, not assumed: an earlier version... went from ~0.245ms... to ~0.359ms... a reproducible ~45% REGRESSION... The fused kernels were deleted, not merely unused..." — replace with one line: composes spMV + Blas.dot (a fused kernel measured slower); the numbers belong in the benchmark doc.
- `Sparse/SparseOP.fProxy.cs:298` — HISTORY (light) — section banner "(Krylov R3, docs/draft-spec-krylov-optimization.md)" — drop the code/path, keep "block triangular sweeps".

## Sparse/UnsafeOP.Sparse.fProxy.cs
- `Sparse/UnsafeOP.Sparse.fProxy.cs:156-169` — HISTORY (worst single finding in Sparse) — banner literally titled "History (docs/draft-spec-krylov-optimization.md, R2/R8)": "R2 introduced a 2-accumulator even/odd pairing... R8 revisited it with a dedicated, repeated (3x) clean-room measurement... REVERTED... (R8 also spiked software prefetch... consistently SLOWER, 8-56%...)" — delete the whole History paragraph; keep only the bit-identical-accumulation-order contract above it (lines 149-154, which is legitimate).
- `Sparse/UnsafeOP.Sparse.fProxy.cs:619-641` — HISTORY — SpMM banner: "Krylov R5..." plus "the old ApplyBlock allocated two Temp vectors and re-walked rowPtr/colInd `rows` times" and an R8 cross-reference — keep layout/stride contract (ldV/ldAV) and bit-identity note in 3-4 lines; delete round codes and old-code comparison.
- `Sparse/UnsafeOP.Sparse.fProxy.cs:1109-1117` — HISTORY — duplicates SparseOP.fProxy.cs's ~45%-regression ApplyDot story a second time ("A/B'd... lost by a wide, reproducible margin (~45% SLOWER at N=5120/float)") — delete; one pointer at the dispatch site is more than enough.
- `Sparse/UnsafeOP.Sparse.fProxy.cs:1119-1129` — HISTORY (light) — "Krylov R2, fProxyBlockJacobi.Apply specialization (docs/draft-spec-krylov-optimization.md, R2)" banner — drop the code/path; keep the one-line kernel description.
- `Sparse/UnsafeOP.Sparse.fProxy.cs:1208-1234` — HISTORY (light) — sweep banner opens "Krylov R3 (docs/draft-spec-krylov-optimization.md, R3)" and cites "Q4 ruling" — the math description (sweepLower/sweepUpper semantics, diagScale) is good and should stay; strip the codes/paths.
- `Sparse/UnsafeOP.Sparse.fProxy.cs:142` — HISTORY (minor) — "Milestone D:" prefix on the specialization banner — delete the milestone label.

## Arena/Arena.cs
- `Arena/Arena.cs:16-46` — TOO-LONG/HISTORY — ArenaCore class doc: multi-paragraph essay citing "docs/dev/rfc-memory-model.md §4 Option A" and per-family migration status — cut to 2-3 sentences: what ArenaCore holds; allocation records live in pointer-stable tables.
- `Arena/Arena.cs:56-71` — TOO-LONG — Safety field doc explains the [NativeContainer]/AtomicSafetyHandle protocol conflict at essay length — one line: dispose-lifetime safety handle, checked at guarded entry points.
- `Arena/Arena.cs:76-109` — TOO-LONG/HISTORY — _busy field doc includes "Audited (2026-07-05) across every Arena.*.cs/Arena.Sparse.*.cs factory..." — drop the dated audit sentence; one paragraph max.
- `Arena/Arena.cs:148-159` — TOO-LONG/HISTORY — AllocationsCount doc: "PERMANENT (not transient) asymmetry for one deliberately-unmigrated family (docs/dev/rfc-memory-model.md §4 Option A)..." — one line for the BSR-builder caveat.
- `Arena/Arena.cs:305-312` — TOO-LONG — ClearCore ordering note ("dispose-then-Free, the OPPOSITE of fProxyN/fProxyMxN.Dispose()'s Free-then-dispose") — 1-2 lines on why this loop is safe.
- `Arena/Arena.cs:473-486` — TOO-LONG — Dispose() doc cross-references "the documented ... footgun on Arena's own class doc" — condense: frees every tracked allocation; double-dispose guarded.
- `Arena/Arena.cs:538-599` — TOO-LONG/HISTORY (top-5 finding) — the Arena struct's public class doc runs ~60 lines with four sub-essays ("Ownership contract", "Threading contract", two-tier model) citing "docs/dev/rfc-memory-model.md, failure mode 2" and "docs/features/dense-types.md" — reduce to a short paragraph: main-thread authoring allocator; copies share one core; exactly one owner calls Dispose; allocate before scheduling jobs, use in-place ops inside jobs.
- `Arena/Arena.cs:624-633` — HISTORY — ctor comment: "That used to be harmless... It stopped being harmless once ChunkedRecordTable<T> joined the field set" — replace with: MemClear so ChunkedRecordTable's IsCreated starts false.
- `Arena/Arena.cs:635-640` — TOO-LONG/HISTORY — try/finally comment cites "Burst's csharp-hpc-overview.md" — one sentence.
- `Arena/Arena.cs:657-665` — HISTORY — "replaces the old private `Arena _arena` field those used to read directly" — delete the historical clause.

## Arena/Arena.bool.cs, Arena.fProxy.cs, Arena.iProxy.cs
- `Arena/Arena.bool.cs:8-13`, `Arena/Arena.fProxy.cs:22-28`, `Arena/Arena.iProxy.cs:28-34` — HISTORY — identical boilerplate in all three: "Guarded (docs/features/dense-types.md's threading contract): ... see ArenaCore's _busy field doc (Arena.cs)" — shorten to: guarded under ENABLE_UNITY_COLLECTIONS_CHECKS; throws on a default/disposed arena.
- `Arena/Arena.fProxy.cs:85`, `Arena/Arena.iProxy.cs:91` — HISTORY — "persistent (backs Copy()); was wrongly the temp list" — delete "was wrongly the temp list".

## Arena/ArenaExtensions.iProxy.cs
- `Arena/ArenaExtensions.iProxy.cs:62-63` — HISTORY — "Previously passed (min, max) here, where min > max — ... returned garbage." — keep only the current contract (smaller bound first); delete the "previously" narration.
- `Arena/ArenaExtensions.iProxy.cs:174-175` — HISTORY — "(previously passed the inverted (min, max), yielding garbage)" — same fix.

## Arena/ChunkedRecordTable.cs
- `Arena/ChunkedRecordTable.cs:7-75` — TOO-LONG/HISTORY/JARGON — class doc: six paragraphs citing "rfc-memory-model.md §4 Option A / A1, §6.1, §7 step 2", "RFC's failure modes 1 and 2", and name-dropping the Rust "bumpalo 'chain of chunks' pattern" — collapse to: pointer-stable chunked slot table (records never move once allocated) + the doubling/free-list contract in 2-3 sentences.
- `Arena/ChunkedRecordTable.cs:190-204` — TOO-LONG — Free()'s exception doc over-explains the double-free aliasing scenario — one line: throws if the slot is already dead.
- `Arena/ChunkedRecordTable.cs:240-253` — TOO-LONG — IsAliveFast doc cross-references guards in four other types — reduce to: O(1) alive check via the Slot/Record layout invariant.
- `Arena/ChunkedRecordTable.cs:260-273` — TOO-LONG — Dispose() doc re-explains the caller's dispose-before-free obligation across two paragraphs — one line.
- `Arena/ChunkedRecordTable.cs:310-320` — TOO-LONG — comment above SlotPtr argues the "never per element" claim at length — trim to: reverse scan, called once per Allocate/Free/Resolve.
- `Arena/ChunkedRecordTable.cs:323-327` — HISTORY — "Without this, idx >= Count used to fall through the loop below silently... a heap-corruption footgun" — keep the current rationale for the unsigned-comparison guard, drop the used-to narrative.

## Arena/boolRecords.bool.cs, fProxyRecords.fProxy.cs, iProxyRecords.iProxy.cs
- `Arena/boolRecords.bool.cs:6-17` — TOO-LONG/HISTORY — class doc cites "rfc-memory-model.md §4 Option A, §7 step 4" — shrink to: arena-owned pointer-stable record backing a boolN; lives in a ChunkedRecordTable.
- `Arena/fProxyRecords.fProxy.cs:6-14` — TOO-LONG/HISTORY — same RFC-citing pattern — same fix.
- `Arena/iProxyRecords.iProxy.cs:11-19` — TOO-LONG/HISTORY — same RFC-citing pattern — same fix.
- (Also applies to `Sparse/fProxyBSRRecords.fProxy.cs:7-9`, which cites the same RFC path — otherwise clean.)

## Analysis/Analysis.iProxy.cs
- `Analysis/Analysis.iProxy.cs:6,10` — HISTORY — "see docs/dev/naming-style-guide.md" (twice) — drop the doc-path references.
- `Analysis/Analysis.iProxy.cs:5-18` — TOO-LONG — 13-line header proving "merge safety" and defending the no-epsilon design ("an epsilon-taking sibling would just mask real off-by-one bugs...") — one line: exact-equality integer predicates (integer arithmetic has no rounding error, so no epsilon overload).

## Analysis/Analysis.Metrics.fProxy.cs
- `Analysis/Analysis.Metrics.fProxy.cs:127-136` — TOO-LONG — logDeterminant doc motivates the slogdet form, digresses into Gaussian log-likelihood, and spells out the recovery formula — trim to: returns log|det(A)| with sign out-param; use over determinant to avoid overflow/underflow; A unchanged.

## Analysis/BoolAnalysis.cs
- `Analysis/BoolAnalysis.cs:59-67` — TOO-LONG — 9-line rationale for why any/all exist as sugar, plus a restated vacuous-truth table already covered by the per-method summaries — one line: any/all mirror math.any/math.all, including empty-input semantics.

## Statistics/Stats.iProxy.cs
- `Statistics/Stats.iProxy.cs:8` — HISTORY — "deliberately excluded, see docs/dev/naming-style-guide.md: ..." — state the rule inline or omit; drop the path.
- `Statistics/Stats.iProxy.cs:14` — HISTORY — "...docs/dev/naming-style-guide.md's \"Split vs merge safety\")." — same fix.

## Statistics/StatsCore.iProxy.cs
- `Statistics/StatsCore.iProxy.cs:16-17` — HISTORY — "collide as CS0111 (see docs/dev/naming-style-guide.md's \"Split vs merge safety\" and docs/dev/codegen-refactor-lessons.md)" — delete both citations; keep the one-sentence split reason.

## ML/KMeansEnums.cs
- `ML/KMeansEnums.cs:6` — WRONG (the only factual error found) — "KMeansPlusPlus = D²-weighted seeding ...; O(k²·N·D)" — the implementation (KMeans.fProxy.cs SeedKMeansPlusPlus) is the incremental version, O(k·N·D); its own comment says "Incremental D2Weights (O(k·N·D)) ... instead of recomputing from scratch (which was O(k²·N·D))". Fix the enum doc to O(k·N·D).

## ML/KMeans.fProxy.cs
- `ML/KMeans.fProxy.cs:142-242` — HISTORY — twelve inline spec-section numbers "5.4.1"–"5.4.10" ("5.4.1 Centroid squared norms", "5.4.10 Divide accumulators -> new centroids") with cross-references — strip the numeric prefixes, keep the plain step descriptions.
- `ML/KMeans.fProxy.cs:9-22` — TOO-LONG — class doc restates the GEMM assignment algebra (‖xₙ−cⱼ‖² expansion) across 4 paragraphs — 1-2 sentences (Lloyd k-means, GEMM-accelerated assignment, k-means++/uniform seeding); the derivation already exists at the GEMM step.

## ML/PCA.fProxy.cs
- `ML/PCA.fProxy.cs:6-50` — TOO-LONG (top-5 finding) — the PCA class doc is a ~45-line, 7-paragraph essay: four-route comparison, denominator-convention proof, degenerate-feature trap, sign convention, no-cache justification, determinism note — cut to ~4 lines (what PCA does; one line per route; details on each method's own doc).

## ML/PCA.Model.fProxy.cs
- `ML/PCA.Model.fProxy.cs:7-20` — TOO-LONG — struct doc defends the design choice ("This is a buffer-carrying (fProxy-prefixed) struct rather than a plain scalar diagnostics struct ... the same justification every Cache ... already has") — keep what it is / allocation / reuse; drop the why-a-struct paragraph.

## ML/PCA.Shared.cs
- `ML/PCA.Shared.cs:6` — JARGON — "Type-agnostic guard hoisted out of the per-type PCA template" — "factored out of", not "hoisted".

## Hash/Hash.fProxy.cs
- `Hash/Hash.fProxy.cs:49` — HISTORY — "See docs/dev/naming-style-guide.md's alsoExpand note." — delete or replace with a one-line inline explanation.
- `Hash/Hash.fProxy.cs:6-30` — TOO-LONG — class doc's float-hashing caveat mixes real contract (−0.0/NaN hash by bit pattern — keep) with a justification paragraph ("usually the RIGHT behavior... real footgun if...") — keep the two-bullet caveat, cut the argument.

## Hash/Hash.Shared.cs
- `Hash/Hash.Shared.cs:5-35` — TOO-LONG/HISTORY — three stacked header essays: split/merge rationale citing docs/dev/naming-style-guide.md, a "NOTE FOR EDITORS" warning, and an "ALGORITHM CHOICE" dev-diary ("'cleanest' in the sense of... 'fastest' in the sense of...") — compress to: xxHash32 (Yann Collet, public domain); non-cryptographic. Trim the editor/codegen warning to 2-3 lines if load-bearing.

## Debug/Export.bool.cs
- `Debug/Export.bool.cs:6-21` — TOO-LONG — 16-line header mixing real contract (ToText/ToCsv format, no truncation, managed-only) with a codegen-hazard essay ("would make TemplateConverter.Execute treat it as a multiplying file... collide on the SAME output path") — keep 2-3 contract lines; move the hazard note to a codegen dev doc.

## Debug/Export.fProxy.cs
- `Debug/Export.fProxy.cs:7-17` — TOO-LONG — three paragraphs (proxy-cast rationale, G9/G17/G7 format justification, truncation contract) restating what the sibling Export headers already say — condense to a short contract note.

## Debug/Export.iProxy.cs
- `Debug/Export.iProxy.cs:7-23` — TOO-LONG — longest of the three near-duplicate Export headers; walks the choose-marker mechanism, references GenUtils.cs/proxyStructs.cs, ends with a meta-instruction to itself ("Do NOT write the literal choose-marker token in this comment...") — condense like its siblings; codegen internals belong in one shared doc.

---

**Clean files (no findings):** Sparse: Export.Sparse.fProxy.cs, Gallery.Sparse.fProxy.cs (fProxyBSRRecords.fProxy.cs near-clean, one RFC citation noted above). Arena: ArenaConversions.fProxy.cs, ArenaExtensions.cs, ArenaExtensions.FFT.fProxy.cs, ArenaExtensions.fProxy.cs, ArenaExtensions.Generators.fProxy.cs, ArenaExtensions.Query.fProxy.cs, ArenaExtensions.Query.iProxy.cs, Gallery.SPD.fProxy.cs, Gallery.Special.fProxy.cs. Analysis: Analysis.fProxy.cs. Statistics: HistogramCore.fProxy.cs, HistogramOP.fProxy.cs, StatsCore.fProxy.cs, StatsOP.fProxy.cs, Structs.fProxy.cs. ML: KMeans.Workspace.fProxy.cs, KMeansEnums.cs is not clean (see above), PCAEnums.cs. Hash: Hash.bool.cs, Hash.iProxy.cs. Debug: Debug.cs, Debug.fProxy.cs, Debug.Histogram.fProxy.cs, Debug.Info.cs, Debug.iProxy.cs, Debug.PCAModel.fProxy.cs. (The Gallery files are a good model for the rest of the codebase: math vocabulary used precisely, contracts stated, no dev history.)
