# DEVLOG — TemplateSourceTests
Code comments state contracts only; history lives here (see CLAUDE.md).

## ArenaHandleTests
- 2026-07-11 | Relocated the "THE OLD BUG (FM2) / THE FIX" postmortem from the file header. Full
  account: Arena used to be a plain struct holding all its mutable tracking state inline, and every
  math struct captured arena identity by RAW ADDRESS (`Arena* _arenaPtr`, set via
  `fixed (Arena* p = &arena) _arenaPtr = p;` in the `in Arena` constructors). Arena's allocator methods
  (e.g. fProxyVec / fProxyMat) are NOT `readonly`, so calling one through an `in Arena` PARAMETER forces
  the C# compiler to make a defensive copy of the arena first -- and the struct being constructed
  captured the address of that dead stack temporary. Once the enclosing helper's frame returned, that
  captured pointer dangled: indexing/Copy()/Dispose() on the returned struct dereferenced freed stack
  memory, surfacing under Burst as a native crash / "allocator handle is not valid". THE FIX: Arena was
  split into ArenaCore (heap-Malloc'd once, holds all mutable state, never copied) and Arena (a thin
  handle wrapping a single `ArenaCore*`). Every math struct now holds an Arena VALUE field. Copying an
  Arena handle -- including compiler-inserted defensive copies of `in Arena` params -- copies only the
  ArenaCore* value, so every copy still resolves to the same live core. The dangling-pointer failure
  mode is now structurally impossible. Originally cited docs/dev/rfc-memory-model.md §1 / §2.2 / §4
  Option A / §6.0 / §6.1 (internal-only doc, not shipped; reference dropped from source).
  (was ArenaHandleTests.fProxy.cs:7-31)

## ArenaWiringTests (fProxy / iProxy / bool)
- 2026-07-11 | Dropped the dangling `docs/dev/rfc-memory-model.md §4 Option A` citation from the file
  header (identical in all three dtype copies: fProxy/ArenaWiringTests.fProxy.cs,
  iProxy/ArenaWiringTests.iProxy.cs, bool/ArenaWiringTests.bool.cs). No other change: the "option
  (b)/(c)" generational-overlay language later in each file already defines both options inline
  (fProxyN/iProxyN/boolN = option (b), Alive-only; fProxyMxN/iProxyMxN/boolMxN = option (c), Alive + a
  `_gen` stamp) before it's used, so that part was left as-is (judged self-contained on re-read).

## SparseArenaWiringTests
- 2026-07-11 | Dropped the dangling `docs/dev/rfc-memory-model.md §4 Option A` citation from the file
  header. (was SparseArenaWiringTests.fProxy.cs:11)

## SparseBSRTests
- 2026-07-11 | Replaced the raw Windows exit code in the growth/dispose regression comment
  (`-1073741819`, i.e. STATUS_ACCESS_VIOLATION) with "an access violation" -- a platform-specific crash
  code has no business in cross-platform library test comments. Context preserved: builder grown past
  capacityHint=8 to 225 triplets (~5 UnsafeList reallocations); pre-fix, arena.Dispose() double-freed
  the stale pre-growth buffer held by the arena's tracked copy. (was SparseBSRTests.fProxy.cs:387-389)

## MIPTests
- 2026-07-11 | Trimmed the debugging-timeline narration on Stage4NodesBranchy12's node-count comment.
  Full history: node count is 199 (down from stage-3's 241, stage-4-without-cache's 218) once
  fProxyLPCache's persisted DSE weights are in play. An earlier coding-in-progress value of 216 traced
  back to a real bug: weight[] was resumed even when the entry eta-capacity check forced a fresh
  Refactorize, letting weight drift across an unbounded refactorization chain instead of being bounded
  like the eta chain; fixed in DualSimplexPivotCore via a `didResumeFactors` guard. Kept in-source: node
  count is 199, stable across reruns (not a per-launch nondeterminism artifact), and why (persisted DSE
  weights change branch-variable pricing at a warm-started non-logical basis). Also dropped the dangling
  `docs/draft-spec-mip.md` header citation and the redundant `docs/draft-spec-mip.md stage 4` pointer
  (both internal-only, not shipped). (was MIPTests.fProxy.cs:11, 73, 972-981)

## QRCPDowndateTests
- 2026-07-11 | Dropped three dangling internal-doc citations: the file-header pointer to
  `docs/dev/spec-qrcp-downdate.md`; the "This was checked, not assumed -- see the review pipeline notes
  in docs/dev/spec-qrcp-downdate.md's implementation report" sentence in GradualDecay's KNOWN, DISCLOSED
  LIMITATION comment (kept the substance: the back-of-envelope bound itself, and that it was checked via
  that bound, not merely assumed); and the `(docs/dev/spec-qrcp-blocked.md OQ-B2)` citation in
  BlockedPanels (the surrounding sentence already states the actual fact -- GEMM accumulation vs the
  oracle's rank-1 chain is a different summation order -- so the citation was pure redundancy).
  (was QRCPDowndateTests.fProxy.cs:14, 341-342, 625)

## PCATests
- 2026-07-11 | Dropped the dangling `docs/dev/spec-pca.md "Tests"` citation introducing the
  acceptance-criterion list; the numbered list itself (#1-#7a etc.) is self-describing.
  (was ML/PCATests.fProxy.cs:18)

## QueryTests / QueryPredicateTests (fProxy / iProxy)
- 2026-07-11 | Dropped dangling citations to `docs/dev/spec-query.md` (fProxy and iProxy QueryTests) and
  `docs/dev/spec-predicate-queries.md` (fProxy and iProxy QueryPredicateTests), plus the internal
  "(policies P2/P3)" and "(Section 4b + T1)" / "(Section 6 = T1..T5)" section labels. Where the spec
  citation was the entire line with no other content (fProxy QueryPredicateTests, fProxy QueryTests),
  the line was deleted outright rather than reworded.

## ControlTests / ControlLQRTests
- 2026-07-11 | Dropped dangling `docs/spec-lqr.md` citations (and the "per that spec's binding rules" /
  "'Tests' section items 1-7" internal labels) from both file headers. The surrounding prose already
  describes what each file covers (smoke tests vs. the full battery) without needing the spec.

## GalleryTests / GalleryPhase2Tests
- 2026-07-11 | Dropped dangling `docs/dev/spec-gallery.md` citations from both file headers.
  GalleryPhase2Tests kept its other parenthetical (the production template file name
  Gallery.Phase2.fProxy.cs), which is a real in-repo reference, not an internal-only doc.

## KrylovFusedKernelTests / KrylovRound2Tests / KrylovVerifyAtExitTests / SSORTests / SparseSpMMTests / LargeSparseBenchmark
- 2026-07-11 | Dropped all `docs/draft-spec-krylov-optimization.md` citations (R1/R2/R3/R3b/R5/R6a round
  labels) across these files' headers and inline comments. Kept the bare "Round-1" / "Round-2" /
  "Round-3" / "Round-3b" labels where they read as a self-contained section name; dropped the "R5"/"R3"
  short-form tags entirely where they added no information beyond what the surrounding sentence already
  said (SparseSpMMTests' "before" oracle comment, LargeSparseBenchmark's SSOR-preconditioner-for-LOBPCG
  comment). Also dropped a vague "(spec §3b/task brief)" aside in the same LargeSparseBenchmark comment.

## SparseSpMMTests
- (see combined Krylov entry above)

## UnsafeSortTests
- 2026-07-11 | Dropped the dangling `docs/spec-shipped-feature.md pillar 3` citation; kept the quoted
  testing policy itself ("New Blas/UnsafeOP kernels get DIRECT tests against a plain scalar reference
  implementation, not just indirect coverage through callers") inline since it's the actual content, not
  just a pointer. (was UnsafeSortTests.fProxy.cs:15)

## LPTests
- 2026-07-11 | Dropped twelve dangling internal-doc citations across this file: repeated
  `docs/spec-revised-simplex.md` "stage 1"/"stage 2" labels on the RevisedSimplex/DualSimplex section
  headers (both the TestType enum comments and the method-body section banners), `docs/draft-spec-mip.md
  stage 1` on the two LPBasis warm-start banners, `docs/spec-lpbasis-persistence.md acceptance 3` /
  `acceptance item 5` on the fProxyLPCache banners (x2) and the (3b) contract-violation comment,
  `docs/spec-lad-frisch-newton.md`'s "Tests section (4 items)" / "item 1" / "item 2" labels on the
  Frisch-Newton section (kept the actual tolerances: 1e-6 rel double / 1e-3 rel float L1-residual match,
  and the LadStackloss 5e-2 published-coefficient tolerance), and `docs/spec-lad-barrodale-roberts.md`'s
  "Tests section (6 items)" label on the Barrodale-Roberts section. All replaced with self-contained
  prose (no doc paths, no acceptance-item numbers) or removed outright where purely redundant with the
  surrounding sentence.
