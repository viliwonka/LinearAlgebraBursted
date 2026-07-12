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
- 2026-07-12 | Dropped agent-workflow narration ("the coder flagged" / "the coder's flagged behavioral
  divergence") from the file header and the BlockJacobi double-dispose comment. Kept the substance
  (the three seam differences from the dense types, and that the readonly-struct same-copy-throws
  divergence is deliberately pinned distinctly). (was SparseArenaWiringTests.fProxy.cs:13, 349-350)
- 2026-07-11 | Dropped the dangling `docs/dev/rfc-memory-model.md §4 Option A` citation from the file
  header. (was SparseArenaWiringTests.fProxy.cs:11)

## SparseBSRTests
- 2026-07-12 | Dropped the "Regression for the fixed use-after-free" / "Pre-fix ... Post-fix" narration
  from the GrowthThenDispose comment, the "used to double-free / use-after-free" framing from the
  ToDense/ToBSR transpose-reference comment, the "Pre-fix this double-freed ... post-fix ... idempotent"
  framing from GrowthClearThenDispose_NoDoubleDispose's comment, and the "Dangling-arena-pointer
  history" pointer in DenseTransMatVec's comment. Full account: fProxyBSR.ToDense and
  fProxyBSRBuilder.ToBSR used to take `in Arena arena`, but both call a mutating arena allocator method
  internally (arena.fProxyMat / arena.fProxyBSR); since those Arena methods aren't `readonly`, calling
  them through an `in Arena` parameter forced the C# compiler to make a defensive copy of the arena, and
  the allocated result's internal arena pointer captured the address of that dead stack temporary -- a
  use-after-scope bug. Reading elements off the result was fine (the Values buffer is a real,
  independent allocation), but any op that allocated through the result's own arena pointer (e.g.
  Blas.trans(dense).fProxyTempMat) dereferenced the dangling pointer and threw "allocator handle is not
  valid" under Burst, breaking the trans(ToDense(...)) validation recipe. Fixed by changing both
  signatures to `ref Arena arena` (matching how ArenaExtensions factory methods take `this ref Arena`).
  Separately, a builder grown past capacityHint=8 to 225 triplets (~5 UnsafeList reallocations) used to
  leave the arena's tracked copy of the builder pointing at a freed pre-growth buffer; arena.Clear()
  then arena.Dispose() (whose own trailing Clear() runs again) used to double-free that stale copy;
  both are now fixed via the builder's idempotent Dispose() (_state null-guard).
  (was SparseBSRTests.fProxy.cs:77, 383-390, 564-579, 600-606)
- 2026-07-11 | Replaced the raw Windows exit code in the growth/dispose regression comment
  (`-1073741819`, i.e. STATUS_ACCESS_VIOLATION) with "an access violation" -- a platform-specific crash
  code has no business in cross-platform library test comments. Context preserved: builder grown past
  capacityHint=8 to 225 triplets (~5 UnsafeList reallocations); pre-fix, arena.Dispose() double-freed
  the stale pre-growth buffer held by the arena's tracked copy. (was SparseBSRTests.fProxy.cs:387-389)

## MIPTests
- 2026-07-12 | Dropped "(third-review regression)" reviewer-workflow tag from the LargeMagnitudeIntegrality
  enum comment (kept the contract: MIP.solve must classify this fractional root via an absolute, not
  relative, integrality tolerance). Dropped the "Baselines were measured on both stages directly (by
  reverting to the stage-2 commit, running a throwaway diagnostic, then restoring)" debugging-methodology
  sentence from the STAGE 3 section banner. Dropped two "(The brief reported ...)" reviewer-brief asides
  (Stein15's header: the brief reported float stein15 finishing in ~261 nodes, which did not reproduce on
  the shipped code; Stage4NodesGomoryWolsey's comment: the brief reported 5 nodes for both dtypes, only
  double reproduced it). (was MIPTests.fProxy.cs:56-58, 96-100, 833, 946)
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
- 2026-07-12 | Dropped "ORCHESTRATOR DIAGNOSIS" / "ORCHESTRATOR VERIFICATION" / "adversarial
  mutation-testing review pass" agent-workflow narration and the OQ-D1/OQ-D2 ticket references from
  KahanSweep's header, GradualDecay's KNOWN, DISCLOSED LIMITATION comment, and TierEDistinctMagnitudes'
  header (kept the underlying engineering facts as plain contract prose). Full history: the KahanSweep
  n=32/64 exclusion was found via mutation testing (no test pinned "no spurious Kahan permutation" on
  float); a naive fix asserting P==identity for the whole n×theta grid failed at n=32/64, root-caused via
  an in-test exact-recompute oracle comparison to the classic Kahan/RRQR trailing-norm collapse (not a
  downdate defect) -- widening the pivot-tie tolerance 2x had zero effect, and adding an unrelated
  Debug.Log in the pivot-selection loop was enough to flip which type failed, confirming an
  edge-of-precision tie-break rather than a reproducible bug. GradualDecay's ORCHESTRATOR VERIFICATION
  (temporary instrumentation, since removed per OQ-D2: production must not carry a guard-fire counter)
  measured the real cumulative guard firing 1 time vs. a naive per-step guard's 0 times on a float,
  cond≈673, n=128 construction, confirming the mechanism activates on realistic ill-conditioned input.
  (was QRCPDowndateTests.fProxy.cs:102-122, 319-325, 542)
- 2026-07-11 | Dropped three dangling internal-doc citations: the file-header pointer to
  `docs/dev/spec-qrcp-downdate.md`; the "This was checked, not assumed -- see the review pipeline notes
  in docs/dev/spec-qrcp-downdate.md's implementation report" sentence in GradualDecay's KNOWN, DISCLOSED
  LIMITATION comment (kept the substance: the back-of-envelope bound itself, and that it was checked via
  that bound, not merely assumed); and the `(docs/dev/spec-qrcp-blocked.md OQ-B2)` citation in
  BlockedPanels (the surrounding sentence already states the actual fact -- GEMM accumulation vs the
  oracle's rank-1 chain is a different summation order -- so the citation was pure redundancy).
  (was QRCPDowndateTests.fProxy.cs:14, 341-342, 625)

## PCATests
- 2026-07-12 | Dropped agent-workflow narration from SvdTruncatedWideThrows's comment ("The coder
  confirmed..."). Full account: svdTruncated is NOT shape-free; pcaSVDTruncated adds the n>=p guard so
  it throws on wide data (p>n) just like pcaSVD/pcaRandomized. Deliberately no "truncated works on wide
  data" test. (was ML/PCATests.fProxy.cs:626)
- 2026-07-11 | Dropped the dangling `docs/dev/spec-pca.md "Tests"` citation introducing the
  acceptance-criterion list; the numbered list itself (#1-#7a etc.) is self-describing.
  (was ML/PCATests.fProxy.cs:18)

## QueryTests / QueryPredicateTests (fProxy / iProxy)
- 2026-07-12 | Dropped the "(review's HIGH finding)" reviewer-note reference from the MinValue-edge
  section banner in iProxy QueryTests. iAbs() on iProxy.MinValue saturates to iProxy.MaxValue (an
  off-by-one on the |MinValue| case, since MinValue has no positive counterpart in two's complement);
  this was originally raised as a HIGH-severity review finding and is now covered by MinValueEdge().
  (was iProxy/QueryTests.iProxy.cs:699)
- 2026-07-11 | Dropped dangling citations to `docs/dev/spec-query.md` (fProxy and iProxy QueryTests) and
  `docs/dev/spec-predicate-queries.md` (fProxy and iProxy QueryPredicateTests), plus the internal
  "(policies P2/P3)" and "(Section 4b + T1)" / "(Section 6 = T1..T5)" section labels. Where the spec
  citation was the entire line with no other content (fProxy QueryPredicateTests, fProxy QueryTests),
  the line was deleted outright rather than reworded.

## ControlTests / ControlLQRTests
- 2026-07-12 | Dropped agent-workflow narration from ControlTests' file header ("written by the coder
  agent alongside the implementation" / "is the test-writer agent's job"). Kept the substance: this file
  is basic smoke coverage (known tiny instance solves, statuses fire, throws throw), not the full battery
  (literature vectors, SDA-vs-oracle cross-check, property-based stability/PSD checks, warm-path
  perturbation convergence, redundant-actuator rank flagging, determinism). (was ControlTests.fProxy.cs:11)
- 2026-07-11 | Dropped dangling `docs/spec-lqr.md` citations (and the "per that spec's binding rules" /
  "'Tests' section items 1-7" internal labels) from both file headers. The surrounding prose already
  describes what each file covers (smoke tests vs. the full battery) without needing the spec.

## CHOTests
- 2026-07-12 | Dropped the "Solver API rework (commit 2)" and "Commit 2.5 (2a)" commit-ticket references
  from the SolveInPlaceExitIsUsableFactor / SolveInPlaceShortCircuitPurity enum-case comments. Kept the
  contracts: solveInPlace's exit factor is a valid decompSolve input, bit-identical to a fresh decomp +
  decompSolve on the same original A; and non-PD input leaves b_to_x untouched (short-circuit purity).
  (was CHOTests.fProxy.cs:41, 44)

## ArenaWiringTests (fProxy) — generational-overlay section
- 2026-07-12 | Dropped the "Stage E" internal stage label from the generational-overlay guard-tests
  section banner and its lead sentence. Kept the contract: a checks-gated "generational overlay" on the
  arena-tracked structs' Data getter throws InvalidOperationException on a stale handle instead of
  silently returning a dead/garbage buffer. (was ArenaWiringTests.fProxy.cs:370-371)

## GalleryTests / GalleryPhase2Tests
- 2026-07-11 | Dropped dangling `docs/dev/spec-gallery.md` citations from both file headers.
  GalleryPhase2Tests kept its other parenthetical (the production template file name
  Gallery.Phase2.fProxy.cs), which is a real in-repo reference, not an internal-only doc.

## KrylovFusedKernelTests / KrylovRound2Tests / KrylovVerifyAtExitTests / SSORTests / SparseSpMMTests / LargeSparseBenchmark
- 2026-07-12 | SSORTests: dropped the "Krylov Round-3" preamble from the file header too (a later pass
  judged the bare round label still read as an internal stage marker, superseding the 2026-07-11 call
  below to keep it there); replaced with a plain "SSOR preconditioner test coverage:" lead-in for the
  (a)-(e) list. (was SSORTests.fProxy.cs:11)
- 2026-07-12 | SparseSpMMTests: dropped the "ApplyBlock now streams ... instead of looping ... k scalar
  BSR.spMV calls" / "OLD per-row-Apply ApplyBlock it replaced" / "no need to check out pre-change
  history" change-history narration from the file header. Kept the contract: ApplyBlock streams the
  matrix once and applies to k row-vectors together (instead of k separate scalar BSR.spMV calls); SpMM
  output must equal k separate spMV rows, bit-identical; LOBPCG's trajectory must be bit-identical against
  an in-test scalar-loop reference operator. (was SparseSpMMTests.fProxy.cs:10-11, 20-23)
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

## LOBPCGSmokeTests
- 2026-07-12 | Dropped agent-workflow narration from the file header ("written by the coder agent
  purely to sanity-check the algorithm while iterating", quoted task-brief language, "left for the
  independent test-writer agent" x2, "the coder's OWN scratch smoke tests"). Kept the substance: this
  file is smoke coverage, not the comprehensive suite (analytic Laplacian oracle across k=1..4,
  BSR+block-Jacobi generalized preconditioned convergence comparison, rank-deficiency stress on the
  generalized path, breakdown/non-SPD-B behavior, and warm-start on the generalized cache are still
  open coverage gaps for a full suite). Also dropped "-- left for the independent test-writer agent"
  from GeneralizedBSRSmokeRunsAndConverges' comment (kept the substance: this only exercises SPD-A/B
  convergence, not the indefinite-A/buckling-shaped BSR case).
  (was LOBPCGSmokeTests.fProxy.cs:11-25, 460-464)

## InPlaceOpTests
- 2026-07-12 | Dropped the "PR #1" ticket reference and bug postmortem from the file header. Full
  account: addInPlace(place, from) is supposed to mutate `place` (place += from); the internal compAdd
  operands were reversed, so the method used to mutate the wrong operand instead — masked end-to-end
  only because the + operators also called it backwards. (was InPlaceOpTests.fProxy.cs:7-10)

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

## ScalarMatrixOpTests (iProxy)
- 2026-07-12 | Dropped the bug-postmortem file header ("the operator delegated to `rhs - lhs`, which
  negates the result since subtraction is not commutative"). Full account: `scalar - matrix` for integer
  matrices used to delegate to `matrix - scalar` (rhs - lhs) internally, which silently negated every
  result; the fix made the operator compute s - A[i,j] directly. The in-body comment already states the
  contract (5 - [[1,2],[3,4]] must be [[4,3],[2,1]]). (was iProxy/ScalarMatrixOpTests.iProxy.cs:9-10)
