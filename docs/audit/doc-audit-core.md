# Doc/Comment Quality Audit — TemplateSource core (root + fProxy/iProxy/bool/Interfaces/Indices/Pivot/Realtime)

Scope: `Assets/LinearAlgebra/CodeGen/TemplateSource/` root-level .cs files, plus `fProxy\`, `iProxy\`,
`bool\`, `Interfaces\`, `Indices\`, `Pivot\`, `Realtime\`. Comment/doc-comment quality only — no code
logic changes proposed.

**Summary**: 49 files scanned, 15 files have findings (34 clean). ~62 findings total: HISTORY ~34,
TOO-LONG ~19, WRONG/STALE 1, JARGON 0 flagged separately (folded into TOO-LONG essays), NOISE 1.
Worst offenders: `fProxyN.cs`/`fProxyMxN.cs` and their `iProxyN.cs`/`iProxyMxN.cs`/`boolN.cs`/`boolMxN.cs`
copies (same arena-plumbing essay+spec-citation block duplicated 4x across the type families), `Consts.cs`
(benchmark-campaign narration baked into public const doc comments), and `Interfaces\LinearOperator.fProxy.cs`
(stale `<c>Solvers</c>` reference + a fusion-history post-mortem inside a public XML doc).

## Top 5 worst findings
1. `Interfaces\LinearOperator.fProxy.cs:7` — WRONG/STALE — XML doc says "see `<c>Solvers</c>`" but the
   `Solvers` class was retired and split into `Krylov`/`Blas`/`SVD`/`Eigen` — dangling reference in a
   public interface doc comment.
2. `Consts.cs:45-60` — HISTORY/TOO-LONG — public const-field block comment narrates a benchmark campaign
   verdict-by-verdict ("Old shared gates... were actively regressing double... double QR at N=64 was ~40%
   slower blocked; Cholesky double at 256 ~15% slower").
3. `fProxyN.cs:168-190` / `fProxyMxN.cs:176-200` — HISTORY — `Dispose()` comment: "LINALG_DEBUG
   NaN-poison-on-dispose removed (2026-07-05): the symbol was defined nowhere in the project, so that
   block was dead code that had never executed" — dated dev-log entry inside shipped code, duplicated
   near-verbatim in `iProxyN.cs`/`iProxyMxN.cs`/`boolN.cs`/`boolMxN.cs`.
4. `Interfaces\LinearOperator.fProxy.cs:27-40` — HISTORY — `ApplyDot` XML doc spends 14 lines recounting
   that "an earlier version genuinely fused the reduction into the dense/BSR kernels, but that was
   measurably SLOWER" with a pointer to A/B numbers elsewhere — a perf post-mortem in a public interface doc.
5. `fProxyN.cs:37-53` (and its `iProxyN.cs`/`boolN.cs`/`*MxN.cs` clones) — TOO-LONG + HISTORY —
   `AssertRecordAlive`/`AssertRecordValid` comments run 12-16 lines of struct-size byte arithmetic and cite
   `docs/dev/rfc-memory-model.md §6.2` and specific test-class names inline.

---

## AssemblyInfo.cs

- `AssemblyInfo.cs:3-8` — HISTORY — "needed so concrete (NOT codegen'd) test files like ChunkedRecordTableTests.cs can exercise internal-only building blocks (e.g. LinearAlgebra.ChunkedRecordTable<TRecord>, docs/dev/rfc-memory-model.md §4/§6.1/§7 step 2) directly" — drop the spec-doc section citation and specific test-file name. Suggested: "Lets the hand-authored test assembly (BurstLinearAlgebra.Tests) see this assembly's internal types."

## Consts.cs

- `Consts.cs:32-41` — HISTORY/TOO-LONG — LQ block-gate comment: "MEASURED, cache-dependent crossover, not derived... Tuned on TallWideSolveBenchmark (A is k x 2k)... float wins from ~256 row-panels, double not until ~512." — benchmark-campaign narration on a public const. Suggested: "Row-count gate for LQ's blocked vs unblocked kernel. Cache-dependent; measured per dtype, pinned conservatively (err high)."
- `Consts.cs:45-60` — HISTORY/TOO-LONG — "Old shared gates (QR/QRCP 64, Cholesky/LU 256) were actively regressing double below its true crossover (double QR at N=64 was ~40% slower blocked; Cholesky double at 256 ~15% slower)." — full perf-campaign verdict narration, including "caught only because we benched" reviewer-style remark. Suggested: "Per-type level-3 blocking gates, measured per dtype from a blocked-vs-unblocked sweep; float/double ordering is not universal so each is measured independently, not derived."
- `Consts.cs:61-68` — HISTORY — inline trailing comments with raw benchmark numbers ("float wins from 128 (64 ~neutral)", "double loses <=256, wins from 512", etc. x8) — delete or move to a benchmark doc; the const values speak for themselves.
- `Consts.cs:70-81` — HISTORY/TOO-LONG — CHOP gate comment: "MEASURED on CholeskyBenchmark's face-off section... 1 warmup + 4 timed runs... N=512 showed a clear, non-overlapping win for both (float ~1.4%, double ~6.4%); N=1024 widened further (float ~5.4%, double ~10.4%)" — benchmark-run-log detail inline. Suggested: "Pivoted-Cholesky (CHOP) blocked-path gate; measured, same convention as the gates above."
- `Consts.cs:85-96` — HISTORY — "see docs/dev/spec-svd-eigen-convergence.md" spec-doc citation inside `sweepBudget`'s doc comment; otherwise this paragraph is a legitimate contract explanation (why it scales with n, floor of 75) and can stay if the citation is dropped.

## fProxy\fProxyN.cs

- `fProxyN.cs:11-16` — HISTORY — "(docs/dev/rfc-memory-model.md §4 Option A)... Replaces the old `Arena _arena` handle field: retiring it keeps this struct's size unchanged" — describes a removed field instead of the current one. Suggested: "Stable pointer into the arena's record table; null for a standalone (non-arena) vector."
- `fProxyN.cs:37-53` — TOO-LONG — `AssertRecordAlive` comment: 16-line essay on struct byte-size (32B = 8 _rec + 24 UnsafeList) and why `IsAliveFast` is used over the index-based lookup, citing `docs/dev/rfc-memory-model.md §6.2` and `ArenaLayoutTests.VectorStructsAreExpectedSize`. Suggested: "Debug-only liveness check (compiles out of player builds); throws on read-after-dispose. Uses the direct pointer-cast IsAliveFast to avoid a per-element chunk-scan cost."
- `fProxyN.cs:61-66` — HISTORY — "used by Copy()/TempCopy() and the cross-type allocation shortcuts... that used to read a private `_arena` field directly" — delete "used to" clause.
- `fProxyN.cs:149` — HISTORY — "// temp pool (was wrongly the persistent Copy path)" — delete parenthetical.
- `fProxyN.cs:168-190` — HISTORY/TOO-LONG — `Dispose()`: "LINALG_DEBUG NaN-poison-on-dispose removed (2026-07-05): the symbol was defined nowhere in the project, so that block was dead code that had never executed... instead of the old silent double-free through a stale value-copy in the arena's tracking list." — dated dev-log entry + description of a prior bug, 22 lines total. Suggested: "Frees the arena slot before disposing native memory, so an aliased double-dispose throws deterministically via the table's double-Free guard rather than double-freeing."

## fProxy\fProxyMxN.cs

- `fProxyMxN.cs:12-17` — TOO-LONG — StructLayout rationale paragraph justifying why `_gen` can rely on a padding hole. Suggested: "Sequential layout pins the trailing padding hole `_gen` (below) relies on."
- `fProxyMxN.cs:44-47` — HISTORY — "used to read a private `_arena` field directly" — delete.
- `fProxyMxN.cs:51-62` — TOO-LONG/HISTORY — `_gen` field comment: full byte-arithmetic derivation (44 bytes rounds to 48) citing `docs/dev/rfc-memory-model.md §6.2` and `ArenaLayoutTests.MatrixStructsAreExpectedSize`. Suggested: "Generation stamp (packed into existing struct padding) for detecting a stale handle into a since-recycled slot; 0/unused on the standalone path."
- `fProxyMxN.cs:136-152` — TOO-LONG — `AssertRecordValid` essay, same pattern as fProxyN's, plus "Alive is true again, but the generation moved on" restatement.
- `fProxyMxN.cs:176-200` — HISTORY/TOO-LONG — `Dispose()`: same "LINALG_DEBUG NaN-poison-on-dispose removed" dated dev-log entry as fProxyN.cs.

## fProxy\fProxyN.Shortcuts.cs / fProxyMxN.Shortcuts.cs

- `fProxyN.Shortcuts.cs:23` — TOO-LONG — multi-line rationale naming a specific downstream consumer file (`Krylov.fProxy.cs`) and a private field (`_rec`) instead of a short contract. Suggested: "Not in copyReplace: no iProxy BSR equivalent. Forwards to the arena's BSR transpose for solvers needing Aᵀ."
- `fProxyMxN.Shortcuts.cs:7` — TOO-LONG — same pattern naming `Hash.fProxy.cs`. Suggested: "`//alsoExpand[uint]` adds uint shortcuts (used by row/col hashing) alongside int/short/long."

(fProxyN.Comparators.cs, fProxyN.Indexing.cs, fProxyN.Operators.cs, fProxyMxN.Comparators.cs, fProxyMxN.Indexing.cs, fProxyMxN.Operators.cs are clean.)

## iProxy\iProxyN.cs

- `iProxyN.cs:15` — HISTORY — `docs/dev/rfc-memory-model.md §4 Option A` citation.
- `iProxyN.cs:17` — HISTORY — "Replaces the old `Arena _arena` handle field" — describes removed field.
- `iProxyN.cs:40-56` — TOO-LONG — same byte-size essay as fProxyN.cs.
- `iProxyN.cs:45-46` — HISTORY — spec-doc + test-name citation inline.
- `iProxyN.cs:65-68` — HISTORY — "used to read a private `_arena` field directly... exactly as the old `_arena` field required."
- `iProxyN.cs:152` — HISTORY — "(was wrongly the persistent Copy path)".
- `iProxyN.cs:175-187` — TOO-LONG — 13-line Dispose()/double-Free essay.
- `iProxyN.cs:182-183` — HISTORY — "instead of the old silent double-free through a stale value-copy in the arena's tracking list."

## iProxy\iProxyMxN.cs

- `iProxyMxN.cs:15-20` — TOO-LONG — StructLayout/padding-hole rationale paragraph.
- `iProxyMxN.cs:27-28` — HISTORY — "(same Option A record-pointer design...)" internal design-doc terminology.
- `iProxyMxN.cs:48-49` — HISTORY — "used to read a private `_arena` field directly".
- `iProxyMxN.cs:54-65` — TOO-LONG/HISTORY — byte-arithmetic essay citing `docs/dev/rfc-memory-model.md §6.2` and a specific test name.
- `iProxyMxN.cs:139-155` — TOO-LONG — `AssertRecordValid` essay mirroring iProxyN's.
- `iProxyMxN.cs:179-194` — TOO-LONG/HISTORY — `Dispose()` essay, same pattern as iProxyN.cs, including "opposite order... for a different reason" cross-narration.

(iProxyN.Comparators.cs, iProxyN.Indexing.cs, iProxyN.Operators.cs, iProxyN.Shortcuts.cs, iProxyMxN.Comparators.cs, iProxyMxN.Indexing.cs, iProxyMxN.Operators.cs, iProxyMxN.Shortcuts.cs are clean — short, legitimate cross-file pointers like "see UnsafeBoolOP.iProxy.cs's ispow2 kernel" are fine and not flagged.)

## bool\boolN.cs

- `boolN.cs:13` — HISTORY — `docs/dev/rfc-memory-model.md §4 Option A` citation.
- `boolN.cs:15` — HISTORY — "Replaces the old `Arena _arena` handle field".
- `boolN.cs:39-54` — TOO-LONG — 16-line `AssertRecordAlive` essay.
- `boolN.cs:43` — HISTORY — spec-doc + `ArenaLayoutTests.VectorStructsAreExpectedSize` citation.
- `boolN.cs:63` — HISTORY — "that used to read a private `_arena` field directly".
- `boolN.cs:78` — HISTORY — "guard a standalone (null-record) source — was dereferencing null for the default allocator" — describes the old bug, not the current guard.
- `boolN.cs:138` — HISTORY — "same ordering rationale as every other migrated family's N.Dispose() (e.g. floatN/intN)" — cross-family migration narration.

## bool\boolN.Operators.cs

- `boolN.Operators.cs:4` — NOISE — "can optimize scalar bool operations by not computing (like vec & false is always false)" — unimplemented TODO-style idea sitting in shipped code — delete.

## bool\boolMxN.cs

- `boolMxN.cs:24` — HISTORY — "see boolN.cs's `_rec` doc comment... (same Option A record-pointer design...)".
- `boolMxN.cs:45` — HISTORY — "that used to read a private `_arena` field directly".
- `boolMxN.cs:51-62` — TOO-LONG — 12-line generation-stamp byte-math essay.
- `boolMxN.cs:56` — HISTORY — `docs/dev/rfc-memory-model.md §6.2` + `ArenaLayoutTests.MatrixStructsAreExpectedSize` citation.
- `boolMxN.cs:104` — HISTORY — "was dereferencing null for the default allocator" — old-bug framing.
- `boolMxN.cs:134-149` — TOO-LONG — 16-line `AssertRecordValid` essay.
- `boolMxN.cs:177` — HISTORY — "same ordering rationale as boolN.Dispose() and every other migrated family's MxN.Dispose() (e.g. floatMxN/intMxN)" — migration narration.

(boolN.Indexing.cs, boolN.Shortcuts.cs, boolMxN.Indexing.cs, boolMxN.Operators.cs, boolMxN.Shortcuts.cs are clean.)

## Interfaces\LinearOperator.fProxy.cs

- `LinearOperator.fProxy.cs:7` — WRONG/STALE — `IfProxyLinearOperator`'s summary says "Lets Krylov solvers... see `<c>Solvers</c>`", but the `Solvers` class was retired (split into `Krylov`/`Blas`/`SVD`/`Eigen`); the same file correctly refers to `Krylov.cg(...)` elsewhere. Fix: `see <c>Krylov</c>`.
- `LinearOperator.fProxy.cs:27-40` — HISTORY — `ApplyDot`'s doc comment spends most of its 14 lines on a fusion-attempt post-mortem: "Krylov R2 (docs/draft-spec-krylov-optimization.md)... an earlier version genuinely fused the reduction into the dense/BSR kernels, but that was measurably SLOWER (see... for the A/B numbers and root cause)". Suggested: "y = Ax, and also returns dot(x,y) — a single call site for cg/pcg's `pAp = dot(p, Ap)`. Only meaningful when x and y are the same length (A square). Every implementation composes Apply then Blas.dot."
- `LinearOperator.fProxy.cs:95-97` — HISTORY — "a genuinely-fused version was tried and measured slower" restates the same fusion-history story a second time as an implementation comment — delete.
- `LinearOperator.fProxy.cs:168-185` — TOO-LONG — `fProxyColScaledOperator`'s doc comment is a 19-line, multi-paragraph tutorial on column-equilibration preconditioning theory plus a Tikhonov-damping aside. Suggested: keep only "Wraps `TInner` with a diagonal column scale D, presenting A·D. Apply forms A(d.*x) via an owned scratch buffer; solve for y, then recover x = D·y. `d`/`scratch` must not alias any vector passed to Apply/ApplyT."
- `LinearOperator.fProxy.cs:123-131` — TOO-LONG — `fProxyIdentityOperator`'s doc comment over-explains why a bit-copy operator reproduces the Euclidean-only formula exactly, with a cross-reference into LOBPCG's own doc. Suggested: "Identity operator (y = x). Lets B=I callers forward into the generalized `Eigen.lobpcg` core without duplicating the Euclidean-only path."

(Interfaces.cs, PredicateQuery.fProxy.cs, PredicateQuery.iProxy.cs, Sampler.fProxy.cs, ScalarFunction.fProxy.cs, Indices\Indices.cs, Pivot\Pivot.cs, Pivot\Pivot.Operations.cs, Realtime\RollingWindow.fProxy.cs are clean.)

---

## Clean files (no findings)

Assume.cs, Assume.fProxy.cs, Assume.iProxy.cs, ChooseMarkerDemo.fProxy.cs, ChooseMarkerDemo.iProxy.cs, markers.cs, proxyShims.cs, proxyStructs.cs, proxyStructs.math.cs, fProxyN.Comparators.cs, fProxyN.Indexing.cs, fProxyN.Operators.cs, fProxyMxN.Comparators.cs, fProxyMxN.Indexing.cs, fProxyMxN.Operators.cs, iProxyN.Comparators.cs, iProxyN.Indexing.cs, iProxyN.Operators.cs, iProxyN.Shortcuts.cs, iProxyMxN.Comparators.cs, iProxyMxN.Indexing.cs, iProxyMxN.Operators.cs, iProxyMxN.Shortcuts.cs, boolN.Indexing.cs, boolN.Shortcuts.cs, boolMxN.Indexing.cs, boolMxN.Operators.cs, boolMxN.Shortcuts.cs, Interfaces.cs, PredicateQuery.fProxy.cs, PredicateQuery.iProxy.cs, Sampler.fProxy.cs, ScalarFunction.fProxy.cs, Indices.cs, Pivot.cs, Pivot.Operations.cs, RollingWindow.fProxy.cs.
