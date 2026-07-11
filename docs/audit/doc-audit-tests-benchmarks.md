# Doc audit: TemplateSourceTests / TemplateSourceBenchmarks / Tools

Fast grep-driven pass ahead of v1.0. Scope: ~125 files in `TemplateSourceTests/`,
~24 files in `TemplateSourceBenchmarks/`, 7 scripts in `Tools/`.

## Summary

- No `TODO|HACK|FIXME|XXX` markers found anywhere in scope.
- No commented-out dead-code blocks found (clusters of `//`-prefixed statement lines).
- `Tools/*.ps1` top-of-file comments are clean, user-facing, and accurate — no findings.
- Two real categories of noise turned up: (1) ~45 inline references to **internal
  planning docs** (`docs/dev/*.md`, `docs/draft-spec-*.md`, `docs/spec-*.md`) that
  won't resolve for anyone who only has the public package; (2) a handful of
  **postmortem-style dev-history narration** in regression-test/benchmark comments
  that read like debugging-journal entries rather than test documentation.
- Everything else that matched the hunt patterns (e.g. the `#pragma warning disable
  618 // intentionally exercises the deprecated cyclic-Jacobi decompInPlace`
  comment repeated identically in 7 files, or the many "used to / no longer"
  regression-test explanations) is legitimate, self-contained test documentation
  and is NOT flagged — it explains a real regression being guarded, doesn't leak
  jargon that requires outside context, and doesn't contain stale claims.

## Findings

### Category A — dangling references to internal/dev-only spec docs

These comments cite `docs/dev/*.md`, `docs/draft-spec-*.md` files as the
authority for why a test exists. If `docs/dev` and the `draft-spec-*` files
aren't shipped in the public UPM package (worth confirming against the release
punch list), these become dead pointers for external readers. Representative
sample (not exhaustive — ~45 total hits across ~20 files):

- `TemplateSourceTests/fProxy/ArenaHandleTests.fProxy.cs:8` — category: internal spec ref — `"(docs/dev/rfc-memory-model.md §1 / §2.2 / §4 Option A / §6.0 / §6.1)"` — fix: drop the internal RFC path/section list, or move the historical rationale to a CHANGELOG-style doc and leave only a one-line "regression test for a dangling-arena-pointer bug" comment in source.
- `TemplateSourceTests/fProxy/ArenaWiringTests.fProxy.cs:9,15,481` (same header duplicated verbatim in `bool/ArenaWiringTests.bool.cs` and `iProxy/ArenaWiringTests.iProxy.cs`) — category: internal spec ref + jargon — `"the one case option (c) buys over option (b)"` — the comment argues for a design choice using labels ("option (b)", "option (c)") from `rfc-memory-model.md` that are meaningless without that file. Fix: either inline what (b)/(c) means in one clause, or cut to the observable behavior being tested.
- `TemplateSourceTests/fProxy/QRCPDowndateTests.fProxy.cs:341-342` — category: internal spec ref — `"This was checked, not assumed — see the review pipeline notes in docs/dev/spec-qrcp-downdate.md's implementation report."` — points a public reader at an internal review writeup that likely isn't shipped. Fix: either ship a trimmed rationale inline or drop the pointer.
- `TemplateSourceTests/fProxy/LPTests.fProxy.cs:66,1220,1803` and `TemplateSourceTests/fProxy/MIPTests.fProxy.cs:973` — category: internal spec ref — repeated `docs/spec-lpbasis-persistence.md acceptance item N` citations — fine internally, but "acceptance item 2/3/5" is an internal tracking label with no meaning to an outside reader of the shipped comment.
- `TemplateSourceBenchmarks/LargeSparseBenchmark.fProxy.cs:23,26,43,75,91-94,132,239,323` and `TemplateSourceBenchmarks/LPBenchmark.fProxy.cs:253,310` — category: internal spec ref — repeated `docs/draft-spec-krylov-optimization.md (R1/R2/R3/R3b/R5/R6a)` round-number citations used as the sole justification for benchmark sections. Fix: at minimum, drop "draft-" from the filename reference if the doc has since been finalized/renamed (worth a stale-path check), and consider whether the round numbers (R1..R6a) mean anything without the spec doc in hand.

### Category B — dev-history / postmortem narration (worth trimming for public release)

- `TemplateSourceTests/fProxy/ArenaHandleTests.fProxy.cs:7-31` — category: dev-history narration — 25-line "THE OLD BUG (FM2) ... THE FIX ..." postmortem, written like an incident report (raw-pointer mechanics, dangling-stack-frame walkthrough) rather than test documentation — fix: compress to ~3 lines ("regression test: Arena used to be capturable by raw address, so a compiler-inserted defensive copy of an `in Arena` param could dangle; guarded by ArenaCore indirection now").
- `TemplateSourceTests/fProxy/SparseBSRTests.fProxy.cs:387-389` — category: dev-history narration + platform-specific leak — `"Pre-fix, arena.Dispose() below double-freed the stale pre-growth buffer held by the arena's tracked copy (native crash, exit code -1073741819)."` — a raw Windows STATUS_ACCESS_VIOLATION exit code has no business in cross-platform library test comments — fix: drop the literal exit code, keep "(native crash)".
- `TemplateSourceTests/fProxy/MIPTests.fProxy.cs:972-981` — category: dev-history narration — narrates a debugging timeline for a magic node-count constant: `"an earlier coding-in-progress value of 216 traced back to a real bug: weight[] was resumed even when the entry eta-capacity check forced a fresh Refactorize... fixed in DualSimplexPivotCore via didResumeFactors"` — reads like a commit message pasted into source, and exposes an internal method/flag name (`didResumeFactors`) as if it's load-bearing documentation — fix: state only what the test asserts (node count is 199, stable across reruns) and drop the debugging narrative.
- `TemplateSourceBenchmarks/LPBenchmark.fProxy.cs:24-35` — category: dev-history narration — explains the reporting design by recounting a past incident: `"an extended benchmark run measured minutes and was killed because of it"` — fix: state the current contract ("each job self-reports objOut/itersOut/statusOut from inside Execute() to avoid a second Mono-interpreted solve") without the "was killed" anecdote.
- `TemplateSourceTests/fProxy/SparseSpMMTests.fProxy.cs:193-195` and `TemplateSourceBenchmarks/LargeSparseBenchmark.fProxy.cs:94` — category: internal round-number + dev narration — `"Krylov R5's 'before' oracle"` / `"R3 verified"` — low severity, same pattern as Category A, listed here because it doubles as narration of a since-superseded implementation kept only as an A/B oracle. Fine to keep the oracle; consider renaming away from the "R5" round label.

## Not flagged (checked, judged fine)

- The `#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi decompInPlace (kept for reference)` comment, identical across 7 files (EigenTests, EigenQRTests, GalleryTests, GalleryPhase2Tests, LiteratureTests, RandomMatrixTests, SolverBatteryTests) — accurate, self-contained, states its own rationale.
- `BoolIndexingTests.cs:171-174`, `LiteratureTests.fProxy.cs:110`, `SVDSolverTests.fProxy.cs:72`, `SVDWorkspaceTests.fProxy.cs:57`, `QRCPTests.fProxy.cs:685,964,989` — all "no longer modifies A" / "previously dereferenced" style notes are self-contained explanations of current contract or the regression being guarded; no external doc dependency.
- `Tools/*.ps1` headers — accurate, describe present-tense behavior only.
