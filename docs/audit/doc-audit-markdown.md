# Documentation audit — public v1.0 readiness

Audited 2026-07-11: `README.md`, `CHANGELOG.md`, `Assets/LinearAlgebra/Source/package.json`, and
all 53 files under `docs/` (19 in `docs/features/`, 19 in `docs/dev/`, 15 loose at the `docs/` root).

## Summary

1. **Biggest gap: the LP/LAD solver stack (9 template files — simplex, dual simplex, revised
   simplex, interior point, sparse LP, two exact LAD engines, LP-cache warm-solve) has zero public
   documentation.** Not in README's feature list, not in `docs/features/`, not in CHANGELOG. A
   shipped, large feature is publicly invisible.
2. `README.md`, `CHANGELOG.md`, and `package.json` are in good shape — clear, short, standard
   format, no commit hashes, no internal narrative — but the CHANGELOG hasn't been updated since
   the `0.1.0` tag despite substantial work landing after it (LP/LAD, LQR, MIP/QP drafts, etc.).
3. `docs/features/*.md` (the intended public feature docs) are mostly accurate and reasonably
   short, but three files leak clearly internal artifacts into public-facing text: an internal
   ticket code (`OQ-7`), a unit-test class name (`ArenaLayoutTests.Arena_IsPointerSized`), and
   another test class name (`ConvergenceBudgetTests`). Several files also link out to
   `docs/dev/spec-*.md` design-rationale docs, which is a soft violation of "no spec references
   in public docs."
4. `docs/dev/*.md` (19 files) are unambiguously internal — commit hashes, "user ruled", PowerShell
   gotchas, codegen internals — correctly siloed under `dev/` and fine to ship as repo-internal
   documentation.
5. 15 spec/draft/research files sit loose at the `docs/` root (not under `docs/dev/`), mixed in
   the same listing a user would see alongside `docs/features/`. Content-wise they're fine to keep
   (internal specs, several for unbuilt features), but their location undermines the public/internal
   split — they should move under `docs/dev/`.

## File classification

| File | Class | Reason |
|---|---|---|
| `README.md` | PUBLIC-READY | Install, quick start, benchmarks, feature list, license — solid. Missing LP/LAD + Realtime feature lines (see findings). |
| `CHANGELOG.md` | PUBLIC-READY (stale) | Good format/tone, but frozen at `0.1.0`; several shipped features postdate it with no entry. |
| `Assets/LinearAlgebra/Source/package.json` | PUBLIC-READY | Description reads well; matches README's feature list (same LP/LAD gap doesn't apply — description doesn't over-claim). |
| `docs/features/comp-elementwise.md` | PUBLIC-READY | Clean, minimal jargon. |
| `docs/features/decompositions.md` | PUBLIC-READY (needs trim) | Accurate but dense; leaks `OQ-7` ticket code and links to `docs/dev` spec files. |
| `docs/features/dense-types.md` | PUBLIC-READY (needs trim) | Correct, but the "Concurrency guards" section is an internal design rationale (mechanism names, a unit-test class name) that reads like an RFC excerpt, not a feature doc. |
| `docs/features/eigen.md` | PUBLIC-READY | Fine; domain-standard numerical terms only. |
| `docs/features/fft.md` | PUBLIC-READY | Clean. |
| `docs/features/generators.md` | PUBLIC-READY | Clean, short. |
| `docs/features/hash.md` | PUBLIC-READY | Clean, short. |
| `docs/features/la-primitives.md` | PUBLIC-READY | Clean. |
| `docs/features/least-squares.md` | PUBLIC-READY | Fine; "CGNR" mentioned once but explained in context. |
| `docs/features/ml.md` | PUBLIC-READY (typos) | Two grammar slips (see findings); otherwise fine. |
| `docs/features/print-export.md` | PUBLIC-READY | Fine; one "as of this writing" forward-looking aside to drop. |
| `docs/features/query.md` | PUBLIC-READY | Fine; links to two `docs/dev` spec files. |
| `docs/features/random.md` | PUBLIC-READY | Clean. |
| `docs/features/realtime.md` | PUBLIC-READY | Content is fine and honest about scope, but the feature isn't linked from README's feature list. |
| `docs/features/select-bits.md` | PUBLIC-READY | Clean, short. |
| `docs/features/solvers.md` | PUBLIC-READY | Fine; links to two `docs/dev` spec files. |
| `docs/features/sparse-bsr.md` | PUBLIC-READY | Fine; links to one `docs/dev` spec file. |
| `docs/features/stats.md` | PUBLIC-READY | Fine; links to one `docs/dev` spec file. |
| `docs/features/svd.md` | PUBLIC-READY (needs trim) | Accurate, but names an internal test class (`ConvergenceBudgetTests`). |
| `docs/dev/codegen-refactor-lessons.md` | INTERNAL-OK | Explicitly a lessons-learned log; PowerShell/codegen war stories, correctly dev-only. |
| `docs/dev/level3-blocking-guide.md` | INTERNAL-OK | Self-labeled "historical document"; kernel-blocking how-to for contributors. |
| `docs/dev/naming-style-guide.md` | INTERNAL-OK | Explicitly "for a linter/reviewer agent"; correct to keep internal. |
| `docs/dev/perf-vectorization-lessons.md` | INTERNAL-OK | Burst vectorization war stories; dev-only by design. |
| `docs/dev/rfc-memory-model.md` | INTERNAL-OK | Architecture RFC with a decision log; correctly internal. |
| `docs/dev/spec-debug-print.md` | INTERNAL-OK | Coder-facing implementation spec, historical. |
| `docs/dev/spec-gallery.md` | INTERNAL-OK | Coder-facing spec with build-order/test plan. |
| `docs/dev/spec-generators.md` | INTERNAL-OK | Coder-facing spec. |
| `docs/dev/spec-histogram-resample.md` | INTERNAL-OK | Coder-facing spec incl. research rationale section. |
| `docs/dev/spec-interop.md` | INTERNAL-OK | Draft spec for an unbuilt feature; fine to keep as planning doc. |
| `docs/dev/spec-kmeans.md` | INTERNAL-OK | Very detailed coder spec (line-number references into source) — clearly dev-only. |
| `docs/dev/spec-pca.md` | INTERNAL-OK | Coder spec with fable-review notes; dev-only. |
| `docs/dev/spec-predicate-queries.md` | INTERNAL-OK | Coder spec, exact-text interface listings. |
| `docs/dev/spec-qrcp-blocked.md` | INTERNAL-OK | Coder spec referencing commit hashes (`e865f27`). |
| `docs/dev/spec-qrcp-downdate.md` | INTERNAL-OK | Coder spec, numerics-heavy, adversarial test plan. |
| `docs/dev/spec-query.md` | INTERNAL-OK | Design-rationale doc, explicitly says "code is the source of truth." |
| `docs/dev/spec-solver-api-rework.md` | INTERNAL-OK | Naming-rework spec with open-questions log and commit plan. |
| `docs/dev/spec-sparse-bsm.md` | INTERNAL-OK | Draft design doc; note terminology (`BSM`) is superseded by shipped `BSR` naming — fine for an internal historical doc, just don't let it get mistaken for current API reference. |
| `docs/dev/spec-svd-eigen-convergence.md` | INTERNAL-OK | Coder spec with open questions and pipeline notes. |
| `docs/draft-spec-krylov-optimization.md` | INTERNAL-OK (misplaced) | Dev planning doc; belongs under `docs/dev/`. |
| `docs/draft-spec-mip.md` | INTERNAL-OK (misplaced) | Draft for an unbuilt feature; belongs under `docs/dev/`. |
| `docs/draft-spec-qp.md` | INTERNAL-OK (misplaced) | Draft for an unbuilt feature; belongs under `docs/dev/`. |
| `docs/draft-spec-sell-c-sigma.md` | INTERNAL-OK (misplaced) | Explicitly post-v1.0 research draft; belongs under `docs/dev/`. |
| `docs/draft-spec-sparse-dual-simplex.md` | INTERNAL-OK (misplaced) | Draft for an unbuilt feature; belongs under `docs/dev/`. |
| `docs/research-lp-preconditioners.md` | INTERNAL-OK (misplaced) | Research notes; belongs under `docs/dev/`. |
| `docs/research-lp-qp-solver-landscape.md` | INTERNAL-OK (misplaced) | Research notes, license survey; belongs under `docs/dev/`. |
| `docs/spec-facades-ls-lad.md` | INTERNAL-OK (misplaced) | Draft for an unbuilt LS/LAD facade; belongs under `docs/dev/`. |
| `docs/spec-lad-barrodale-roberts.md` | INTERNAL-OK (misplaced) | Spec for a **shipped** feature with no public doc counterpart (see finding #1). |
| `docs/spec-lad-frisch-newton.md` | INTERNAL-OK (misplaced) | Spec for a **shipped** feature with no public doc counterpart. |
| `docs/spec-lpbasis-persistence.md` | INTERNAL-OK (misplaced) | Perf spec with measured numbers; belongs under `docs/dev/`. |
| `docs/spec-lqr.md` | INTERNAL-OK (misplaced) | Spec for a **shipped** feature (per memory: LQR shipped) with no public doc counterpart. |
| `docs/spec-revised-simplex.md` | INTERNAL-OK (misplaced) | Spec for a **shipped** feature with no public doc counterpart. |
| `docs/spec-shipped-feature.md` | INTERNAL-OK (misplaced) | Internal process/definition-of-done doc; belongs under `docs/dev/`. |
| `docs/spec-sparse-lp.md` | INTERNAL-OK (misplaced) | Spec for a **shipped** feature with no public doc counterpart. |

**Counts:** 56 files audited · 22 PUBLIC-READY (3 with typos/trim notes) · 34 INTERNAL-OK
(19 correctly placed under `docs/dev/`, 15 misplaced at the `docs/` root) · 0 files needed as
INTERNAL-SHOULD-NOT-SHIP-AS-IS (nothing egregious enough to pull — the loose root files are a
location problem, not a content problem).

## Detailed findings

### 1. Shipped LP/LAD feature has no public documentation (highest priority)
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/` has nine `LP.*.fProxy.cs` template files
(`LP.fProxy.cs`, `LP.DualSimplex`, `LP.RevisedSimplex`, `LP.InteriorPoint`, `LP.FrischNewton`,
`LP.BarrodaleRoberts`, `LP.Sparse`, `LP.Cache`, `LP.Info.cs`) — a full LP/LAD solver stack
(simplex, dual/revised simplex, Mehrotra interior point, two exact L1 engines, sparse matrix-free
variant, warm-solve caching). None of this appears in `README.md`'s feature list, in
`docs/features/`, or in `CHANGELOG.md`. A user reading the public docs would not know the library
can solve linear programs or do L1/LAD regression at all.
**Fix:** add a `docs/features/lp-lad.md` (short, plain: what `LP.solve`/`LP.lad` do, when to use
which method) and a README feature bullet + CHANGELOG entry before v1.0.

### 2. `docs/features/decompositions.md:29-43` — internal ticket code + spec links in a public doc
Quote: `QRCP shares no cache (OQ-7): its pivot kernel recomputes column norms...` and `see
[naming-style-guide]... and [spec-solver-api-rework]... for the rationale`.
`OQ-7` is an internal open-question tracking code (confirmed still open — no `fProxyQRCache`
exists in the source yet). Ticket-style codes and direct links to `docs/dev` spec files are
exactly what the owner's public-doc standard rules out.
**Fix:** drop the `(OQ-7)` parenthetical; keep the spec cross-links only if the owner wants an
"advanced/internals" pointer — otherwise cut them from the public doc.

### 3. `docs/features/dense-types.md:42-80` — internal implementation narrative in a public doc
The "Concurrency guards (detection, not prevention)" section documents `Interlocked.CompareExchange`,
`AtomicSafetyHandle` lifecycle, and explicitly names a unit test:
`ArenaLayoutTests.Arena_IsPointerSized` (confirmed to exist at
`Assets/LinearAlgebra/SourceTests/ArenaLayoutTests.cs`). It also argues why Unity's
`[NativeContainer]` protocol wasn't used — a design-rationale discussion, not a usage doc.
**Fix:** keep a two-line practical summary ("don't share one Arena across concurrent jobs;
violations throw under collections-checks"); move the mechanism-level rationale into
`docs/dev/rfc-memory-model.md` (which already covers this ground).

### 4. `docs/features/svd.md:70-77` — internal test class name in a public doc
Quote: `the convergence battery (ConvergenceBudgetTests) asserts ≤¼ of the budget is used...`.
`ConvergenceBudgetTests` is a real test-suite class name, not something a library user can act on.
**Fix:** rephrase to "verified by an internal convergence-budget test suite" without the class name.

### 5. `docs/features/ml.md` — grammar/typos
- Line 30: `loses accuracy covariance matrix construction` — missing words, reads as a fragment.
  Should be something like "loses accuracy building the covariance matrix."
- Line 37: `Transform new data info fitted model:` — "info" should be "into".
**Fix:** two small copyedits.

### 6. CHANGELOG is frozen at `0.1.0` while the library has moved on
`CHANGELOG.md`'s only entry is `[0.1.0] — 2026-07-03`. Since then (per repo memory/commit log)
LP/LAD, LQR, MIP/QP research, and other work have landed. Nothing wrong with the existing entry's
tone/format (it's a good template — no commit hashes, no dev narrative), but it will read as
stale/misleading at the actual v1.0 cut unless updated.
**Fix:** add an `[Unreleased]` section now, or fold new features into a `0.2.0`/`1.0.0` entry
before release, using the same terse style as the existing entry.

### 7. README feature list is missing two shipped features
`README.md`'s "Features" list (lines 99-118) has 17 bullets but omits **LP/LAD** (see #1) and
**Realtime / RollingWindow** (`docs/features/realtime.md` exists and describes a real, if small,
shipped type — `LinearAlgebra.Realtime.floatRollingWindow`). Both are linkable today with no new
writing beyond a one-line bullet.
**Fix:** add two bullets to the Features list, mirroring the existing style.

### 8. Loose spec/draft/research files clutter the `docs/` root
15 files (`draft-spec-*.md`, `research-*.md`, `spec-*.md`) sit directly under `docs/`, at the same
level as the public `docs/features/` folder, rather than under `docs/dev/` where 19 similar files
already live. Content-wise all 15 are clearly internal (open questions for the user, commit
references, "verified 2026-07-XX" annotations) — nothing here needs rewriting — but a user
browsing the `docs/` folder on GitHub sees an internal planning file (e.g.
`draft-spec-krylov-optimization.md`) with the same visual weight as `docs/features/svd.md`.
**Fix:** `git mv` all 15 into `docs/dev/` (pure organization, no content change) to make the
public/internal split match the folder structure the owner already established for the other 19
dev docs.

### 9. Minor: forward-looking/uncertain phrasing in a couple of feature docs
- `docs/features/print-export.md:16`: `LOBPCGInfo doesn't have one yet as of this writing but is
  expected to follow the same convention.` — "as of this writing" is a documentation-freshness
  smell (it will silently go stale). Either implement it or state the gap without the
  time-stamped hedge.
- `docs/features/realtime.md:16-18`: "still unsettled design, not implemented" for related
  features — fine as an honest scope note, no action needed, flagged only for awareness since it
  borders on roadmap talk inside a public doc.

### 10. `package.json` / README — no issues of substance
`package.json`'s `description` field matches what's actually shipped (does not claim LP/LAD or
anything else undocumented), reads cleanly, and the `keywords` array is reasonable for package
discovery. README's install instructions, quick-start snippet, and benchmark table were spot
checked against `docs/features/decompositions.md`/`solvers.md`/`svd.md` and are consistent (same
method names, same benchmark numbers). No broken links found in README.
