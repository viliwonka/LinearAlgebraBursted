# W1 — Comments & XML documentation

Scope: `Assets/LinearAlgebra/CodeGen/TemplateSource/**`, `TemplateSourceTests/**`, `TemplateSourceBenchmarks/**` (templates only, read-only). Policy: CLAUDE.md "Comment policy (strict)" — code comments/XML docs state contracts only; history, dev-speak, benchmark verdicts, ticket refs, and reviewer/agent notes belong in the folder's `DEVLOG.md` only.

## Tools/check-doc-leaks.ps1 output

```
check-doc-leaks: clean (no internal artifacts in shipped surfaces).
```

**This "clean" result does not mean the templates are clean** — the script's `$targets` are `Assets/LinearAlgebra/Source`, `docs/features`, `README.md`, `CHANGELOG.md` only. It never reads `TemplateSource*/**` directly, and it never reads `SourceTests/Generated` or `Benchmarks/Generated` at all. See Finding 15: several leaks below are template-only (not yet regenerated into Source) and others live in test-template output the script doesn't watch regardless of regeneration state.

## Findings

**1. MEDIUM — dev-history baked into 3 public XML docs (`Eigen.Info.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.Info.cs:61, 129, 182`
`EigenSolveInfo.op_Implicit`, `EigenInfo.op_Implicit`, and `LanczosInfo.op_Implicit` each carry the identical historical aside: "Same as `Solved`, so `if (lanczos(...))` keeps compiling after **the return type changed from bool to this struct**." This is API-migration history, not a contract. Fix: state only the current contract ("Implicit bool conversion so `if (...)` compiles against this struct"); move history to `OP/DEVLOG.md`: `## Eigen.Info.cs` / `- 2026-07-13 | EigenSolveInfo/EigenInfo/LanczosInfo all used to return a plain bool; kept an implicit bool operator when they became structs so old call sites still compile. (was Eigen.Info.cs:61,129,182)`

**2. MEDIUM — benchmark narrative in a public constructor doc (`Kalman.UKFCache.fProxy.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.UKFCache.fProxy.cs:123-134`
The `fProxyUKFCache(int,Allocator)` ctor's `<summary>` justifies `alpha=1` with: "...measured (float32 prototype) to produce catastrophic cancellation in this library's precision range (a 1e-3 tracking error against an exact linear-KF oracle blew up to ~1 with alpha=1e-3, vs ~1e-6 with alpha=1)." This is a benchmark verdict inline in a shipped doc, and the template generates both float and double constructors from this text — but the cited evidence is explicitly float32-only, so the double constructor's doc generalizes float-only evidence. Fix: keep only the contract; move the numeric narrative to `OP/DEVLOG.md`; verify (or caveat) whether the claim actually holds for double.

**3. LOW/MEDIUM — stale implementation history in a public doc (`fProxyBSRBuilder.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSRBuilder.cs:177-179`
`ToBSR`'s doc: "Kept as `ref Arena` for API stability, but **this is no longer load-bearing**... Arena **is now** a thin copyable handle..." Fix: contract-only doc; move history to `Sparse/DEVLOG.md`.

**4. LOW — historical phrasing in a concurrency-guard comment (`Arena.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.cs:35-36`
"an Arena is single-threaded by contract, but **nothing previously enforced** that..." Fix: drop "nothing previously enforced", state only what the mechanism currently detects.

**5. LOW — benchmark-provenance pointer in a public doc (`LP.fProxy.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs:302-303`
"...see the comment on the dispatch expression below **for the benchmark it was set from**..." Fix: drop the provenance pointer from the XML doc; the plain code comment below already states the contract.

**6. MEDIUM — unsourced perf multiplier in 3 shipped `[Obsolete]` messages (`Eigen.fProxy.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.fProxy.cs:913, 1073, 1084`
`[System.Obsolete("Prefer Eigen.symmetricInPlace (..., ~30x faster) ...")]` — ships as a public compiler warning, repeated 3x, with no test enforcing the "~30x" claim. Fix: drop the specific multiplier from the shipped message; keep the number (if wanted) in `OP/DEVLOG.md`.

**7. MEDIUM (systemic, 7 occurrences) — ticket code "R6a" in `KrylovVerifyAtExitTests.fProxy.cs`**
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovVerifyAtExitTests.fProxy.cs:16, 62, 96, 169, 177, 223, 225`
"...proven directly against an in-test replica of the **PRE-R6a** cg loop..." / "the **R6a contract**." `R6a` is exactly `check-doc-leaks.ps1`'s own `Krylov R\d+[a-z]?` pattern — invisible here only because the script never scans this folder. Fix: replace with descriptive names; keep the round label (if wanted) in `TemplateSourceTests/DEVLOG.md` only.

**8. MEDIUM (systemic, 6 occurrences) — bug-postmortem + ticket tag "FM2" in `ArenaHandleTests.fProxy.cs`**
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ArenaHandleTests.fProxy.cs:7-14 (header), 32, 129, 134, 139`
Full postmortem: "Regression suite for a historical Arena dangling-pointer bug (**labeled FM2 in project history**)... Arena **used to** capture its identity by raw address... **Fixed by**..." Fix: one-line contract statement in the test file; move postmortem + "FM2" mapping to `TemplateSourceTests/DEVLOG.md`.

**9. MEDIUM (systemic, 8 occurrences) — "STAGE n" round labels as section banners**
`TemplateSourceTests/fProxy/MIPTests.fProxy.cs:12-13, 53, 70, 602, 795` (+3 more), `TemplateSourceTests/fProxy/SparseSolverTests.fProxy.cs:1532`
"// Grows by stage: (a)-(e) **STAGE 2**... (f) **STAGE 3**... (g) **STAGE 4**..." Same forbidden pattern class as R6a/FM2/OQ-n (`STAGE \d+ \(` is literally one of the guard script's 5 regexes). Fix: rename banners to describe the feature under test; keep stage history in DEVLOG only.

**10. MEDIUM (systemic, ~10+ occurrences) — bug postmortems inline instead of in DEVLOG**
Representative sample: `VectorCopyTests.fProxy.cs:7-9` ("Previously both routed to the temp pool, so Copy() returned a vector that ClearTemp would free out from under the caller"), `StatsTests.fProxy.cs:107-108` ("Previously 1/(M-1) = 1/0 = Inf..."), `BoolIndexingTests.cs:172-173` ("...previously dereferenced a null arena core..."), `SparseBSRTests.fProxy.cs:366, 545`, `LOBPCGSmokeTests.fProxy.cs:179`, `QueryTests.fProxy.cs:800` ("review's CRITICAL regression"). Fix: keep a one-line "regression: <current invariant>" comment per site; relocate narratives to `TemplateSourceTests/DEVLOG.md`.

**11. MEDIUM — direct audit citation + duplicated postmortem (`MPCTests.fProxy.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MPCTests.fProxy.cs:282-291`
"Regression test for **a post-ship audit finding (2026-07-12, OP/DEVLOG.md)**: the prestabilized input-bound rows used the WRONG Phi/Gamma block... **verified in the design's numpy prototype before this fix**..." Names the audit/date/DEVLOG directly and duplicates content that (per its own citation) already lives in `OP/DEVLOG.md`. Fix: trim to a one-line contract regression statement; drop citation/postmortem.

**12. LOW (4 occurrences) — uncited "per the spec" references**
`TemplateSourceTests/ML/PCATests.fProxy.cs:531`, `TemplateSourceTests/fProxy/LPTests.fProxy.cs:1526, 1542`, `TemplateSourceTests/fProxy/LOBPCGSmokeTests.fProxy.cs:284`. Matches the guard script's own "agent/workflow ref" pattern (`per the spec\b`). Fix: drop the qualifier; state the rule directly.

**13. LOW — reviewer-address language (`AccuracySweepTests.fProxy.cs`)**
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/AccuracySweepTests.fProxy.cs:32`
"...so **a reviewer can see** the input really was ill-cond." Fix: "...to confirm the input really is ill-conditioned."

**14. LOW (systemic, ~10 occurrences) — empirical/benchmark numbers justifying test tolerances**
`UKFTests.fProxy.cs:53-55`, `SVDRandomizedTests.fProxy.cs:226, 281, 322`, `ControlLQRTests.fProxy.cs:232`, `MIPTests.fProxy.cs:663, 687, 757, 846, 958` (measured node-count baselines, e.g. "Measured baselines (double): stage 2 = 267 nodes / obj 6..."). Fix: keep the resulting bound in the test; move measured figures to `TemplateSourceTests/DEVLOG.md`.

**15. MEDIUM — `check-doc-leaks.ps1` scope and pattern gaps**
`Tools/check-doc-leaks.ps1`
Scope gap: `$targets` never includes `TemplateSource*/**`, `SourceTests/Generated`, or `Benchmarks/Generated` — every leak in Findings 7-14 lives in `TemplateSourceTests`, invisible to this script regardless of regeneration state. Pattern gap: none of its 5 regexes match Findings 1-6's phrasing ("changed from bool to this struct", "no longer load-bearing", "nothing previously enforced", "measured (float32 prototype) to produce catastrophic cancellation") — equivalent phrasing already in (or later added to) regenerated Source would currently pass the guard too. Fix direction: add the Generated test/benchmark folders to `$targets`; broaden the "dev history" regex; consider a template-scanning mode of the same script.

## Areas confirmed clean
- TODO/FIXME/HACK/XXX: zero hits across all 398 template files.
- Commented-out code: none found.
- Trivial narration comments ("// increment i"): none found.
- Doc-vs-code spot check: `SVD.Solvers.fProxy.cs` `pinvSolve`'s "A is NOT modified" claim verified accurate against the implementation.
- Literal "float" surviving into double-generating docs: broad sweep found only correctly-scoped mentions (both variants named, or genuinely differing values given for both) — no literal wrong-word leak found, aside from Finding 2's evidence-generalization issue.
- Small utility files (`Indices.cs`, `Pivot.cs`): doc coverage proportionate; no missing-doc gaps found in spot check.

## Summary table

| Severity | Count |
|----------|-------|
| HIGH | 0 |
| MEDIUM | 10 |
| LOW | 5 |

(Findings 7, 8, 9, 10, 14 each represent multiple/systemic occurrences; total individual comment sites touched is ~45-50.)
