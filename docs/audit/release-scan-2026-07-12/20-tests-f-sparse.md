# Release scan 2026-07-12 — area: tests-f-sparse

Scanned 14 template files (tests). Findings: total 5 — confirmed 5, uncertain 0, unverified 0, refuted 0; severity: high 0, medium 0, low 5.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SSORTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseArenaWiringTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseBSRTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseCompNormsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseEigenTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseGalleryTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseIC0Tests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseILU0Tests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseSolverTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseSpMMTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseStructuralTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseSymmetricTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseTransposeTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseUnrollTests.fProxy.cs

## Findings

### 1. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseILU0Tests.fProxy.cs:123 — Test named/documented as ILU(0) 'BeatsPlain' (fewer iterations) but the assertion only requires <=, so it passes even when the preconditioner gives NO reduction.

**Evidence**

```
Header: "pbiCGStab with ILU(0) converges ... and needs fewer iterations than unpreconditioned biCGStab". Assertion: `Assert.IsTrue(infoIlu.iterations <= infoPlain.iterations);`
```

`<=` admits an exact tie, so a run where ILU(0) helped nothing still passes, contradicting the 'fewer'/'beats' claim.

**Verifier**

Line 123 asserts `infoIlu.iterations <= infoPlain.iterations`, which passes on an exact tie (zero preconditioner benefit). This directly contradicts the header contract at lines 12-13 ("needs fewer iterations than unpreconditioned biCGStab") and the test method name `PbiCGStabConvergesAndBeatsPlain`. The sibling `SparseIC0Tests.fProxy.cs` at lines 129 and 155 uses the stricter `<= infoJ.iterations * 0.9` pattern the reviewer suggests, so the ILU(0) test is out-of-pattern with its own family. Suggested fix (tighten to `< infoPlain.iterations` or `* 0.9`) is correct; severity 'low' is appropriate since the test still catches gross regressions.

**Suggested fix**

Use a strict margin like the SSOR/IC0 sibling tests (`<= infoPlain.iterations * 0.9`, or at least strict `<`) so the test actually verifies the preconditioner reduces iteration count.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseArenaWiringTests.fProxy.cs:13 — Code comments reference an agent/workflow ('the coder flagged' / 'coder's flagged behavioral divergence'), violating the contracts-only comment policy.

**Evidence**

```
Line 13: "...whose seams differ from the dense types in three load-bearing ways the coder flagged"; line 349-350: "This is the coder's flagged behavioral divergence; pin it distinctly."
```

CLAUDE.md forbids 'notes to reviewers or references to agents/workflow' in code.

**Verifier**

Line 13 verbatim contains "the coder flagged" and lines 349-350 contain "This is the coder's flagged behavioral divergence". CLAUDE.md's strict comment policy explicitly lists "notes to reviewers or references to agents/workflow ('coder report', 'third-review finding')" as prohibited in code comments and mandates they be moved to the folder's DEVLOG.md. Both cited phrases are textbook instances of the forbidden pattern. Severity low as stated — the behavioral pinning is valid but the agent-role framing must be stripped and rationale relocated to DEVLOG.

**Suggested fix**

Strip the agent references; keep only the behavioral contract (readonly struct => Dispose cannot null _rec => same-copy re-dispose throws). Move rationale to the folder DEVLOG.md.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseBSRTests.fProxy.cs:564 — Bug-postmortem / development-history narration embedded in code comments (contracts-only policy violation).

**Evidence**

```
Lines 564-579: "fProxyBSR.ToDense and fProxyBSRBuilder.ToBSR used to take `in Arena arena` ... forced the C# compiler to make a defensive copy ... a use-after-scope bug ... Fixed by changing both signatures to `ref Arena arena`". Also line 77 "Dangling-arena-pointer history:" and the 'Pre-fix ... Post-fix' regression narration at 383-390/600-606.
```

**Verifier**

Lines 564-579 are a multi-paragraph bug postmortem narrating a fixed use-after-scope defect: prior `in Arena arena` signature, C# defensive-copy mechanism, dangling arena pointer, the Burst error message observed, and the fix to `ref Arena arena`. The same forbidden narration recurs at line 77 ('Dangling-arena-pointer history'), 383-390 ('Regression for the fixed use-after-free', 'Pre-fix', 'Post-fix'), and 600-606 ('Pre-fix this double-freed... post-fix... idempotent'). CLAUDE.md's comment policy explicitly forbids 'development history' and 'bug postmortems and debugging narration' in code comments and directs them to a per-folder DEVLOG.md — a genuine text-vs-policy contradiction, not a contract statement.

**Suggested fix**

Reduce to the standing contract (ToDense/ToBSR take `ref Arena`); relocate the postmortem to DEVLOG.md.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SSORTests.fProxy.cs:11 — Internal milestone/spec reference ('Krylov Round-3 new surfaces') in a code comment, disallowed by the contracts-only policy.

**Evidence**

```
Line 11: "// Krylov Round-3 new surfaces:"
```

An internal stage/spec reference of the kind CLAUDE.md routes to DEVLOG, not code comments.

**Verifier**

Line 11 of Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SSORTests.fProxy.cs reads "// Krylov Round-3 new surfaces:" — an explicit internal stage/milestone label. CLAUDE.md's strict comment policy routes "internal spec/ticket references" (e.g. STAGE n, R6a) to DEVLOG.md and reserves code comments for contracts only. The (a)-(e) block below could stay as a plain description of coverage, but the "Round-3" preamble is the exact disallowed pattern. Low severity, naming/style only — no behavior impact.

**Suggested fix**

Drop the 'Round-3' label; describe what the tests cover in plain terms, keep ticket/stage references in DEVLOG.md.

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseSpMMTests.fProxy.cs:10 — Change-history narration in file/comment ('ApplyBlock now streams ... instead of looping ... it replaced'), a contracts-only policy violation.

**Evidence**

```
Lines 10-11: "ApplyBlock now streams the matrix once and applies to k row-vectors together instead of looping k scalar BSR.spMV calls..."; lines 21-23 similarly narrate the kernel swap ('since SpMM is row-for-row bit-identical to the OLD per-row-Apply ApplyBlock it replaced').
```

**Verifier**

Lines 10-11 and 20-23 of Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SparseSpMMTests.fProxy.cs contain change-history narration ("ApplyBlock now streams ... instead of looping k scalar BSR.spMV calls", "the OLD per-row-Apply ApplyBlock it replaced", "no need to check out pre-change history") that directly contradicts CLAUDE.md's contracts-only comment policy, which lists 'an earlier version...', 'changed from...' as examples that must live in DEVLOG.md and never in code comments. Severity low is accurate — the current test invariants are correct, only the historical framing is out of place.

**Suggested fix**

State the current contract (SpMM output == k separate spMV rows, bit-identical) without the 'now ... instead of ... replaced' history; move the migration story to DEVLOG.md.

## Scanner notes

Scanned all 14 sparse/fProxy test templates in full. No high/medium correctness defects found: hand-computed expected values (BlockJacobi 1/11 inverse, block [[2,1],[0,2]] products, duplicate-summation, Strang line-fit x=(5,-3)/rnorm=sqrt(6)/xnorm=sqrt(34), Laplacian closed-form eigenvalues) all check out; tolerances are relative/scaled and precision-split via //+choose markers (looser float / tighter double); symmetric-storage dense references are genuinely symmetric; oracle cross-checks (spMM vs per-row spMV bit-exact, transpose vs spMVT, dense LU/CHO/QR references) are independent, not tautological; managed guard/throw tests use try/finally so no leak on the throwing path. Burst-abort paths in the value jobs skip arena.Dispose() only when an Assert already failed (established suite-wide pattern, not reported). The only non-comment issue is the ILU0 weak `<=` assertion. Remaining findings are contracts-only comment-policy violations (dev history / agent references) that are pervasive across these test headers; I reported the clearest representatives rather than every instance.
