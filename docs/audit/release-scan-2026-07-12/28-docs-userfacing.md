# Release scan 2026-07-12 — area: docs-userfacing (non-template)

Scanned 19 files (docs). Counts: total 8, confirmed 8, uncertain 0, unverified 0, refuted 0 — severity: high 1, medium 2, low 5.

## Scope

- README.md
- CHANGELOG.md
- docs/features/comp-elementwise.md
- docs/features/control.md
- docs/features/decompositions.md
- docs/features/dense-types.md
- docs/features/eigen.md
- docs/features/fft.md
- docs/features/generators.md
- docs/features/hash.md
- docs/features/la-primitives.md
- docs/features/least-squares.md
- docs/features/lp-lad.md
- docs/features/ml.md
- docs/features/print-export.md
- docs/features/qp-mip.md
- docs/features/query.md
- docs/features/random.md
- docs/features/realtime.md

## Findings

### 1. [high/naming/CONFIRMED] docs/features/control.md:10 — Warm-LQR overload and its constructor are documented with the codegen placeholder type `fProxyLQRState`, which is not a real symbol — the copy-paste example will not compile.

**Evidence**

```
Doc: "`Control.lqr(in A, in B, in Q, in R, ref K, ref fProxyLQRState state[, maxIterations])`" and "`state` must be constructed via `new fProxyLQRState(n, allocator)`". Source has no `fProxyLQRState`; the real types are `floatLQRState`/`doubleLQRState` (Assets/LinearAlgebra/Source/OP/Control.float.cs:502 `public struct floatLQRState`, Control.double.cs:502 `public struct doubleLQRState`).
```

The documented warm-state type is a template-only placeholder; no such symbol ships in the generated Source.

**Verifier**

Verified against source. docs/features/control.md:10 and :13 literally contain the codegen placeholder `fProxyLQRState` (`ref fProxyLQRState state`, `new fProxyLQRState(n, allocator)`). No such symbol exists in Assets/LinearAlgebra/Source/ — the real generated types are `floatLQRState` (Control.float.cs:502, ctor at :515) and `doubleLQRState` (Control.double.cs:502, ctor at :515). Per project rules, `fProxy` is a template-only token that must not appear in user-facing docs (`docs/features/*.md` are user-facing per CLAUDE.md). The rest of the same file avoids the `fProxy` prefix on matrix arguments, confirming this is an unintentional leak, not a per-dtype convention. A user copy-pasting the snippet gets a nonexistent type and won't compile. Suggested fix — replace with the concrete per-dtype names, or document one dtype (e.g. `floatLQRState` / `new floatLQRState(n, allocator)`) with a note that a `doubleLQRState` twin exists — matches how other feature docs handle codegen'd types.

**Suggested fix**: Replace `fProxyLQRState` with the concrete per-dtype names (e.g. `floatLQRState state` / `new floatLQRState(n, allocator)`), matching how the rest of the docs name generated types.

### 2. [medium/naming/CONFIRMED] docs/features/decompositions.md:37 — QR's zero-alloc cache overload is documented with the codegen placeholder type `fProxyQRCache`, which is not a real symbol.

**Evidence**

```
Doc: "(3) `ref fProxyQRCache cache` — zero-alloc AND blocked". Source defines `floatQRCache`/`doubleQRCache` (Assets/LinearAlgebra/Source/OP/QR.Workspace.float.cs:56 `public struct floatQRCache`, QR.Workspace.double.cs:56). No `fProxyQRCache` symbol exists.
```

Another `fProxy` template-token leak into a user-facing doc; the generated names differ.

**Verifier**

docs/features/decompositions.md:36 uses `ref fProxyQRCache cache`, but `fProxy` is a codegen template placeholder — no `fProxyQRCache` type exists in the generated Source. Actual types are `floatQRCache` (Assets/LinearAlgebra/Source/OP/QR.Workspace.float.cs:56) and `doubleQRCache` (QR.Workspace.double.cs:56). Sibling user-facing doc docs/features/eigen.md:41 correctly uses the concrete `floatLOBPCGCache` name, confirming the intended convention. Per CLAUDE.md, docs/features/*.md is user-facing and must not carry codegen placeholder tokens; this is a real doc naming bug. Fix: use `floatQRCache`/`doubleQRCache` (or equivalent wording that references the real generated names).

**Suggested fix**: Use `floatQRCache`/`doubleQRCache` (the generated names), matching e.g. `floatLOBPCGCache` already used correctly in eigen.md.

### 3. [medium/naming/CONFIRMED] docs/features/print-export.md:15 — Doc claims `LOBPCGInfo` has no `Print.Log` overload yet, but one exists — the statement contradicts current behavior.

**Evidence**

```
Doc: "`LOBPCGInfo` doesn't have one yet as of this writing but is expected to follow the same convention." Source has it: Assets/LinearAlgebra/Source/Debug/Debug.Info.cs:53 `public static void Log(in LOBPCGInfo info)`.
```

Stale exception clause; the overload is implemented in both the generated output and the template source of truth.

**Verifier**

The doc at C:\Users\viliv\Documents\LinearAlgebraBursted\docs\features\print-export.md lines 15-16 explicitly states: "`LOBPCGInfo` doesn't have one yet as of this writing but is expected to follow the same convention." This is contradicted by both the generated output and the template source of truth:

- C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\Source\Debug\Debug.Info.cs:53 — `public static void Log(in LOBPCGInfo info)` fully implemented, mirroring the other diagnostics-struct overloads (FixedString128Bytes + UnityEngine.Debug.Log).
- C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\TemplateSource\Debug\Debug.Info.cs:49 — same overload present in the template (source of truth), so it's not a stray hand-edit; the feature is real and shipped.

Additionally, the enumeration on lines 12-14 of the doc lists `DirectSolveInfo`, `RankInfo`, `SolveInfo`, `LstsqInfo`, `EigenSolveInfo`, `LanczosInfo` but omits `LOBPCGInfo`, so both the list and the exception clause are stale. Suggested fix as reported: drop the "doesn't have one yet" clause and add `LOBPCGInfo` to the diagnostics-struct enumeration.

**Suggested fix**: Drop the exception clause and add `LOBPCGInfo` to the list of diagnostics structs with a matching `Print.Log(in <Struct>)` (see eigen.md#diagnostics-structs).

### 4. [low/naming/CONFIRMED] README.md:88 — Benchmark row labeled 'dense' describes the operand as 'dense BSR (4×4 blocks, 7% fill)', which is self-contradictory (7% fill is sparse, not dense) and duplicates the sparse row's description.

**Evidence**

```
Line 88: "`Krylov.cg` ... dense | 1024×1024, double; dense BSR (4×4 blocks, 7% fill), 40 iterations | 15.05 ms" vs line 89: "... sparse | ...; sparse BSR (4×4 blocks, 7% fill), 40 iterations | 0.37 ms". Both cite the same '4×4 blocks, 7% fill' yet the dense case is 40× slower — the dense row should describe a dense-matrix operand, not a 7%-fill BSR.
```

Copy-paste artifact: the "dense" row actually measures the dense-storage baseline, not a BSR case.

**Verifier**

README.md:88 labels the operand for the "CG, ... dense" row as "dense BSR (4×4 blocks, 7% fill)". This is self-contradictory in two ways: (1) BSR = Block Sparse Row is a sparse storage format, so "dense BSR" is nonsense terminology, and (2) it duplicates the sparse row's "sparse BSR (4×4 blocks, 7% fill)" operand description (line 89), which reads as a copy-paste artifact.

Verified against the actual benchmark generating those numbers (Assets/LinearAlgebra/Benchmarks/Generated/SparseSolverBenchmark.double.cs, section 4 around lines 750-774, with confirming comment "CG dense-vs-sparse comparison at the same convention (7% block density, K=40, tol=0)"). The harness builds one SPD matrix at 7% block density and runs CG two ways on it: `CG-dense` uses the fully materialized 1024×1024 dense matrix with dense GEMV inside CG; `CG-sparse` uses the BSR view with sparse spMV. The ~40× gap (15.05 ms vs 0.37 ms) is exactly the cost of iterating over ~1M dense entries per iter vs ~70k sparse entries per iter — confirming the dense row is a dense-storage baseline, not a BSR case.

So the dense row's description is incorrect: the storage format is dense, not BSR. It should say something like "dense 1024×1024 storage of the same 7%-fill SPD matrix" and drop the "BSR" token. Severity is low (docs-only, no code or numerical impact).

**Suggested fix**: Change the dense row's operand description to the actual dense-storage case (e.g. 'dense 1024×1024 matrix operand') and drop the '7% fill' from it.

### 5. [low/naming/CONFIRMED] docs/features/dense-types.md:5 — User-facing feature doc links to internal `docs/dev/` spec files, which the doc policy forbids in public docs.

**Evidence**

```
Line 5: "see [spec-interop.md](../dev/spec-interop.md)'s ..."; line 47: "See [rfc-memory-model.md](../dev/rfc-memory-model.md) for the detection mechanism." CLAUDE.md: docs/features/*.md are user-facing with "no internal spec/ticket references".
```

Two live links from a shipping feature doc into the internal `docs/dev/` tree.

**Verifier**

Verified against the actual file at C:\Users\viliv\Documents\LinearAlgebraBursted\docs\features\dense-types.md:

- Line 6: `see [spec-interop.md](../dev/spec-interop.md)'s "Row-major ↔ column-major" section for the full correctness argument.`
- Line 47: `See [rfc-memory-model.md](../dev/rfc-memory-model.md) for the detection mechanism.`

Both `../dev/spec-interop.md` and `../dev/rfc-memory-model.md` exist under `docs/dev/` (confirmed by directory listing) and are unambiguously internal per CLAUDE.md: "`README.md`, `CHANGELOG.md`, and `docs/features/*.md` are user-facing: short, concrete, no dev history, no commit hashes, **no internal spec/ticket references**, no test-class names. `docs/dev/` is internal and exempt."

The finding is a direct policy violation — dense-types.md is a `docs/features/*.md` file (user-facing) and it emits two hyperlinks pointing into the internal `docs/dev/` tree, one of which is even labelled "spec-interop.md" (a spec reference) and the other "rfc-memory-model.md" (an RFC — internal design doc). Neither is a documented contract; both are pointers to internal design/history material the policy forbids exposing to users.

Only nit: the finding reports "line 5", but the `../dev/spec-interop.md` link actually lands on line 6 (the sentence beginning on line 4 wraps to line 6). This is a trivial line-number offset that does not affect the substance of the defect. The second occurrence at line 47 is exact.

Suggested fix (as claimed) is correct: strip the `../dev/*.md` links, either inlining the needed contract (e.g., the transpose reminder is already stated in the same sentence — the "for the full correctness argument" tail is dispensable) or replacing with prose that stands alone.

No mitigating context found: this is a policy compliance defect in a file that will ship in the UPM package's docs surface. Severity "low" is appropriate — cosmetic/policy, not a functional bug.

**Suggested fix**: Remove the `../dev/*.md` links (or inline the needed detail); keep only user-facing references.

### 6. [low/naming/CONFIRMED] docs/features/decompositions.md:9 — User-facing decompositions doc links to the internal `docs/dev/level3-blocking-guide.md` in two places.

**Evidence**

```
Line 9: "see [level3-blocking-guide](../dev/level3-blocking-guide.md) for how that's done"; line 50: "tracked in [level3-blocking-guide](../dev/level3-blocking-guide.md) as GEBRD". Internal dev-doc references are forbidden in public docs per CLAUDE.md.
```

Both links resolve into the explicitly-internal `docs/dev/` tree.

**Verifier**

Verified against C:\Users\viliv\Documents\LinearAlgebraBursted\docs\features\decompositions.md. Line 9 reads `see [level3-blocking-guide](../dev/level3-blocking-guide.md) for how that's done` and line 50 reads `tracked in [level3-blocking-guide](../dev/level3-blocking-guide.md) as GEBRD`. Both links resolve into `docs/dev/`, which CLAUDE.md marks internal. The public-docs rule for `docs/features/*.md` forbids "internal spec/ticket references"; a hyperlink from a user-facing feature doc into an explicitly-internal dev-doc path is that leak. The link target exists on disk (`docs\dev\level3-blocking-guide.md`), so a public reader really does get dropped into internal material. Severity is low (naming/reference leak, not a functional bug) but the finding stands. Suggested fix: remove the `../dev/` links (keep the descriptive prose) or replace with the plain phrase without a hyperlink, since the surrounding sentences already communicate the essential fact (blocked kernel exists, GEBRD not yet blocked).

**Suggested fix**: Drop the `../dev/` links or fold the relevant facts inline.

### 7. [low/naming/CONFIRMED] docs/features/print-export.md:4 — User-facing print/export doc links to internal `docs/dev/spec-debug-print.md`.

**Evidence**

```
Line 4: "Design doc: [spec-debug-print.md](../dev/spec-debug-print.md)." Public feature docs must not reference internal dev/spec docs.
```

An explicit "Design doc" pointer from a public feature page into the internal spec tree.

**Verifier**

Verified against C:\Users\viliv\Documents\LinearAlgebraBursted\docs\features\print-export.md line 4, which reads verbatim: `Design doc: [spec-debug-print.md](../dev/spec-debug-print.md).`

The linked target C:\Users\viliv\Documents\LinearAlgebraBursted\docs\dev\spec-debug-print.md exists, so this is a live cross-doc link from a user-facing feature page into the internal `docs/dev/` tree.

Per project rules (CLAUDE.md, "Public docs" section): "`README.md`, `CHANGELOG.md`, and `docs/features/*.md` are user-facing: short, concrete, no dev history, no commit hashes, no internal spec/ticket references, no test-class names. `docs/dev/` is internal and exempt." An explicit `Design doc:` pointer to `../dev/spec-debug-print.md` is exactly the internal spec reference this rule forbids from user-facing feature docs.

Cross-check: `Grep` for `docs/dev/` across `docs/features/` returns no other matches — this line is the sole offender, not a pattern that's been implicitly permitted elsewhere, and it isn't guarded by any surrounding "internal" heading (it sits on the second content line of the file, directly under the H1). No plausible refutation: the link is not to another `docs/features/*.md` page, not to a README, not to a source file, and the target unambiguously lives under the internal tree.

Suggested fix direction: drop the "Design doc: …" sentence entirely (simplest, matches how the other feature pages read), or move the pointer into a DEVLOG entry / an internal-only doc if discoverability of the spec is still wanted.

Severity assessment matches the original ("low"): it's a doc-policy violation with no runtime impact, but it is a concrete, verifiable breach of a written rule for the pre-v1.0 public docs surface.

Relevant files:
- C:\Users\viliv\Documents\LinearAlgebraBursted\docs\features\print-export.md (line 4)
- C:\Users\viliv\Documents\LinearAlgebraBursted\docs\dev\spec-debug-print.md (link target, confirmed to exist)
- C:\Users\viliv\Documents\LinearAlgebraBursted\CLAUDE.md ("Public docs" section, policy source)

**Suggested fix**: Remove the internal design-doc link.

### 8. [low/naming/CONFIRMED] docs/features/query.md:4 — User-facing query doc links to two internal `docs/dev/` spec files.

**Evidence**

```
Lines 4-5: "Full design rationale: [spec-query.md](../dev/spec-query.md) and [spec-predicate-queries.md](../dev/spec-predicate-queries.md)." Internal spec references are forbidden in public docs.
```

Two live links into the internal spec tree from a shipping feature doc.

**Verifier**

Verified against the code and against the project's documented policy.

Concrete evidence in C:\Users\viliv\Documents\LinearAlgebraBursted\docs\features\query.md lines 3-5:

    `Query` (bare, de-genericized class). Search and selection over the rows/columns of a matrix, or
    flat over a vector. Full design rationale: [spec-query.md](../dev/spec-query.md) and
    [spec-predicate-queries.md](../dev/spec-predicate-queries.md).

Both target files exist at C:\Users\viliv\Documents\LinearAlgebraBursted\docs\dev\spec-query.md and C:\Users\viliv\Documents\LinearAlgebraBursted\docs\dev\spec-predicate-queries.md — so these are real links into the internal `docs/dev/` tree from a public-facing feature doc.

The project's CLAUDE.md is explicit under "Public docs":
    "`README.md`, `CHANGELOG.md`, and `docs/features/*.md` are user-facing: short, concrete, no dev history, no commit hashes, no internal spec/ticket references, no test-class names. `docs/dev/` is internal and exempt."

`docs/features/query.md` falls squarely in the user-facing bucket, and the two markdown links point into `docs/dev/` spec files that are the exact "internal spec references" the policy forbids. Nothing elsewhere carves out an exception. This is not a misread, not a false positive; it is a direct policy violation.

Severity is correctly "low" (cosmetic policy compliance, not a runtime/correctness defect), and the suggested fix (remove the two `../dev/` links, or rewrite the sentence without them) is appropriate. No other public docs concerns to raise on this file.

**Suggested fix**: Remove the `../dev/` spec links.

## Scanner notes

Scanned all 19 listed docs in full and verified every concrete API claim against Assets/LinearAlgebra/Source. Confirmed CORRECT (no defect): README benchmark method names (QR/LU/CHO/QRCP/LQ.minNormSolve, Eigen.symmetricInPlace/valuesSymmetricInPlace/lobpcg, SVD.thin/truncated/randomized, FFT.fft/rfft); QR.solveInPlace returns DirectSolveInfo with status 'Success'; floatLOBPCGCache, floatLPCache/doubleLPCache, LPBasis, floatRollingWindow, Stats.covarianceInto, Norms.L1, Consts.floatZeroThreshold, Blas.columnNormsSquared/buildJacobiScale all exist; CHANGELOG's minNormSolveInPlace on QRCP/LQRP is accurate; all cross-doc feature links (solvers.md/svd.md/sparse-bsr.md/stats.md/select-bits.md) and the referenced docs/dev files physically exist (so no broken links — but the docs/dev references are themselves a policy violation, reported above). LOBPCGInfo field list in eigen.md matches the struct. The two high/medium proxy-name findings are the only ones that would actually break a user's compile.
