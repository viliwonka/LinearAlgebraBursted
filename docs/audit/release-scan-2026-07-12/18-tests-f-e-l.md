# Release scan 2026-07-12 — area: tests-f-e-l

Scanned 24 template files (tests). Findings: total 4 — 4 confirmed, 0 uncertain, 0 unverified, 0 refuted; severity: 0 high, 2 medium, 2 low.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/EigenQRTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/EigenSymWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/EigenTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/FFTTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/FullStatsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/GalleryPhase2Tests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/GalleryTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/GeneratorTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/HashTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/HistogramTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/InPlaceOpTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/IndexingTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/InitTest.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/JacobiPrecondTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KMeansTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovFusedKernelTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovRound2Tests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovVerifyAtExitTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LOBPCGSmokeTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LPTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LQRPTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LQWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LUTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LiteratureTests.fProxy.cs

## Findings

### 1. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/IndexingTests.fProxy.cs:157 — RandomCalc() creates a Persistent Arena and never disposes or clears it, leaking the arena core plus all its allocations on every run.

**Evidence**

```
public void RandomCalc()
{
    var arena = new Arena(Allocator.Persistent);
    ...
    Assert.IsTrue(mat[r, c] == (fProxy)(r * c * 2));
}  // no arena.Dispose();
```

Every sibling method (VectorIndexing/MatrixIndexing1D/MatrixIndexing2D) ends with arena.Dispose(). Arena.Dispose() is the only path that frees the UnsafeUtility.Malloc'd core; Clear() (not even called here) would not free it. The core is raw unmanaged memory so Unity's NativeContainer leak detector never flags it — a silent leak.

**Verifier**

RandomCalc() at lines 157-177 of IndexingTests.fProxy.cs creates `new Arena(Allocator.Persistent)` on line 159 and returns at line 177 with no `arena.Dispose()`. The three sibling methods in the same IndexingTestJob (VectorIndexing L77, MatrixIndexing1D L108, MatrixIndexing2D L154) all dispose their arenas — RandomCalc is the sole omission. The [Test] at L200 runs the job via .Run(), so the leak occurs every time the test executes; being a template, it doubles after codegen (float + double).

**Suggested fix**

Add arena.Dispose(); before the method returns (mirror the other three Indexing methods).

### 2. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/InitTest.fProxy.cs:26 — InitVecTestJob calls arena.Clear() but never Dispose(); Clear() only resets allocations and keeps the Persistent-allocated arena core, so the arena backing buffer leaks.

**Evidence**

```
var arena = new Arena(Allocator.Persistent);
... arena.Clear();
Assert.AreEqual(0, arena.AllocationsCount);
// method ends — no arena.Dispose().
```

Arena.Clear() = ClearCore()+ClearTempCore() (resets allocation tracking only); only Arena.Dispose() does _core->Dispose()+UnsafeUtility.Free(_core). The sibling InitMatrixTestJob correctly calls arena.Dispose(). The leaked core is raw UnsafeUtility memory, so it passes CI silently.

**Verifier**

InitVecTestJob at InitTest.fProxy.cs:15-29 creates a Persistent-allocated Arena and calls arena.Clear() (line 26) but never arena.Dispose(). Arena.cs:508-533 shows the ctor UnsafeUtility.Malloc's _core in the Persistent allocator; only Arena.Dispose() (Arena.cs:585-594) does UnsafeUtility.Free(_core, allocator). Arena.Clear() (Arena.cs:225-238, 570) routes to ClearCore+ClearTempCore which only walk the record tables — they never free the _core block. The paired InitMatrixTestJob at line 53 correctly calls arena.Dispose(), confirming the asymmetry: the vec test leaks the persistent ArenaCore each Run().

**Suggested fix**

After asserting AllocationsCount==0, call arena.Dispose() to free the core (Clear() is not a substitute for Dispose()).

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LOBPCGSmokeTests.fProxy.cs:11 — Class/method header comments carry agent-workflow and reviewer narration, which the contracts-only comment policy explicitly forbids in code.

**Evidence**

```
"SCRATCH / smoke-test coverage for the new Eigen.lobpcg implementation, written by the coder agent purely to sanity-check the algorithm while iterating ... This is NOT the comprehensive suite the spec calls for ... that is left for the independent test-writer agent."
```

CLAUDE.md: 'notes to reviewers or references to agents/workflow ... never go in code comments' (belongs in the folder DEVLOG.md).

**Verifier**

Lines 11-25 of Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LOBPCGSmokeTests.fProxy.cs contain explicit agent-workflow narration ("written by the coder agent", "left for the independent test-writer agent", quoted task-brief text, and a bullet list of tests deferred to another agent). CLAUDE.md's strict comment policy states code comments must state contracts only and explicitly forbids "notes to reviewers or references to agents/workflow" — that content belongs in the folder's DEVLOG.md. This is a genuine contradiction between the file's comments and the project's comment policy.

**Suggested fix**

Move the agent/workflow and 'not the comprehensive suite / left for the test-writer agent' narration into the template folder's DEVLOG.md; keep only contract-stating comments in the file.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/InPlaceOpTests.fProxy.cs:7 — Header comment references an internal PR and narrates bug postmortem history, contrary to the contracts-only comment policy.

**Evidence**

```
// Regression tests (from PR #1's ideas): ... the internal compAdd operands were reversed, so the method used to mutate the wrong operand — masked end-to-end only because the + operators also called it backwards.
```

Dev history + internal ticket/PR reference that CLAUDE.md routes to DEVLOG.md, not code comments.

**Verifier**

Lines 7-13 of InPlaceOpTests.fProxy.cs contain both a "PR #1" internal ticket reference and a bug postmortem ("operands were reversed... masked end-to-end only because the + operators also called it backwards"). CLAUDE.md's comment policy explicitly lists both "internal spec/ticket references" and "bug postmortems and debugging narration" as content that belongs in DEVLOG.md rather than code comments. This is a genuine contradiction between the file's contents and the project's stated comment policy.

**Suggested fix**

Replace with a short contract-only description of what the tests assert; relocate the PR reference and bug postmortem to DEVLOG.md.

## Scanner notes

Scope: read all 24 listed fProxy test templates in full. These are fProxy templates, so they only expand to float/double (int/uint pitfalls are out of scope for these files); tolerance/precision handling and the //+choose[...] per-precision gating are consistent throughout. The numerical oracles (known eigenvalue/SVD/LU/LP/FFT closed forms, cross-checks, and per-precision sqrtEps-scaled bands) are sound and I found no wrong-expected-value, tautological-assertion, or skipped-loop defects. Destructive-contract usage (solveInPlace/decompInPlace destroying inputs) is correctly guarded with .Copy() before every reuse, and the many Pivot/NativeArray/senses allocations are disposed on all paths I traced (including the managed Assert.Throws tests, which dispose in finally). The only concrete defects are the two silent arena leaks (raw UnsafeUtility core memory that Unity leak detection does not track, so they pass CI green while still leaking). The two comment-policy findings are representative — several other test files in this set (LPTests, KrylovVerifyAtExitTests, LQRPTests, GalleryPhase2Tests) also embed spec/ticket references (e.g. 'task T1–T10', 'R6a contract', dated dev notes) and first-person postmortem narration that the same contracts-only policy would move to DEVLOG.md; I did not enumerate each to avoid noise.
