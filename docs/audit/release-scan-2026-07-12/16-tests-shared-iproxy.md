# Release scan 2026-07-12 — area: tests-shared-iproxy

Scanned 35 template files (tests). Findings: total 6 — confirmed 6, uncertain 0, unverified 0, refuted 0; severity: high 0, medium 3, low 3.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolAnalysisTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolBridgeTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolDebugExportTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolHashTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolIndexingTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolOperationsTest.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolRandomTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/DebugInfoTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/RandomSharedTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxyPivotTests.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/bool/ArenaWiringTests.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/ML/PCATests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/AnalysisTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/ArenaWiringTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/BridgeFillTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/ChooseMarkerTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/ClampTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/CompBitsTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/CompMathTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/CompareTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/DebugExportTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/DotOperationTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/DotRefTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/HashTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/IndexingTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/InitTest.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/NormsTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/OperationsTest.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/QueryPredicateTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/QueryTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/RandomTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/ScalarMatrixOpTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/SelectRefTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/StatsTests.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/TransposeTests.iProxy.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/DotOperationTests.iProxy.cs:169 — Self-dot sub-test computes D = dot(C,C) but then asserts on C instead of D, so the product is never verified.

**Evidence**

```csharp
C = arena.iProxyIdentityMat(matLen);
iProxyMxN D = Blas.dot(C, C);
for (...) { if (i == j) Assert.IsTrue(C[i, j] == (iProxy)1f); else Assert.IsTrue(C[i, j] == (iProxy)0f); }  // asserts on C (already identity), D is unused
```

The verification loop asserts on `C[i, j]`, which is trivially the identity by construction, while `D` — the actual product — is never read.

**Verifier**

At DotOperationTests.iProxy.cs:167-178, C is reassigned to a fresh identity matrix at line 167, then D = Blas.dot(C, C) is computed at line 169, but the verification loop asserts on C[i,j] — which is trivially the identity by construction — while D is never read. Any bug in the self-dot code path (e.g., aliasing when the same buffer is passed as both operands) would leave this test silently green. The two preceding sub-tests correctly assert on the product matrix (lines 148-155 on C = dot(A,B), lines 161-165 on C = dot(A,R)), confirming the third block's intent was to check D, not C. Fix: replace C[i,j] with D[i,j] in the loop at lines 175/177.

**Suggested fix**

Assert on D[i,j] (the actual matrix product) rather than C[i,j]; C is trivially identity so the loop passes without ever checking dot(C,C).

### 2. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/DotOperationTests.iProxy.cs:217 — MatMatDotNonSquare() has an empty body, so the [Test] MatrixMatrixDotNonSquareTest asserts nothing (vacuous pass) despite promising non-square matrix*matrix coverage.

**Evidence**

```csharp
public void MatMatDotNonSquare()
{

}
... [Test] public void MatrixMatrixDotNonSquareTest() { new DotOperationTestsJob() { Type = ...MatMatNonSquare }.Run(); }
```

The [Test] dispatches to an empty method body, so the test passes without exercising any non-square GEMM code path.

**Verifier**

Lines 217-220 of the iProxy template contain an empty MatMatDotNonSquare() body, and the [Test] MatrixMatrixDotNonSquareTest (lines 297-301) dispatches to it via TestType.MatMatNonSquare in the switch at line 56, so the test passes without asserting anything - a real vacuous-pass gap in non-square GEMM coverage. The paired fProxy template has the identical empty stub at the same lines, confirming this is a missing implementation across both codegen variants, not something filled in elsewhere.

**Suggested fix**

Implement a non-square GEMM check (e.g. (MxK)*(KxN) against a hand-computed or identity-based oracle) or remove the empty test so the absence of coverage is not disguised as a passing test.

### 3. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/IndexingTests.iProxy.cs:162 — RandomCalc() allocates a Persistent Arena but never disposes it, leaking native memory on every test run (all sibling methods dispose).

**Evidence**

```csharp
public void RandomCalc()
{
    var arena = new Arena(Allocator.Persistent);
    ... (fills/asserts) ...
    Assert.IsTrue(mat[r, c] == (iProxy)(r * c * 2));
}  // no arena.Dispose()
```

The Persistent-allocator arena is never disposed, unlike every sibling method in the same struct.

**Verifier**

Read lines 160-180 of the iProxy template: RandomCalc() creates `new Arena(Allocator.Persistent)` on line 162, allocates `arena.iProxyMat(rows, cols)` on line 167, and returns after the final Assert on line 179 without calling arena.Dispose(). All three sibling methods in the same struct dispose their arenas (lines 77, 109, 157). The exact same bug also exists in the paired fProxy template (Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/IndexingTests.fProxy.cs lines 157-177), so the leak is doubled after codegen expansion (fProxy -> float+double, iProxy -> int+uint = four leaked Persistent arenas per RandomCalc test run, which will trip Unity's leak detector).

**Suggested fix**

Add arena.Dispose() at the end of RandomCalc() (matching VectorIndexing/MatrixIndexing1D/2D) to avoid a Persistent-allocator leak that can trip Unity's leak sentinel.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/ML/PCATests.fProxy.cs:626 — Code comment references an agent/workflow ('The coder confirmed ...'), violating the contracts-only comment policy.

**Evidence**

```csharp
// The coder confirmed svdTruncated is NOT shape-free; pcaSVDTruncated adds the n>=p guard, so
// it throws on wide data just like pcaSVD/pcaRandomized ...
```

The comment narrates a workflow/agent role ("the coder"), which the strict comment policy forbids in code comments.

**Verifier**

Line 626 literally begins "// The coder confirmed svdTruncated is NOT shape-free; ..." — direct reference to an agent/workflow role ("the coder"), which CLAUDE.md's strict comment policy explicitly forbids in code comments ("notes to reviewers or references to agents/workflow"). The contract fact (svdTruncated requires n>=p; wide data throws) is legitimate but must be stated without the workflow narration; the "deliberately NO 'truncated works on wide data' test" aside is also review/dev narrative that belongs in DEVLOG.md.

**Suggested fix**

Reword to state the contract only (svdTruncated requires n>=p; wide data throws) and move any workflow/agent narration to the folder DEVLOG.md.

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/QueryTests.iProxy.cs:699 — Code comment embeds a reviewer-note reference ("review's HIGH finding"), violating the contracts-only comment policy.

**Evidence**

```csharp
// MinValue EDGE — the iAbs() off-by-one fix (review's HIGH finding).
```

The "(review's HIGH finding)" clause is a reviewer-note reference, which the comment policy routes to DEVLOG.md.

**Verifier**

Line 699 of Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/QueryTests.iProxy.cs contains verbatim: "// MinValue EDGE — the iAbs() off-by-one fix (review's HIGH finding)." CLAUDE.md's comment policy is explicit: code comments state contracts only, and "notes to reviewers or references to agents/workflow" belong in DEVLOG.md, not in code. The "(review's HIGH finding)" clause is exactly the forbidden reviewer-note reference. Fix by dropping the reviewer reference (and ideally the "off-by-one fix" history) and keeping only a contract-shaped note about the MinValue/iAbs saturation edge; move the history to the folder DEVLOG.md.

**Suggested fix**

Drop the '(review's HIGH finding)' reference; keep only the contract description of the MinValue/iAbs edge behavior. Move history to DEVLOG.md.

### 6. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/iProxy/ScalarMatrixOpTests.iProxy.cs:9 — Header comment narrates a bug postmortem ('the operator delegated to rhs - lhs, which negates ...') in code, which the comment policy routes to DEVLOG.md.

**Evidence**

```csharp
// Regression test for the `scalar - matrix` sign bug (integer matrices): the operator delegated to
// `rhs - lhs`, which negates the result since subtraction is not commutative.
```

The header describes the prior implementation's bug rather than stating the contract, which the policy forbids in code comments.

**Verifier**

Lines 9-10 contain a bug postmortem describing the prior implementation ("the operator delegated to `rhs - lhs`, which negates the result since subtraction is not commutative"). The CLAUDE.md comment policy explicitly forbids "bug postmortems and debugging narration" and "development history" in code comments, routing them to DEVLOG.md; comments must state contracts only. Line 23's inline contract comment ("5 - [[1,2],[3,4]] must be [[4,3],[2,1]]") already carries the contract, so the header narration is redundant policy violation.

**Suggested fix**

State the contract (scalar - matrix must equal s - A[i,j], non-commutative) without the postmortem of the prior delegation bug; move the bug history to DEVLOG.md.

## Scanner notes

Scope: 35 test-template files across the bool / shared / iProxy / ML(PCA) surfaces. Overall these are high-quality tests: tolerances are scaled via Consts.fProxySqrtEps (loose-for-float/tight-for-double), per-type //+choose literals correctly pin abs-overflow / ceilpow2-wrap / widened-sum overflow behavior for short vs int vs long vs uint, integer oracles are exact, and Dispose/aliasing/generation-guard contracts are exercised carefully. The only functional test defects are the two in DotOperationTests (unused D self-dot check; empty MatMatDotNonSquare) and the RandomCalc arena leak. Random-vector 'IsFalse(IsAllSame)' style asserts (BoolAnalysisTests, CompareTests) rely on non-degenerate random fills but are astronomically safe with the fixed seeds/dims used, so not flagged. The three comment-policy items are low severity but explicitly in-scope per the project's contracts-only rule.
