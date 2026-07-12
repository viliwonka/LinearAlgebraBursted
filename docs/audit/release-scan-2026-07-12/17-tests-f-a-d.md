# Release scan 2026-07-12 — area: tests-f-a-d

Scanned 22 template files (tests). Findings: total 6 — 5 confirmed, 0 uncertain, 0 unverified, 1 refuted; severity: 0 high, 4 medium, 1 low (refuted finding excluded from severity counts where applicable).

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/AccuracySweepTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/AnalysisTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ArenaConversionsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ArenaHandleTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ArenaWiringTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BidiagTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BidiagWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BridgeFillTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/CHOPTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/CHOPWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/CHOTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ChooseMarkerTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/CompMathTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/CompareTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ConjugateGradientTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ControlLQRTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ControlTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ConvergenceBudgetTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DebugPrintTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DotOperationTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DotRefGuardTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DotRefTests.fProxy.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DotOperationTests.fProxy.cs:169 — MatMatDot's identity-self-product sub-check computes D = dot(C,C) but then asserts on C (the identity input) instead of D, so the product is never validated (tautological assertion).

**Evidence**

```
C = arena.fProxyIdentityMat(matLen);
fProxyMxN D = Blas.dot(C, C);
... if (i == j) Assert.IsTrue(C[i, j] == (fProxy)1f); else Assert.IsTrue(C[i, j] == (fProxy)0f);
```

The loop reads C[i,j] (freshly built identity => trivially true) and never touches D, the actual matrix product under test.

**Verifier**: Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DotOperationTests.fProxy.cs:167-178 assigns C = fProxyIdentityMat(matLen), computes D = Blas.dot(C, C), but the following double loop asserts C[i,j] == 1 on diagonal / 0 elsewhere. C is the freshly constructed identity so those assertions are tautological; D is never read, meaning the self-product path of MatMatDot is not validated. Fix: assert on D[i,j] instead of C[i,j].

**Suggested fix**: Assert on D[i,j] (== 1 on the diagonal, 0 off-diagonal), not C[i,j].

### 2. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/DotOperationTests.fProxy.cs:217 — MatMatDotNonSquare has an empty body, yet the [Test] MatrixMatrixDotNonSquareTest runs it — non-square matrix*matrix multiplication is claimed to be covered but nothing is exercised.

**Evidence**

```
public void MatMatDotNonSquare()
{

}
... [Test] public void MatrixMatrixDotNonSquareTest() { new DotOperationTestsJob() { Type = ...MatMatNonSquare }.Run(); }
```

**Verifier**: DotOperationTests.fProxy.cs line 217-220 defines `MatMatDotNonSquare()` with an empty body. The switch at line 56-58 dispatches `TestType.MatMatNonSquare` directly to this empty method, and `[Test] MatrixMatrixDotNonSquareTest` (line 298-301) runs the job with exactly that type. Sibling non-square tests (`MatVecDotNonSquare`, `VecMatDotNonSquare`) do have real bodies, and the square `MatMatDot` does exercise real assertions, so the empty method is not a codegen stub or intentional dead path — the non-square GEMM case is claimed to be covered but nothing runs, in both float and double expansions.

**Suggested fix**: Implement the non-square A(MxK)*B(KxN) case with a value check against a hand/oracle product, mirroring MatMatDot.

### 3. [medium/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/CompMathTests.fProxy.cs:382 — RemapTest comment claims the test currently FAILS due to a live argument-order bug in the production kernel, but the kernel is correct and the test passes — the comment is stale/false and also embeds a reviewer/agent note.

**Evidence**

```
// NOTE: this asserts the intended behaviour and currently FAILS - the kernel
// (UnsafeMathOP.remap) calls math.remap(x, oldMin, oldMax, newMin, newMax) ... See the
// agent report: this is a genuine argument-order bug in the production kernel
```

The actual kernel (UnsafeMathOP.fProxy.cs:284) does math.remap(oldMin, oldMax, newMin, newMax, x[i]) — value LAST, matching the test oracle at line 387, so the test passes.

**Verifier**: Kernel at Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeMathOP.fProxy.cs:284 calls math.remap(oldMin, oldMax, newMin, newMax, x[i]) — value LAST, matching the test oracle at CompMathTests.fProxy.cs:387 exactly, so the test passes. The comment block at lines 381-385 claims the opposite (kernel passes value FIRST, test currently FAILS) — this is false. Additionally, the "See the agent report" phrasing violates the CLAUDE.md contracts-only comment policy which bans agent/reviewer references in code comments (belongs in DEVLOG).

**Suggested fix**: Delete the stale/false paragraph (value is correctly passed last); move any history to DEVLOG per the contracts-only comment policy.

### 4. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ConjugateGradientTests.fProxy.cs:281 — SingularConsistent is defined in the enum and implemented but has no [Test] wrapper, so the rank-deficient/consistent CG acceptance case never actually runs.

**Evidence**

```
void SingularConsistent() { ... } // enum value at line 23, dispatch at line 63
```

Every other enum case has a matching [Test] (AddScaledInPlaceTest ... GalleryMinIJTest) but there is no SingularConsistentTest method, so TestType.SingularConsistent is never invoked.

**Verifier**: The enum value TestType.SingularConsistent (line 23), the switch case (lines 63-65), and the method SingularConsistent() (lines 281-305) all exist, but the [Test]-attributed wrapper block (lines 403-467) contains 11 wrappers covering every other enum value and no SingularConsistentTest. NUnit therefore never invokes the singular-consistent case, so the documented rank-deficient acceptance check (no NaN, and if ok then A x = b) is silently not exercised. Fix: add [Test] public void SingularConsistentTest() => new ConjugateGradientTestJob { Type = ConjugateGradientTestJob.TestType.SingularConsistent }.Run(); to the template at Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ConjugateGradientTests.fProxy.cs.

**Suggested fix**: Add [Test] public void SingularConsistentTest() => new ConjugateGradientTestJob { Type = ...SingularConsistent }.Run();

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ControlTests.fProxy.cs:11 — Code comments reference agents/roles and commit/stage tickets, which the project's contracts-only comment policy says must live in DEVLOG, not source.

**Evidence**

```
// BASIC smoke tests ... written by the coder agent alongside the implementation. The FULL test battery ... is the test-writer agent's job
```

Similar dev/agent/commit references appear elsewhere (e.g. CHOTests.fProxy.cs:41 'Solver API rework (commit 2)', :44 'Commit 2.5 (2a)'; ArenaHandleTests header 'FM2'; ArenaWiringTests 'Stage E').

**Verifier**: Verified by direct read. ControlTests.fProxy.cs line 11 literally contains "written by the coder agent" and "test-writer agent's job" plus dev-history narration ("BASIC smoke tests ... the FULL test battery ... is the test-writer agent's job"), which CLAUDE.md's contracts-only comment policy explicitly bans in source and directs to DEVLOG.md. The corroborating examples at CHOTests.fProxy.cs:41 ("Solver API rework (commit 2)") and :44 ("Commit 2.5 (2a)") are real internal ticket/commit references, also forbidden. This is a genuine contradiction between the code comments and the stated project policy.

**Suggested fix**: Move development history, agent/role notes, and commit/stage/ticket references to the folder DEVLOG.md; keep only contract statements in comments.

## Refuted

| file:line | claim | why refuted |
|---|---|---|
| Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/AnalysisTests.fProxy.cs:180 | IsDiagonalEpsilon builds a 'random' diagonal matrix with min == max == -1f, making the RNG degenerate (all diagonal entries -1) so the 'random' aspect is vacuous. | Line 180 constructs a diagonal matrix (all diagonal entries = -1, off-diagonals = 0) via fProxyRandomDiagonalMat with degenerate range; Analysis.isDiagonal(A, 1e-6) correctly returns true, the assertion passes, and the test verifies its stated contract. The reviewer's own evidence only observes that randomness is vacuous, which is a test-quality nit (mild redundancy with the identity check at line 174) rather than a defect — no failing scenario, no wrong behavior, no leaked/aliased memory. Severity "low" plus category "logical" don't correspond to any concrete incorrectness in the code. |

## Scanner notes

Verification: I confirmed the RemapTest finding by reading the production kernel at Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeMathOP.fProxy.cs:279-285 — it passes the value as the LAST argument to math.remap, which matches the test's oracle, so the test passes and the comment claiming a live production bug/failing test is false. The remaining ~18 files (AccuracySweep, ArenaConversions, ArenaHandle, ArenaWiring, Bidiag, BidiagWorkspace, BridgeFill, CHOP, CHOPWorkspace, CHO, ChooseMarker, CompMath (aside from the remap comment), Compare, ConvergenceBudget, ControlLQR, DebugPrint, DotRefGuard, DotRef) were read in full and are functionally sound: allocations are arena/Temp-scoped and disposed on all paths, tolerances are precision-scaled via Consts.fProxySqrtEps/Epsilon (no double-only hardcodes), residual oracles accumulate in double, and assertions carry real expected values. Many of these files also carry heavy dev-history/agent/commit narration in comments (contracts-only policy violations) but I did not enumerate each; the ControlTests entry is representative.
