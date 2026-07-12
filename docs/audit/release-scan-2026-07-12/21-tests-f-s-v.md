# Release scan 2026-07-12 — area: tests-f-s-v

Scanned 19 template files (tests). Findings: total 5 — 5 confirmed, 0 uncertain, 0 unverified, 0 refuted; severity: 0 high, 3 medium, 2 low.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDLowRankTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDRandomizedTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDSolverTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDSubspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ScalarMatrixOpTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SelectRefTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SolverBatteryTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SolversTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SpecialConstructorsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/StatsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SvdFullWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SvdRandomizedWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SvdThinValuesWorkspaceTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/TransformsTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/TransposeTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/UnsafeSortTests.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/VectorCopyTests.fProxy.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SpecialConstructorsTests.fProxy.cs:79 — The [Test] RandomRangeMat runs a job whose Execute() switch has no case for RandomRangeMat, so the test silently exercises nothing and always passes.

**Evidence**

```
Execute() switch jumps from `case TestType.RandomMat: RandomMat(); break;` (line 79-81) directly to `case TestType.RotationMat:` (line 82) — there is NO `case TestType.RandomRangeMat`.
```

The enum contains RandomRangeMat (line 32), the method RandomRangeMat() exists (line 279), and `[Test] public void RandomRangeMat()` (line 425) calls `Run()` with Type=RandomRangeMat — but the switch default does nothing, so RandomRangeMat() is never invoked and fProxyRandomMat(...,-6f,6f) range clamping is never verified.

**Verifier**

The Execute() switch at lines 43-91 has no case for TestType.RandomRangeMat and no default arm — it jumps directly from `case RandomMat` (line 79) to `case RotationMat` (line 82). The enum value (line 32), the method `RandomRangeMat()` (lines 279-289) that asserts fProxyRandomMat(16,16,-6f,6f) stays within [-6,6], and the `[Test] RandomRangeMat` (line 425) all exist and wire TType=RandomRangeMat, but Execute() returns without invoking the method, so the range-overload's clamping is never verified and the test passes vacuously.

**Suggested fix**

Add `case TestType.RandomRangeMat: RandomRangeMat(); break;` to the Execute() switch.

### 2. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SpecialConstructorsTests.fProxy.cs:306 — Tautological assertion: `-math.abs(...) < eps` is always true, so the intended check of the rotation matrix off-diagonal element m[0,1] never actually validates anything.

**Evidence**

```
Assert.IsTrue(-math.abs((fProxy)0.70710678118654752440084436210485d - m[0, 1]) < 0.00001f);
```

`-math.abs(x)` is always <= 0, hence always < 0.00001f regardless of m[0,1]. The neighboring lines correctly check m[0,0], m[1,1], m[1,0] via `math.abs(0.7071 - ...) < eps`; this line was meant to verify m[0,1] ≈ -0.7071 but can never fail.

**Verifier**

Line 306 asserts `-math.abs(0.7071 - m[0,1]) < 0.00001f`. Since math.abs is non-negative, its negation is <= 0, which is unconditionally < 0.00001f — the assertion cannot fail for any value of m[0,1]. Cross-checking fProxyRotationMat (ArenaExtensions.fProxy.cs:197: `matrix[i, j] = -s;`), the intended check is m[0,1] ≈ -sin(π/4) = -0.7071, i.e. the ONLY element in this 2x2 test that carries the negative sign. As written, a codegen bug that flipped the sign at matrix[i,j] would go undetected by this test.

**Suggested fix**

Change to `Assert.IsTrue(math.abs((fProxy)(-0.70710678...) - m[0, 1]) < 0.00001f);` (check against the negative expected value).

### 3. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SpecialConstructorsTests.fProxy.cs:190 — IndexZeroMat() and IndexOneMat() allocate a Persistent Arena but never call arena.Dispose(), leaking native memory on every run (all sibling tests dispose).

**Evidence**

```
IndexZeroMat() (lines 190-197): `var arena = new Arena(Allocator.Persistent); var m = arena.fProxyIndexZeroMat(16, 16); for(...) Assert.IsTrue(m[i] == (fProxy)i); }` — closing brace with no `arena.Dispose();`.
```

IndexOneMat() (lines 199-207) has the identical omission. Every other method in the struct (BasisVec, IdentityMat, DiagonalMat, etc.) ends with `arena.Dispose();`.

**Verifier**

Verified at Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SpecialConstructorsTests.fProxy.cs:190-207: IndexZeroMat() and IndexOneMat() both create `new Arena(Allocator.Persistent)` and fall off the end without calling arena.Dispose(), while every other sibling in the same struct (BasisVec, LinVec at 187, IdentityMat at 209, etc.) does dispose. Both methods are wired into the Execute() switch (lines 73-78) so they actually run, and Persistent leaks survive the job boundary — Unity will report leaked native allocations. Fix: add `arena.Dispose();` before the closing brace of each.

**Suggested fix**

Add `arena.Dispose();` before the closing brace of both IndexZeroMat() and IndexOneMat().

### 4. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SpecialConstructorsTests.fProxy.cs:300 — The mᵀm identity check computes Analysis.isIdentity(mTm) but discards its bool return with no Assert, so the check is dead (occurs in RotationMat, PermutationMat, HouseholderMat).

**Evidence**

```
var mTm = Blas.dot(m, m, true);
Analysis.isIdentity(in mTm, 0.00001f);
```

Lines 299-300: return value ignored. Same pattern at line 321 (PermutationMat) and line 343 (HouseholderMat). Only the separate `Assert.IsTrue(Analysis.isOrthogonal(in m, ...))` actually enforces orthogonality; the mTm line asserts nothing.

**Verifier**

Lines 300, 321, and 343 all read `Analysis.isIdentity(in mTm, 0.00001f);` with no `Assert.IsTrue` wrapper. `Analysis.isIdentity` (defined at Analysis.fProxy.cs:105) is a pure bool predicate with no side effects, so the return value is silently discarded and the intended assertion never fires. The redundant `Assert.IsTrue(Analysis.isOrthogonal(...))` above each dead call means the acceptance criterion is still covered, so severity is low as reported, but the dead-code claim is factually correct.

**Suggested fix**

Wrap in an assert: `Assert.IsTrue(Analysis.isIdentity(in mTm, 0.00001f));` (or delete as redundant with the isOrthogonal check).

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SVDTests.fProxy.cs:273 — ThinKnownWideViaTranspose_6x15 never builds a wide matrix nor calls a transpose; it just runs SVD.thin on a tall 15×6 T, so the name/comment ('validates the transpose contract for wide inputs') overstates what is exercised.

**Evidence**

```
Comment (lines 270-272) claims the route thin(in trans(W)) is validated, but the body builds `var T = arena.fProxyMat(m, n);` (15×6) via BuildRandSvd and calls `CheckThinKnown(in T, in sigma, m, n, ref arena);` — no wide W is constructed and Blas.trans is never called.
```

It is functionally a duplicate tall-thin test; the documented wide→transpose entry path is not covered.

**Verifier**

SVDTests.fProxy.cs lines 273-285: the body allocates `T = arena.fProxyMat(m, n)` with m=15,n=6 (a tall matrix), calls BuildRandSvd on T, then CheckThinKnown(in T, ...) which just invokes SVD.thin(in T, ...). No 6x15 wide W is constructed and Blas.trans is never called anywhere in the function. The comment at 270-272 explicitly claims "Validates the transpose contract for wide inputs" and describes the route `thin(in trans(W))`, but the transpose call and wide input do not exist in the code — the test is structurally identical to the other tall-thin CheckThinKnown cases (Geometric/Arithmetic/Clustered/FlatCliff), just at aspect 15x6. A genuine name/comment vs behavior mismatch, matching the low-severity naming classification.

**Suggested fix**

Actually construct the wide W (6×15) and call SVD.thin(in Blas.trans(W), ...) so the transpose contract is genuinely exercised, or rename the case to reflect that it only tests a tall matrix.

## Scanner notes

Scanned all 19 listed template test files in full. The SVD family (SVDLowRankTests, SVDRandomizedTests, SVDSolverTests, SVDSubspaceTests, SVDTests, SVDWorkspaceTests, SvdFullWorkspaceTests, SvdRandomizedWorkspaceTests, SvdThinValuesWorkspaceTests), StatsTests, TransformsTests, SelectRefTests, SolverBatteryTests, TransposeTests, UnsafeSortTests, and VectorCopyTests are solid: tolerances are per-precision (scaled by Consts.fProxySqrtEps so both float and double expansions hold), Allocator.Temp/Persistent buffers are disposed on all paths, and assertions match documented behavior. Separately: many test comments carry dev-history / commit-hash / benchmark-verdict narration (e.g. 'de74c48', 'FIX 1/FIX 2', 'Measured worst relative error ... < 1e-4', 'Solver API rework (commit 2)') which technically violates the CLAUDE.md contracts-only comment policy; I did not file these individually since these test files never ship in the UPM package and they are not correctness defects, but a policy sweep of test comments may be warranted if the rule is meant to apply to SourceTests too. SolversTests has two dead enum entries (USolveIdentity, LSolveIdentity) with no switch case and no [Test], but no test claims to cover them, so no false-pass results.
