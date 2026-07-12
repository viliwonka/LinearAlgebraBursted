# Release scan 2026-07-12 — area: tests-handwritten (non-template)

Scanned 11 files (tests) — counts: total 3, confirmed 3, uncertain 0, unverified 0, refuted 0; severity: high 0, medium 0, low 3.

## Scope

- Assets/LinearAlgebra/SourceTests/ArenaConcurrencyTests.cs
- Assets/LinearAlgebra/SourceTests/ArenaLayoutTests.cs
- Assets/LinearAlgebra/SourceTests/ChunkedRecordTableTests.cs
- Assets/LinearAlgebra/SourceTests/DebugExportTests.cs
- Assets/LinearAlgebra/SourceTests/HashSourceTests.cs
- Assets/LinearAlgebra/SourceTests/LadFrischNewtonQuantileTests.cs
- Assets/LinearAlgebra/SourceTests/QPActiveSetTests.cs
- Assets/LinearAlgebra/SourceTests/QPEqpTests.cs
- Assets/LinearAlgebra/SourceTests/QPSolveTests.cs
- Assets/LinearAlgebra/SourceTests/RandomLongRangeTests.cs
- Assets/Demos/Tests/DemoSmokeTests.cs

## Findings

### 1. [low/pointer/CONFIRMED] Assets/LinearAlgebra/SourceTests/DebugExportTests.cs:26 — Persistent Arena disposed as a trailing statement (no try/finally); any failed assertion leaks the native arena and the leak error masks the real failure message.

**Evidence**

```
Pattern repeated in almost every test here, e.g. FloatToCsvVectorIsOneValuePerLine:
`var arena = new Arena(Allocator.Persistent);` (line 20), asserts at line 24, then
`arena.Dispose();` (line 26). If line 24's Assert.AreEqual fails, line 26 never runs
-> Persistent allocator leak flagged by Unity's leak detector, which surfaces instead
of/alongside the assertion. Same shape at 29-39, 42-51, 54-68, 90-103, 108-117,
120-130, 153-167. Contrast HashSourceTests.cs which correctly wraps every case in
try/finally.
```

Every test in the file allocates a Persistent arena and disposes it only as the last top-level statement after the asserts, so the failure path skips disposal.

**Verifier**

Verified by reading the full DebugExportTests.cs (170 lines). Every one of the 10 tests allocates `new Arena(Allocator.Persistent)` and calls `arena.Dispose()` as a trailing top-level statement after the Assert calls, with no try/finally around the arena lifetime. The line numbers in the claim are exact: FloatToCsvVectorIsOneValuePerLine has arena@20, Assert.AreEqual@24, arena.Dispose()@26; the same shape repeats at 29-39, 42-51, 54-68, 90-103, 108-117, 120-130, 153-167. The two SaveCsv tests (71-88, 133-149) DO use try/finally, but only around the temp-file path — the arena.Dispose() is still outside that block, so an assertion failure inside the try still skips arena disposal. NUnit Assert failures throw AssertionException which unwinds past the trailing Dispose. Contrast confirmed with HashSourceTests.cs (lines 24-26, 41-43, 63-65, 86-88, 110-112, 133-135), which correctly wraps each Allocator.Persistent arena in try { ... } finally { arena.Dispose(); }. Severity is genuinely low (failure-path noise only; Persistent memory is reclaimed on domain reload) but the described defect is real: on any test-assertion failure the Native leak detector will surface a leak error alongside the real assertion, obscuring the failing message. Suggested fix in the claim (mirror the HashSourceTests try/finally pattern) is mechanically correct. File: C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\SourceTests\DebugExportTests.cs. Reference correct-pattern file: C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\SourceTests\HashSourceTests.cs.

**Suggested fix**

Wrap the arena lifetime in try/finally (as HashSourceTests.cs and the QP validation tests already do) so Dispose runs on the assertion-failure path.

### 2. [low/pointer/CONFIRMED] Assets/LinearAlgebra/SourceTests/RandomLongRangeTests.cs:36 — Same leak-on-failure pattern: Persistent Arena disposed only as the final statement, so an Assert.Throws that does not throw leaks the arena.

**Evidence**

```
MinBelowIntRangeThrows creates `var arena = new Arena(Allocator.Persistent);` (line 25),
runs two Assert.Throws (31, 34), then `arena.Dispose();` (line 36) with no try/finally.
If either Assert.Throws fails, Dispose is skipped. Repeated at 39-53, 58-77, 80-89.
```

All four test methods in the class follow the same unguarded dispose pattern.

**Verifier**

Verified against Assets/LinearAlgebra/SourceTests/RandomLongRangeTests.cs. Every one of the four [Test] methods (lines 22-37, 39-54, 58-78, 80-90) instantiates `new Arena(Allocator.Persistent)` at the top and calls `arena.Dispose()` as the final unguarded statement, with NUnit `Assert.Throws` / `Assert.IsTrue` calls between them and no try/finally, no `using`, and no [TearDown] on the class. If any assertion fails (the exact scenario the tests are designed to catch — a broken int-range guard), NUnit's AssertionException unwinds past Dispose and the Persistent arena leaks, triggering Unity's leak detector and polluting subsequent tests. Severity "low" is appropriate: primary test signal still fires; leak is diagnostic-hygiene noise. Suggested fix (try/finally around Dispose, or SetUp/TearDown) is correct.

**Suggested fix**

Put arena.Dispose() in a finally block.

### 3. [low/logical/CONFIRMED] Assets/LinearAlgebra/SourceTests/DebugExportTests.cs:42 — Invariant decimal-point formatting is pinned only for float; the double CSV path is never checked with a fractional value, so a culture-dependent (comma) decimal bug in the double overload would pass unnoticed.

**Evidence**

```
FloatToCsvUsesInvariantDecimalPoint (lines 42-52) writes 1.5f and asserts "1.5\n".
The double tests DoubleToCsvVectorIsOneValuePerLine (108-118) and
DoubleToCsvMatrixIsRowPerLineCommaSeparated (120-131) only use integer-valued data
(1,2,3,4), and DoubleSaveCsvRoundTrips (133-149) compares ToCsv against itself, so
none exercise a fractional double under a non-invariant culture.
```

The double path has no test that would fail if invariant-culture formatting were dropped from the double overload.

**Verifier**

Traced the code path and confirmed the test-coverage gap the finding claims.

FloatToCsvUsesInvariantDecimalPoint (Assets/LinearAlgebra/SourceTests/DebugExportTests.cs:42-52) is the only test in the file (or anywhere in Assets/LinearAlgebra per Grep) that pins the invariant decimal-point contract, and it does so with `1.5f -> "1.5\n"`.

The three double tests all use only integer-valued data:
- DoubleToCsvVectorIsOneValuePerLine (lines 108-118): v[0..2] = 1,2,3 -> "1\n2\n3\n"
- DoubleToCsvMatrixIsRowPerLineCommaSeparated (lines 120-131): entries 1,2,3,4 -> "1,2\n3,4\n"
- DoubleSaveCsvRoundTrips (lines 133-149): v = 1,2,3,4 AND the assertion compares `Print.ToCsv(in v)` against `File.ReadAllText(path)` where the file was written from that same `Print.ToCsv(in v)`. Even for fractional data this is a self-comparison; any culture-dependent formatting would produce equal strings on both sides and the test would still pass.

Production code (Assets/LinearAlgebra/Source/Debug/Export.double.cs:53,65) currently does pass `CultureInfo.InvariantCulture` explicitly, so there is no live bug — but the double contract is unpinned by the test suite. If a future edit dropped the `CultureInfo.InvariantCulture` argument from the double `ToString("G17", …)` calls, a developer running the suite under a comma-decimal culture (de-DE, fr-FR, sl-SI, etc.) would emit `"1,5\n"` and every double test above would still pass while the float sibling would (correctly) fail. The finding's category (logical / test gap), low severity, and suggested fix (add a mirror `DoubleToCsvUsesInvariantDecimalPoint` with e.g. `1.5 -> "1.5\n"`) all hold.

Relevant absolute paths:
- C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\SourceTests\DebugExportTests.cs (lines 42-52, 108-149)
- C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\Source\Debug\Export.double.cs (lines 45-69, generated — do not edit; the template is Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Export.fProxy.cs)

**Suggested fix**

Add a DoubleToCsvUsesInvariantDecimalPoint pinning e.g. 1.5 -> "1.5\n" mirroring the float test.

## Scanner notes

Read all 11 files in full and hand-verified the load-bearing numeric content. Verified correct (no defect): QPActiveSetTests/QPSolveTests HS21/HS35/HS52/HS76 Q,c,A,b literals, constant offsets, and expected optima (0.04; 1/9-9; 1859/349-6; -103/22); the HeavyDegeneracy closed-form obj=-5 (c=-1.5 => -Qx0-r1-r2, 0.5*2-1.5*4); the brute-force 3^n box active-set oracle and its objective 0.5xQx+cx; QPEqpTests KKT saddle assembly signs (top-right -A^T, bottom-left A, rhs -c/b) and Mode-1 objStar=-0.5 x^T Q x; ChunkedRecordTable chunk cumulative capacities 8/24/56/120/248 => 5 chunks at n=200, LIFO free-list order, generation bumps; RandomLongRange guard isolation (int-range vs ordering tested separately); Hash length/cross-type/sign-zero/NaN-payload pins are structurally sound (golden uint constants not independently recomputed but documented as cross-checked). No high/medium defects found: no wrong expected values, no input-only/tautological asserts substituting for output checks, no meaningless tolerances, no success-path native leaks, no missing [Test] attributes, no float/double copy-paste divergences. In-job NUnit Assert.IsTrue does not abort under Burst (confirmed by QPEqpTests' post-assert `arena.Dispose(); return;` guard), so the QP jobs always reach arena.Dispose — the leak-on-failure items are confined to the managed [Test] methods. Reported findings are all low-severity test hygiene only.
