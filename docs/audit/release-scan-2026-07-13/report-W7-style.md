# W7 -- Style & consistency

Scanner: W7 (style, code smell, sore thumbs -- templates only)
Date: 2026-07-13

---

## Findings

### 1. MEDIUM -- maxIter vs maxIterations parameter name split across the public API

**Files (maxIter):** LP.fProxy.cs, LP.BarrodaleRoberts.fProxy.cs, LP.FrischNewton.fProxy.cs,
LP.InteriorPoint.fProxy.cs, LP.RevisedSimplex.fProxy.cs, LP.DualSimplex.fProxy.cs,
LP.Sparse.fProxy.cs, QP.fProxy.cs, MIP.fProxy.cs, MPC.fProxy.cs, NLS.fProxy.cs,
Optimize.fProxy.cs, PCA.fProxy.cs (13 files)

**Files (maxIterations):** Krylov.fProxy.cs, Krylov.PBiCGStab.fProxy.cs, Eigen.fProxy.cs,
SVD.fProxy.cs, SVD.LowRank.fProxy.cs, SVD.Subspace.fProxy.cs, SVD.Randomized.fProxy.cs,
SVD.RandomizedWorkspace.fProxy.cs, LOBPCG.fProxy.cs, Control.fProxy.cs, Kalman.fProxy.cs,
KMeans.fProxy.cs (12 files)

The T5 BREAKING rename (maxIterations/tolerance) was applied to the iterative-solver family
(Krylov, Eigen, SVD, LOBPCG, Control, Kalman, KMeans) but missed the optimization-solver family
(LP, QP, MIP, MPC, NLS, Optimize, PCA). A user writing LP.solve(..., maxIter: 100) and
Krylov.cg(..., maxIterations: 100) sees two names for the same concept in the same library.
This is visible in the generated public package.

**Suggested fix:** Rename maxIter to maxIterations in the 13 files that still use the old
form, or document the intentional split if it was deliberate (but neither the naming-style-guide
nor the OP/DEVLOG.md record such a decision).

---

### 2. MEDIUM -- Retired term BSM in public XML doc comment

**File:** TemplateSource/OP/LOBPCG.fProxy.cs:623

BSM was renamed to BSR library-wide. This stale term survives in a summary XML doc
comment on a public method (lobpcg(in fProxyBSR ...)), so it will appear in generated
IntelliSense / doc output for both floatBSR and doubleBSR.

**Suggested fix:** Replace BSM with BSR in the doc comment.

---

### 3. MEDIUM -- UNITARY OPERATIONS region should be UNARY OPERATIONS

**Files:**
- TemplateSource/bool/boolN.Operators.cs:53
- TemplateSource/bool/boolMxN.Operators.cs:54

Unitary is a specific mathematical term (norm-preserving linear operator / unitary matrix).
The intended word is unary (single-operand). These region tags survive verbatim into the
generated boolN.Operators.cs and boolMxN.Operators.cs in the public UPM package.

**Suggested fix:** Rename to UNARY OPERATIONS.

---

### 4. MEDIUM -- fProxySolversTests named after retired Solvers class + dead enum values

**File:** TemplateSourceTests/fProxy/SolversTests.fProxy.cs

The test class is named fProxySolversTests but tests QR.decompInPlace and Blas.triUpper,
not a Solvers class (which was retired and split into Krylov/Blas/SVD/Eigen). The
TestType enum declares USolveIdentity and LSolveIdentity values that have no corresponding
switch cases or test methods -- dead code. Also has unused using System.Collections and
using System.Collections.Generic.

**Suggested fix:** Rename file and class to TriangularSolveTests.fProxy.cs /
fProxyTriangularSolveTests, delete the dead enum values, and remove unused usings.

---

### 5. MEDIUM -- SVD workspace test classes use inconsistent casing (Svd vs SVD)

**Files:**
- SvdThinValuesWorkspaceTests.fProxy.cs -> class fProxySvdThinValuesWorkspaceTests
- SvdFullWorkspaceTests.fProxy.cs -> class fProxySvdFullWorkspaceTests
- SvdRandomizedWorkspaceTests.fProxy.cs -> class fProxySvdRandomizedWorkspaceTests

All other SVD test files use all-caps SVD: fProxySVDTests, fProxySVDSolverTests,
fProxySVDWorkspaceTests, fProxySVDLowRankTests, etc. Per the naming style guide, SVD
is a literature-recognized acronym and should be all-caps. These three workspace test files
use Pascal-case Svd instead. The generated test class names (floatSvdThinValuesWorkspaceTests
vs floatSVDWorkspaceTests) are visibly inconsistent in the test runner.

**Suggested fix:** Rename files and classes to use SVD (e.g.,
SVDThinValuesWorkspaceTests.fProxy.cs / fProxySVDThinValuesWorkspaceTests).

---

### 6. LOW -- Unused using directives in Pivot and Consts files

**Files:**
- TemplateSource/Consts.cs:1-3: using System.Collections, using System.Collections.Generic,
  using UnityEngine -- none of these namespaces are referenced in the file body.
- TemplateSource/Pivot/Pivot.Operations.cs:1-2,8: using System.Collections,
  using System.Collections.Generic, using UnityEngine -- none referenced.
- TemplateSource/Pivot/Pivot.cs:1-2: using System.Collections,
  using System.Collections.Generic -- not referenced (UnsafeList is from
  Unity.Collections.LowLevel.Unsafe, not System.Collections.Generic).

These survive into the generated public package as dead imports.

**Suggested fix:** Remove the unused using directives.

---

### 7. LOW -- Pivot.Print() vestigial method with no separators or metadata

**File:** TemplateSource/Pivot/Pivot.cs:117-125

This method concatenates all pivot indices with no separator (outputs 20143 instead of
2 0 1 4 3), has no summary metadata (size, sign), and does not match the library
Print.Log / ToFixedString pattern. The same file already has a proper ToFixedString()
method that produces Pivot[N=5, sign=+1]: (2 0 1 4 3).

**Suggested fix:** Delete Print() in favor of Print.Log calling ToFixedString(), or at
minimum add separators and mark it as the legacy path. It is public API surface.

---

### 8. LOW -- Three fully-qualified Unity.Collections.Allocator.Temp among ~710 short-form uses

**Files:**
- TemplateSource/Analysis/Analysis.fProxy.cs:238
- TemplateSource/Interfaces/LinearOperator.fProxy.cs:210-211

Every other file in the codebase writes Allocator.Temp (via using Unity.Collections).
These three occurrences use the fully-qualified Unity.Collections.Allocator.Temp, a minor
formatting inconsistency in the generated output.

**Suggested fix:** Change to Allocator.Temp (the files already have using Unity.Collections).

---

### 9. LOW -- region usage inconsistent across sibling type files

Regions appear in bool, fProxy, iProxy operator/comparator files, Arena files,
UnsafeBoolOP, and StatsCore -- but no OP algorithm files use them. The regions are confined
to the data-type/comparator layer and a few Arena files, making the pattern look half-applied.
Not wrong, but a stranger reviewing the generated package would notice the inconsistency.

**Suggested fix:** Either remove all regions or add them consistently. Low priority -- taste only.

---

### 10. LOW -- fProxyPivotTests.cs at TemplateSourceTests root instead of fProxy/ subdirectory

**File:** TemplateSourceTests/fProxyPivotTests.cs

Every other fProxy-prefixed test file lives in the fProxy/ subdirectory. This one sits at
the root alongside Bool*Tests.cs files. Not visible in the public package (tests only).

**Suggested fix:** Move to TemplateSourceTests/fProxy/fProxyPivotTests.cs.

---

### 11. LOW -- OperationsTest / InitTest singular suffix vs *Tests plural convention

**Files:**
- TemplateSourceTests/fProxy/OperationsTest.fProxy.cs (class fProxyOperationsTest)
- TemplateSourceTests/iProxy/OperationsTest.iProxy.cs (class iProxyOperationsTest)
- TemplateSourceTests/BoolOperationsTest.cs (class BoolOperationsTest)
- TemplateSourceTests/fProxy/InitTest.fProxy.cs (class fProxyInitTest)
- TemplateSourceTests/iProxy/InitTest.iProxy.cs (class iProxyInitTest)

The rest of the test suite (>120 files) uses the plural *Tests suffix. These five use the
singular *Test. Consistent within their own cohort but inconsistent with the majority. Test
files only, not in the public package.

**Suggested fix:** Rename to *Tests for consistency with the rest of the suite.

---

### 12. LOW -- Triple blank lines in three template files

**Files:**
- TemplateSource/OP/UnsafeOP.fProxy.cs
- TemplateSource/OP/UnsafeOP.iProxy.cs
- TemplateSource/Analysis/BoolAnalysis.cs

Three or more consecutive blank lines in these files. Survives into generated output. Minor.

**Suggested fix:** Collapse to at most one blank line between members.

---

## Areas confirmed clean

- **Naming grid (decomp/decompInPlace/decompSolve/solveInPlace):** Consistent across all
  direct solvers (CHO, LU, QR, QRCP, LQ, LQRP, CHOP). No method mutates without the
  InPlace/Inpl suffix where siblings have it.
- **_OP suffix purge:** Complete. No _OP suffix on non-data types. CHOP is correctly
  the Cholesky-Pivoted abbreviation, not a suffix.
- **Elem / Linear / BSM purge:** Complete in code identifiers. Only one stale BSM in
  an XML doc comment (finding #2).
- **Workspace struct naming:** Uniformly fProxyAlgoCache across all 21 workspace structs.
- **TODO/FIXME/HACK comments:** Zero in templates. Clean.
- **Tabs vs spaces:** All files use spaces. No tab contamination.
- **Type-casing (QR, LU, SVD, FFT, etc.):** Consistent with the style guide all-caps rule
  for literature acronyms, except the three test files noted in finding #5.
- **Arena naming (Vec/Mat abbreviations):** Consistent throughout.
- **Internal-vs-public visibility:** Kernel classes (UnsafeOP, UnsafeBoolOP, etc.) are
  correctly in LinearAlgebra.Internal. Core helpers (StatsCore, HistogramCore, etc.)
  are correctly internal. No accidental public leaks found.
- **Class-echo rule:** KMeans.fit (not kmeans), FFT.fft (echo exception), LOBPCG.lobpcg
  (echo exception). All correct.
- **Predicate casing:** isSymmetric, isDiagonal, isZero, whichTrue -- lowercase
  camelCase throughout, no Pascal Is found.
- **Exception types:** Only ArgumentException and ArgumentOutOfRangeException found; no
  bare System.Exception, no custom types.
- **Commented-out code:** None found.
- **Debug prints in production code:** Debug.Log only appears in the Debug/ folder (which
  IS the print feature) and Pivot.Print() (finding #7). No stray debug prints in algorithm
  templates.

---

## Summary table

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 5     |
| LOW      | 7     |
| **Total**| **12**|