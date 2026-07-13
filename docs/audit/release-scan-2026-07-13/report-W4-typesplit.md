# W4 -- Type-split correctness / choose gating

Scanner: W4 (type-split correctness)
Date: 2026-07-13
Scope: all 236 templates in TemplateSource/, plus TemplateSourceTests/ and TemplateSourceBenchmarks/

---

## Methodology

1. Read TemplateConverter.cs and GenUtils.cs to learn the exact codegen mechanics.
2. Searched every choose block for correct branch count vs. the file expansion set.
3. Searched every skipFor block for correct pairing and tag coverage.
4. Searched all fProxy templates for hardcoded epsilon/tolerance literals that should differ between float and double.
5. Searched all iProxy templates for inherently-real operations that might leak into integer types.
6. Searched for float-suffixed literals in fProxy and iProxy templates.
7. Searched for hardcoded type names where proxy tokens were expected.

---

## Findings

### 1. MEDIUM -- const float type survives into double variant
**File:** Sparse/Gallery.Sparse.fProxy.cs:25
**Line:** const float fProxySparseOffScale = 0.3f;
**Issue:** The type float is a literal C# keyword, not a proxy token, so it persists into the
double-generated variant as const float doubleSparseOffScale = 0.3f. The generated
doubleGallery partial class carries a float constant in a double context. Functionally
harmless because the constant is only used as a bound for rng.NextFloat which always
returns float anyway, and in (fProxy)fProxySparseOffScale * BR where float is promoted
to double. But the generated code has a gratuitous float constant in a double class.
**Fix direction:** Change to const fProxy fProxySparseOffScale = (fProxy)0.3; or use a choose block.

### 2. LOW -- Float-suffixed literals (0f, 1f) in fProxy templates
**Files and representative lines:**
- Analysis/Analysis.fProxy.cs:70,79 -- fProxy maxError = 0f;
- Analysis/Analysis.fProxy.cs:96,99,114,159,187,215 -- != 1f, != 0f, - 1f
- Statistics/StatsCore.fProxy.cs -- many 0f, 1f, 2f literals (25+ occurrences)
- OP/UnsafeOP.fProxy.cs -- accumulators initialized = 0f (lines 257-264, 381-388, 1049-1156)
- OP/NormsOP.fProxy.cs -- 0f and 1f in norm kernels (lines 114-177)
- OP/UnsafeMathOP.fProxy.cs -- 0f, 2f, 1f literals (lines 357, 369, 376-378)
- Arena/ArenaExtensions.fProxy.cs:39,53 -- 1f, -1f
- Debug/Debug.fProxy.cs:88 -- Spy(m, 0.01f)
- OP/LOBPCG.fProxy.cs:114, OP/SVD.LowRank.fProxy.cs:113 -- 2f - 1f

**Issue:** In the double variant these become double var = 0f; where the float literal 0f
is implicitly widened to 0.0. For exact values (0, 1, 2) there is zero precision impact.
For 0.01f (Debug.Spy default), the double variant receives 0.009999999776 instead of 0.01,
which is cosmetically different but irrelevant for a visualization threshold. The rest of
the codebase consistently uses (fProxy)0, (fProxy)1 style. Purely a style inconsistency.
**Fix direction:** Replace 0f with (fProxy)0, 1f with (fProxy)1, etc. throughout.

### 3. LOW -- Float literal 0f in iProxy division-by-zero guards
**Files:**
- iProxy/iProxyN.Operators.cs:78,101 -- if (s == 0f)
- iProxy/iProxyMxN.Operators.cs:76,96 -- if (s == 0f)

**Issue:** These files have alsoExpand[uint]. For the uint variant, uint == 0f involves
implicit uint-to-float promotion. The comparison is correct (only 0u maps to 0.0f), but
using a float literal for an integer zero check is unexpected. Siblings use (iProxy)0 or
plain 0 elsewhere.
**Fix direction:** Replace 0f with (iProxy)0 or 0.

### 4. LOW -- (fProxy)1.01f uses inexact float literal as safety margin
**File:** OP/SVD.LowRank.fProxy.cs:182,289
**Line:** anorm = math.max(anorm, (fProxy)1.01f * svdAnormBlock(...));
**Issue:** 1.01f is not exactly representable in float (it is 1.0099999904632568 as double).
In the double variant, (double)1.01f gives approximately 1.00999999 instead of 1.01. Since
this is a safety margin multiplier (anorm estimate, not an equality test), the difference
is negligible.
**Fix direction:** Use (fProxy)1.01 (double literal cast to fProxy) instead of (fProxy)1.01f.

---

## Areas confirmed clean

- **All choose blocks** have correct branch counts: 2 values in fProxy files, 3 in iProxy
  without alsoExpand, 4 in iProxy with alsoExpand[uint]. Verified exhaustively.

- **All skipFor blocks** correctly paired (13 open, 13 close). Tags: [u] (5 files),
  [int,short,long] (1 file), [int,long,uint] (2 occurrences).

- **Consts.fProxy* proxy constants** consistently used throughout and correctly expand to
  per-type values in the singular Consts.cs.

- **LP/QP pivot/feasibility tolerance patterns** use math.max with Consts.fProxy* and
  hardcoded floors that correctly adapt per type.

- **LP per-dtype choose blocks** correctly used for precision-dependent thresholds
  (ratioTieTol, near-tie window, lad crossover).

- **fProxy.MaxValue/MinValue/PositiveInfinity/NaN** used throughout instead of hardcoded
  float.MaxValue etc.

- **iProxy.MaxValue/MinValue** only used in files WITHOUT alsoExpand[uint], so unsigned-
  problematic MinValue patterns never generate for uint.

- **No MathF calls, no float.MaxValue/MinValue/Epsilon in fProxy templates.**

- **Test templates** use per-type choose blocks for tolerances matching each generated type.

- **Integer Norms/Stats** returning double are intentional designs, not leaks of real ops.

- **UnsafeBitsOP choose blocks** correctly handle all 4 types with proper short-specific
  width corrections.

- **Export formatting** correctly uses choose for type casts and format strings.

- **No asymmetric fProxy coverage**: all fProxy templates generate for both float and double.

---

## Summary table

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 1     |
| LOW      | 3     |
| OPEN Q   | 0     |
