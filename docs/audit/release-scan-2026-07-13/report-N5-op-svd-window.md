# Release scan 2026-07-13 — N5 narrow pass: TemplateSource/OP, SVD.Randomized → WindowType (20 files)

Partition (case-insensitive alphabetical): SVD.Randomized.fProxy.cs, SVD.RandomizedWorkspace.fProxy.cs,
SVD.Solvers.fProxy.cs, SVD.Subspace.fProxy.cs, SVD.ThinWorkspace.fProxy.cs, SVD.TruncatedWorkspace.fProxy.cs,
SVD.ValuesWorkspace.fProxy.cs, SVD.Workspace.fProxy.cs, SwapOP.cs, UnsafeBitsOP.iProxy.cs,
UnsafeBoolOP.bool.cs, UnsafeBoolOP.fProxy.cs, UnsafeBoolOP.iProxy.cs, UnsafeMathOP.fProxy.cs,
UnsafeMathOP.iProxy.cs, UnsafeOP.bool.cs, UnsafeOP.fProxy.cs, UnsafeOP.iProxy.cs, Wave.fProxy.cs, WindowType.cs.

Every line read. Cross-checked against TemplateConverter.cs/GenUtils.cs token rules (fProxy expands to
float,double; iProxy to int,short,long plus alsoExpand uint here), the OP DEVLOG, and the in-library
callers of every kernel in the partition (BoolOP.cs, OP.Component.*, NormsOP, Blas.Fused,
Pivot.Operations, SVD facade).

---

## HIGH

### H1 — Scalar-shifted-by-vector bitwise shifts compute in 32-bit int for the long variant
- **File:** Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.iProxy.cs:329-339 (both bitwiseLeftShift(int value, ...) and bitwiseRightShift(int value, ...))
- **Defect:** the value being shifted is declared int for every generated type and the shift is performed in 32-bit arithmetic before widening:
  TargetWShift[i] = (iProxy)(value << (int)TargetWShift[i]);
  In the long variant the shift count is masked mod 32 and the result is truncated to int width, so e.g. Comp.bitwiseLeftShiftInPlace(1, longVec) with an element of 40 stores 256 (1 << (40 & 31)) instead of 2^40 = 1099511627776 — silently wrong on a supported generated type. The public wrapper hardcodes the same int valueToBeShifted (OP.Component.iProxy.cs:213-230, N3 partition), so the 32-bit narrowing is baked into the long public surface too.
- **Fix direction:** type the shifted operand as iProxy and perform the shift at the proxy's own width (a choose split if the per-type shift-count masking semantics need pinning); mirror the fix in the OP.Component wrapper's parameter type.

---

## MEDIUM

### M1 — UnsafeBoolOP bool kernels declare [NoAlias] but the library's own primary callers alias the pointers
- **File:** Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeBoolOP.bool.cs:11-78 (kernels), violated at OP/BoolOP.cs:20-100 (caller, N1 partition)
- **Defect:** every kernel marks all pointer params [NoAlias], yet BoolOP's in-place wrappers pass the SAME pointer as input and target, e.g. UnsafeBoolOP.not(a.Data.Ptr, a.Data.Ptr, ...), or(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, ...) — a formal Burst aliasing-contract lie (addendum pattern 4). Benign today only because these are perfect-overlap same-index maps, but it is exactly the class of UB a Burst upgrade is licensed to exploit.
- **Fix direction:** drop [NoAlias] from the (input, target) pair on these kernels, or give BoolOP dedicated single-pointer in-place kernels.

### M2 — pinvSolve zero-alloc overload's XML doc claims it allocates from the arena
- **File:** Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Solvers.fProxy.cs:19 (doc attached to the caller-scratch primitive at :29)
- **Defect:** the summary on the caller-provided-scratch (zero-alloc) pinvSolve overload states "Allocates temporaries from A's arena via fProxyTempVec/fProxyTempMat (not an InPlace op)" — false for this overload, whose entire purpose is zero-alloc; only the non-XML // line below it (:27-28, invisible in IntelliSense) says "zero-alloc". The allocation behavior belongs on the wrapper at :152.
- **Fix direction:** move the allocation sentence to the allocating wrapper's summary; state "no allocation; scratch is caller-provided" on the primitive.

### M3 — maxSweeps vs maxIterations straggler inside the SVD facade
- **File:** Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Solvers.fProxy.cs:30,153,176,209 (etc.) vs SVD.fProxy.cs:26,166 and SVD.Randomized.fProxy.cs:51
- **Defect:** pinvSolve/pseudoInverse expose the public parameter maxSweeps while thin/values/randomized in the SAME class call the identical budget (forwarded directly into thin's maxIterations, defaulted from the same Consts.sweepBudget) maxIterations — a rename straggler after the settled maxIterations/tolerance standardization (addendum pattern 2). ("Sweeps" is legitimately kept for Eigen's Jacobi, but here the two names label the same forwarded value.)
- **Fix direction:** rename the pinvSolve/pseudoInverse parameter (and its doc mentions) to maxIterations.

### M4 — Unguarded division by a possibly-zero norm in the normalize kernels (and project)
- **File:** Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.fProxy.cs:1047-1162 (normalizeL2InPlace/L1/LMax/LP, all overloads); UnsafeMathOP.fProxy.cs:385-392 (project divides by dot(b,b))
- **Defect:** a zero vector (or zero sub-range) yields x/0 producing a NaN-filled buffer; no guard in the kernel and no nonzero-input contract stated on the public Norms.normalize* / project wrappers (NormsOP.fProxy.cs:49-58, 249-266 carry no such doc).
- **Fix direction:** either document "input must be nonzero" as the contract on the public facade, or early-return when the norm is exactly 0 (structural zero, no tolerance needed).

### M5 — compMul kernel parameter order reversed vs its comp* siblings (the root of the known mulInPlace role-swap)
- **File:** Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.iProxy.cs:230 and UnsafeOP.fProxy.cs:983
- **Defect:** compMul([NoAlias] fProxy* from, [NoAlias] fProxy* target, int n) puts the mutated operand SECOND, while compAdd/compSub (:765/:962, iProxy :193/:200) put it FIRST and compDiv/compMod name it targetDividend first. This is the divergence that produced the wide-pass HIGH on the mulInPlace(this T from, T to) wrapper (receiver not mutated).
- **Fix direction:** flip compMul to target-first in the same change that fixes the OP.Component wrapper, so kernel and wrapper conventions agree.

---

## LOW

### L1 — Swap.Vec misuses the ArgumentOutOfRangeException(paramName) ctor; guard style differs from Rows/Columns
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SwapOP.cs:18,22 — throw new ArgumentOutOfRangeException("i and j must be bounded inside vector dimensions") puts the message into ParamName (renders "...Parameter name: i and j must be bounded..."); siblings Rows/Columns (:38,:60) throw ArgumentException for the same condition class.
- Fix: use ArgumentException (matching Rows/Columns) or the two-arg ctor.

### L2 — "more rows than columns" message wrong for the allowed m == n case
- SVD.RandomizedWorkspace.fProxy.cs:17, SVD.Subspace.fProxy.cs:44,131 — guard is m < n (m >= n allowed) but the message parenthetical says "(more rows than columns)". Say "at least as many rows as columns".

### L3 — Grammar typos in workspace-allocator docs
- SVD.ThinWorkspace.fProxy.cs:45 "Allocates an thin workspace"; SVD.ValuesWorkspace.fProxy.cs:36 "Allocates an values workspace".

### L4 — Workspace-guard messages prefixed "SVD:" instead of the op name
- SVD.ThinWorkspace.fProxy.cs:17, SVD.ValuesWorkspace.fProxy.cs:13 — every other Require* in the partition prefixes the op ("randomized:", "pinvSolve:", the who param in RequireSvdTruncatedWorkspace); these two say just "SVD:".

### L5 — Perf-verdict comment in sincos (comment policy)
- UnsafeMathOP.fProxy.cs:397 — "// more cache efficient than calling sin&cos at same time and writing to both arrays" is a perf verdict, not a contract.
- Proposed DEVLOG entry: under "## UnsafeMathOP": "- 2026-07-13 | sincos: two separate sin/cos passes measured more cache-efficient than one fused pass writing both outputs. (was UnsafeMathOP.fProxy.cs:397)"

### L6 — Orphaned codegen-justification comment for the default sketch seed
- SVD.Randomized.fProxy.cs:28-30 — "Default sketch seed (golden-ratio constant). Inlined rather than a const field because ... (CS0102 duplicate)." is a codegen-decision justification, physically detached from the three 0x9E3779B1u literals it explains (:63,:108,:148).
- Proposed DEVLOG entry: under "## SVD.Randomized": "- 2026-07-13 | Default sketch seed 0x9E3779B1u is inlined at each use: a const field would be emitted into both the float and double generated partials of class SVD (CS0102). (was SVD.Randomized.fProxy.cs:28)"

### L7 — rol/ror comment references the skipFor template mechanism in shipped generated code
- UnsafeBitsOP.iProxy.cs:83-89 — the comment phrase "skipFor'd away for every other generated type" ships verbatim into the generated int/long/uint files, where the marker and the shiftMod16 block it explains were stripped — template-machinery jargon in package output.
- Fix: reword to describe behavior per type without naming the marker (or move the marker-specific half into the alsoExpand-stripped header block).

### L8 — normalize kernel naming/return drift among siblings
- UnsafeOP.fProxy.cs:1047,1077,1105,1133 — normalizeL2InPlace carries the InPlace suffix; equally-mutating normalizeL1/normalizeLMax/normalizeLP don't. Also the whole-buffer L2 overload (:1047) returns void while every sibling (and L2's own range overload) returns the pre-normalization norm. Internal API only, but the public Norms facade inherits the missing-InPlace naming (NormsOP, N3 partition — flagged here for cross-reference: Norms.normalizeL2(in x) overwrites x with no InPlace suffix; addendum pattern 3).

### L9 — Float-suffixed literals in fProxy templates that also generate double (benign, exact values)
- UnsafeOP.fProxy.cs:257-264,381-388 (matMatDot/matMatDotTransA "= 0f" c-locals), :1049,1063,1079 etc. ("fProxy sum = 0f"); UnsafeMathOP.fProxy.cs:357,369,376-378 (0f/1f/2f in dot/reflect/refract). All compile and are exact (0/1/2), so no numeric defect — but the same files use (fProxy)0-style casts elsewhere; style drift only (addendum pattern 6; checked: no inexact literal survives into the double variant anywhere in the partition).

### L10 — Fallback GEMM ranges index with int arithmetic while the tiled bulk casts (long)
- UnsafeOP.fProxy.cs:350-353,471-474 (matA[r * n + nCols], matC[r * k + kCols]) vs the tiled paths' (long)(i + t) * n — inconsistent overflow discipline; only matters for matrices with more than 2^31 elements, but the tiled code went out of its way and the fallback silently didn't.

### L11 — PascalCase parameter TargetWShift
- UnsafeOP.iProxy.cs:330,336 — parameter name TargetWShift (PascalCase) is a naming outlier vs every other camelCase param in the file. (Same lines as H1; fix together.)

### L12 — Norms L2 passes the same pointer to both [NoAlias] params of vecDot/vecDotRange
- Kernel UnsafeOP.fProxy.cs:87,108; callers NormsOP.fProxy.cs:45,235 (vecDot(a.Data.Ptr, a.Data.Ptr, ...)). Read-only (no stores), so unlike M1 no reordering can change results — formal contract violation only. Fix: a dedicated normSq kernel or drop [NoAlias] from one param.

---

## Open questions for maintainer

None. No inherently-real ops leak into int/uint variants in this partition; the iProxy files gate abs/relu/signFlip/sumAbs/maxAbs away from uint correctly via the unsigned skip tag, and the uint alsoExpand set is bit-op/comparator-only as documented.

## Areas confirmed clean

- SVD workspace structs vs Require* guards vs Arena allocators: field-by-field size agreement verified for fProxySVDRandomizedCache, fProxySVDThinCache, fProxySVDTruncatedCache (both p conventions), fProxySVDValuesCache, fProxySVDCache (At default-when-tall documented and matched by pinvSolve's wide-only guard).
- SVD.Solvers algebra: tall/wide branches of pinvSolve (vector + multi-RHS) and pseudoInverse verified against A = U S V^T / A^T = U S W^T index-by-index (loop bounds j<n tall, j<m wide match S length k); NotConverged zeroing matches docs; n==0/m==0/S[0]==0 short-circuits are safe (no S[0] out-of-bounds read).
- SVD.Randomized HMT pipeline: Blas.dot transpose-flag usage, subspace iteration, B = Q^T A via thin(B^T) back-mapping (U = Q*Vp, V = Up), k <= l guarantee, seed==0 sentinel — all correct; convenience-overload docs match the forwarded defaults exactly.
- UnsafeBitsOP short-width corrections: countbits/tzcnt/lzcnt/reversebits/rol/ror/ceilpow2 choose-branches verified per type including edge cases (rotate shift 0 and [1,15]: carry-free adds proven disjoint-bit; ceilpow2(0)->0, (1)->1; boundary 0x4000/0x4001 overflow quirks are parity with the int container's own behavior); "+" instead of "|" inside choose branches is a parser constraint, correctly applied throughout.
- UnsafeOP.fProxy SIMD reductions and WY/Francis/Jacobi/SYRK/TRSM kernels: accumulator/fold orders match the frozen determinism contract everywhere; matMatDot/matMatDotTransA remainder routing arguments verified against the Range signatures (no off-by-one, no seam); formT recursion matches LARFT (tau folded to 1); wyTriTransMul/wyTriMul iteration directions are correct for their in-place data dependencies; syrkLowerSub/syrkUpperSub disjoint-region [NoAlias] justifications check out; heapsort (sortByKeyAscending/SiftDown) correct.
- UnsafeBoolOP comparator kernels (fProxy/iProxy): exact float ==/!= is intentional (elementwise operator mirror); uint alsoExpand safe (relational only); ispow2 per-type choose verified (short via int promotion, long via widened formula, int/uint native).
- Wave functors + WindowType: formulas match their XML contracts (Saw is [-1,1) with -1 at boundaries, Triangle peaks at frac 0.5, Square duty semantics); Cycles==0/Duty==0 exact-equality sentinels are documented API; PI literal is per-type precision-correct.
- Swap facade being float/double-only while iProxy/bool swap kernels exist is not accidental: integer/bool row/col swaps are served through Pivot.Operations (copyReplaceAll), which is the only int/bool caller of those kernels.
- DEVLOG.md present; already carries the UnsafeOP perf history; no code comment in the partition duplicates a DEVLOG entry.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 1     |
| MEDIUM   | 5     |
| LOW      | 12    |

Files with no findings at all: SVD.Workspace.fProxy.cs, UnsafeBoolOP.fProxy.cs, UnsafeBoolOP.iProxy.cs, UnsafeMathOP.iProxy.cs, UnsafeOP.bool.cs, Wave.fProxy.cs, WindowType.cs.
