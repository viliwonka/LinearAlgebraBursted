# W5 — Logic errors (wide scan, templates only)

Scanner: W5. Dimension: logic bugs (indexing, copy-paste divergence between siblings,
in-place aliasing, pivot/permutation direction, early exits, benchmark timing).
Method: read TemplateConverter.cs token rules first; deep-read of the highest-risk
kernels and the newest (least-battle-tested) features; targeted pattern sweeps
(`* M_Rows +` misindexing, row/col loop-bound swaps, matVecDot/vecMatDot/matTrans
argument order, same-pointer calls into `[NoAlias]` kernels) across all 398 template
files. All indexing checked mentally against rectangular (M≠N) shapes.

---

## Findings

### HIGH

**H1. `mulInPlace(T, T)` mutates its ARGUMENT, not the receiver — inverted vs every sibling**
- Template: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Component.fProxy.cs:122` and
  `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Component.iProxy.cs:108`
- Defect: `public static void mulInPlace<T>(this T from, T to)` forwards to
  `UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length)`, and `compMul(from, target, n)`
  does `target[i] *= from[i]` — so the extension-method call `a.mulInPlace(b)` computes **b *= a**,
  leaving the receiver `a` unchanged. Every sibling pairwise op mutates the receiver:
  `addInPlace(this T place, T from)` → place += from; `subInPlace`,
  `divInPlace(this T targetDividend, T fromDivisor)`, `modInPlace` likewise. The comment above it
  even claims it is "matching addInPlace/subInPlace's existing pattern".
- Failure scenario: any user writes `x.mulInPlace(y)` expecting x *= y (the pattern every other
  `*InPlace(T,T)` op teaches); result: x is silently unchanged and y is clobbered — wrong results
  with no error, for all 5 generated element types (float/double/int/short/long).
- The internal call sites compensate for the inversion — `fProxyMxN.Operators.cs:141` /
  `fProxyN.Operators.cs:146` (and iProxy twins at `iProxyMxN.Operators.cs:203` /
  `iProxyN.Operators.cs:209`) deliberately pass `mulInPlace(rhs, matrix)` so the `*` operator itself
  is CORRECT — which is exactly why the test suite doesn't catch the public landmine.
- Fix direction: swap the parameter roles so the receiver is the mutated target
  (`mulInPlace(this T place, T from)` → `compMul(from.Ptr, place.Ptr, ...)`) and flip the four
  operator call sites (fProxy/iProxy × MxN/N) to `mulInPlace(matrix, rhs)`.

### MEDIUM

**M1. `signFlipInPlace` passes the same pointer to both `[NoAlias]` parameters of `signFlip`**
- Template: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Component.fProxy.cs:165`,
  `OP/OP.Component.iProxy.cs:158`, and `Sparse/Comp.Sparse.fProxy.cs:23`
- Defect: `UnsafeOP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length)` — but the kernel is declared
  `signFlip([NoAlias] fProxy* target, [NoAlias] fProxy* from, int n)`
  (`OP/UnsafeOP.fProxy.cs:758`, `OP/UnsafeOP.iProxy.cs:185`). Passing an identical pointer for both
  violates the no-alias promise Burst's alias analysis is told to rely on. For this exact
  same-index elementwise map it happens to be benign under any per-index-preserving vectorization,
  but it is undefined by the `[NoAlias]` contract and a compiler upgrade is free to break it.
- Fix direction: add a single-pointer in-place kernel (`signFlipInPlace(fProxy* target, int n)`,
  like `scalMul`) or drop `[NoAlias]` from `signFlip`'s parameters.

**M2. `Blas.dotRows` never validates `rows` against either matrix — out-of-bounds write on caller error**
- Template: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.fProxy.cs:194`
- Defect: `dotRows(in a, in b, ref c, int rows)` MemClears `rows * kk` elements of `c` and runs
  `matMatDot(..., rows, nn, kk)` with no check that `rows <= a.M_Rows` or `rows <= c.M_Rows`
  (or `rows >= 0`); the sibling `dot` overloads validate every dimension.
  `dotRows(A, B, ref C, C.M_Rows + 1)` silently reads past A and writes past C.
- Fix direction: add `rows` range validation mirroring the sibling guards.
  (Overlaps W2's dimension; listed here because the siblings' guard pattern diverges.)

### LOW

**L1. `rowSum`/`rowMean`/`colSum`/`colMean` lack the empty-matrix guard their statistic siblings have**
- Template: `Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsCore.fProxy.cs:263, 284, 304, 317`
- Defect: `rowMin`/`rowMax`/`rowVariance`/`colVariance` throw `InvalidOperationException` on an
  empty matrix; `rowMean`/`colMean` instead divide by zero (`divInPlace(dest, A.N_Cols)` with
  `N_Cols == 0`) and silently fill dest with NaN.
- Fix direction: add the same empty-matrix guard (or document the NaN behavior) so the sibling
  family behaves uniformly.

**L2. `Analysis` predicates loop the column index over `A.M_Rows`**
- Template: `Assets/LinearAlgebra/CodeGen/TemplateSource/Analysis/Analysis.fProxy.cs:92, 111, 129, 143, 157, 171`
  and `Analysis/Analysis.iProxy.cs:38, 57, 71`
- Defect: `for (int c = 0; c < A.M_Rows; c++)` where the loop variable indexes COLUMNS
  (`A[r, c]`). Correct today only because each method first requires `M_Rows == N_Cols`; a future
  edit relaxing that guard turns this into an out-of-bounds read or a silent partial scan. No
  wrong result is currently produced.
- Fix direction: use `A.N_Cols` as the column bound.

---

## Areas confirmed clean (verified, not sampled)

- **UnsafeOP.fProxy.cs / UnsafeOP.iProxy.cs (full read)** — GEMM 8x16 register tiling: tile/remainder
  seams (`mTiles`/`kTiles`), `matMatDotTransA`'s transposed indexing (`matA[p*m + r]`) and both
  remainder-range calls checked argument-by-argument against the `(rowStart, rowEnd, m, n, k,
  colStart, colEnd)` signature; SIMD reduction tails (`nQ<<2`); trsm/syrk panel helpers; compact-WY
  (`wyVtC`/`wySubVW`/`wyTriMul` up/down iteration directions); heapsort. No defects.
- **Blas.Triangular.fProxy.cs (full read)** — all 12 variants (upper/lower × vec/multi-RHS ×
  forward/LU/TransA): substitution directions, pivot-indirected row reads `U[RP[r], c]`,
  unit-diagonal handling, and the right-looking TransA formulations verified against the math.
- **Pivot.cs / Pivot.Operations.cs (full read)** — cycle-following apply verified by hand on a
  3-cycle: `ApplyVec` = scatter (x[P[i]] ← v[i]), `ApplyInverseVec` = gather (Pb). Cross-checked
  against LU.decompSolve (gather before forward solve) and LU.decompSolveTransA (scatter after) —
  both directions correct, including the multi-RHS `ApplyRow`/`ApplyInverseRow` twins.
- **LU.fProxy.cs (full read)** — blocked and unblocked paths of `decomp` and compact
  `decompInPlace`: panel bounds (`kMax = min(panelEnd, m-1)` matching the unblocked `k < m-1`),
  L-row swap ranges `[0,k)`, TRSM/GEMM trailing updates, scattered-row gather in the compact path,
  final-diagonal singularity checks. No defects.
- **OP.Dot.fProxy.cs (full read)** — accumulate-kernels' zero-destination preconditions met at
  every wrapper (`MemClear` before `matVecDot`/`matMatDot`; `vecMatDot` self-zeroing); alias guards
  present; transposeA dimension mapping correct (except dotRows' missing `rows` guard, M2).
- **Bidiag.fProxy.cs (full read)** — rectangular m≥n sweep, left/right reflector column/row ranges,
  backward thin-U reconstruction ordering. No defects.
- **SparseOP.Transpose.fProxy.cs (full read)** — CSR-of-transpose histogram/scatter and per-block
  transpose indexing (`dst[c*BR + r] = src[r*BC + c]`) correct for rectangular blocks.
- **ResampleOP.fProxy.cs + OpHelpers.Shared.cs (full read)** — Catmull-Rom taps, separable 2D
  passes (scratch srcM×dstN, vertical endpoint pin reads `scratch[srcM-1, c]` — correct), and the
  Clamp/Wrap/Mirror `idx` mapping (mirror period 2(n-1), no edge repeat). No defects.
- **HistogramCore.fProxy.cs (full read)** — bin-edge rule (closed upper edge, NaN drop, rounding
  clamp), auto-range seeding, CDF pinning, 2D variant. No defects.
- **StatsCore.fProxy.cs row/col reduction block (read)** — every row*/col* pair diffed; loop bounds
  and dest lengths all correct (only the guard divergence L1).
- **Kalman.fProxy.cs, Kalman.UKF.fProxy.cs (full read)** — Joseph-form update, transposed-system
  gain solves (Kᵀ via CHOP), Q/R joint rescale invariance in steadyStateGain, UKF sigma-point
  permutation scatter (`diff[Piv[i]] = L[i,k]`) verified against the pivoted-Cholesky identity,
  `P -= Pxz·Kᵀ` ≡ K·Pzz·Kᵀ. No defects.
- **MPC.State.fProxy.cs, MPC.fProxy.cs (full read)** — Phi/Gamma block recursions, block-(k−1)
  prestabilization convention, deltaU tridiagonal Hessian scales (2/1) and gradient cross-term,
  warm-start shift/tail-fill/clip/simulate ordering, pre-QP u0 capture for the Fallback contract.
  No defects.
- **NLS.fProxy.cs (full read)** — LM damped-step assembly (augmented rows zeroed each call),
  Nielsen mu/nu updates, predicted-reduction formula re-derived and confirmed
  (0.5·hᵀ(μD²h − g)), post-accept Jacobian/gradient refresh ordering. No defects.
- **Benchmark templates** — pattern check across the 27 files plus a full read of
  TriangularSolveBenchmark.fProxy.cs: factorization runs outside `Bench.Time`, timed jobs re-copy
  their destroyed inputs inside the job by documented design, results live in native buffers
  (no dead-code-elimination risk). No timing-the-wrong-thing instances found.
- **SelectOP, SwapOP, GenOP, Blas.ColumnScaling, fProxyMxN Indexing/Operators/Shortcuts** — clean.
- Global sweeps: zero hits for `i*M_Rows+j`-style misindexing anywhere in TemplateSource; the only
  `r < N_Cols` / `c < M_Rows` loop-bound swaps are QR.fProxy.cs:644 (intentional: copies the first
  N_Cols of b into x) and the guarded Analysis predicates (L2).

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| HIGH     | 1     | H1 (mulInPlace mutates argument, not receiver) |
| MEDIUM   | 2     | M1 (signFlipInPlace violates [NoAlias]), M2 (dotRows missing rows guard) |
| LOW      | 2     | L1 (mean/sum empty-guard divergence), L2 (Analysis column bound uses M_Rows) |