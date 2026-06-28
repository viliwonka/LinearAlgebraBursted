# Performance Audit — Unity Burst Linear Algebra Library

**Date:** 2026-06-28  
**Reviewer:** Claude Sonnet 4.6  
**Scope:** `Assets/LinearAlgebra/CodeGen/TemplateSource/` (template source of truth)  
**Method:** Systematic read of every numerical kernel (OP/, ML/, Realtime/, Arena/, Statistics/, Interfaces/)

---

## Executive Summary

The library is broadly well-structured for Burst. There are no managed allocations, no LINQ, no boxing. The `[NoAlias]` annotations on `UnsafeOP` pointer parameters are correct and help Burst's alias analysis. The `matMatDot` kernel uses the IKJ loop order (cache-friendly, inner loop Burst-vectorizable). Most ops expose a zero-alloc ref-dest primitive alongside an allocating wrapper.

The five most impactful issues are all cache/SIMD problems, not algorithm errors:

1. **`vecMatDot` inner loop is column-strided (stride-N)** — prevents cache-line reuse and Burst vectorization; one loop-order swap fixes it.
2. **`covarianceInto` walks O(N²) matrix column-pairs** — each pair does two stride-N column reads over M rows; centering into a scratch and using `matMatDotTransA` would give the same result with row-major access.
3. **QR Householder application reads/writes columns with stride-N_Cols** — for tall matrices (m >> n) this is the dominant cost.
4. **SVD Jacobi fused alpha/beta/gamma inner loop reads two columns stride-N** — same structural problem as QR.
5. **Jacobi eigendecomposition has a branch inside its hot inner loop** — suppresses Burst vectorization; two-element special case can be handled outside.

Everything else is LOW priority (minor micro-nits or deliberate design choices). The Arena temp-pool pattern is intentional, but callers in per-frame loops need awareness of when to call `ClearTemp`.

---

## Section 1 — Cache Locality and Access Patterns

### 1.1 `vecMatDot` — column-strided inner loop

**File:** `OP/UnsafeOP.fProxy.cs` lines 102–115  
**Severity: HIGH**

```csharp
public static void vecMatDot(fProxy* y, fProxy* mat, fProxy* x, int m, int n)
{
    for (int c = 0; c < n; c++)            // outer: column index
    for (int r = 0; r < m; r++)            // inner: row index
        x[c] += y[r] * mat[r * n + c];    // stride-n read of mat
}
```

With `c` fixed and `r` varying, `mat[r * n + c]` steps by `n` elements (one full row) per iteration — every access is a potential cache miss for matrices wider than a cache line. Burst cannot auto-vectorize a gather.

**Fix:** swap the loop order. The rewritten version has unit-stride reads on both `mat` and write-back on `x[c]`, which Burst will vectorize:

```csharp
for (int r = 0; r < m; r++) {
    fProxy yr = y[r];
    for (int c = 0; c < n; c++)
        x[c] += yr * mat[r * n + c];   // row-major, unit-stride
}
```

Estimated impact: 2–5× on large (64×+) matrices; free for small sizes.

---

### 1.2 `covarianceInto` — O(N²) column-pair stride-N reads

**File:** `Statistics/StatsOP.fProxy.cs` lines 633–665  
**Severity: HIGH**

```csharp
for (int i = 0; i < N; i++)
for (int j = i; j < N; j++) {
    fProxy acc = 0f;
    for (int r = 0; r < M; r++)
        acc += (A[r, i] - means[i]) * (A[r, j] - means[j]);  // two column reads, stride-N
    C[i, j] = C[j, i] = acc * invDenom;
}
```

For each of the O(N²/2) column pairs (i,j), the inner loop walks M rows of column i and column j — both accessed at stride N_Cols in a row-major matrix. For N=50 variables and M=1000 observations, this is 1,250 pairs each doing two 1000-element stride-50 column reads.

**Better algorithm:**
1. Center data into a scratch matrix `Xc[r, c] = A[r, c] - means[c]` (one O(M×N) row-major pass).
2. Compute `C = Xc^T · Xc / (M-1)` using the already-optimised `matMatDotTransA` kernel (IKJ-ordered, cache-friendly).

Same flop count, radically better cache utilization. The existing `matMatDotTransA` (UnsafeOP.fProxy.cs:152–168) handles exactly this shape.

Note: `covarianceInto` also allocates `means` via `A.tempfProxyVec(N)` (arena temp pool, not function-local Allocator.Temp). Contrast with `colVariance` (line 481) which correctly uses `new fProxyN(n, Allocator.Temp)` + `Dispose`. If `covarianceInto` is called repeatedly before `ClearTemp`, the means vectors accumulate.

---

### 1.3 QR Householder application — column-strided reads and writes

**File:** `OP/OrthoOP.fProxy.cs` lines 112–134  
**Severity: MED**

```csharp
for (int c = d; c < Q.N_Cols; c++) {
    fProxy dotProduct = 0;
    for (int k = d; k < Q.M_Rows; k++)
        dotProduct += u[k] * Q[k, c];   // stride-N_Cols column read
    for (int r = d; r < Q.M_Rows; r++)
        Q[r, c] -= u[r] * dotProduct;   // stride-N_Cols column write
}
```

`Q[k, c]` with fixed `c` and varying `k` = `Data[k * N_Cols + c]`, stepping by `N_Cols` elements. For m=200, n=10, each of the n Householder steps reads/writes ~200-element columns at stride 10.

**Alternative:** Accumulate into a temporary row-vector `v[c] = u dot col_c`, then apply update in one row-major pass. Equivalent work, row-major access:

```csharp
// dot products: for each row k, accumulate u[k]*Q[k,c] into dp[c]
float* dp = stackalloc float[N_Cols - d];  // or a scratch vector
for (int k = d; k < M; k++) {
    fProxy uk = u[k];
    for (int c = d; c < N_Cols; c++)       // row-major, contiguous
        dp[c - d] += uk * Q[k, c];
}
// rank-1 update: Q[r,c] -= u[r]*dp[c]
for (int r = d; r < M; r++) {
    fProxy ur = u[r];
    for (int c = d; c < N_Cols; c++)       // row-major, contiguous, Burst-vectorizable
        Q[r, c] -= ur * dp[c - d];
}
```

Same floating-point result; the inner loop is now a row-wise AXPY which Burst will vectorize. Impact is real for overdetermined systems (m >> n), which is the common QR use-case.

---

### 1.4 SVD Jacobi — fused alpha/beta/gamma loop reads two columns stride-N

**File:** `OP/SVD.fProxy.cs` lines 75–81  
**Severity: MED**

```csharp
for (int i = 0; i < m; i++) {
    fProxy bip = U[i, p];   // stride-N_Cols column read
    fProxy biq = U[i, q];   // stride-N_Cols column read
    alpha += bip * bip;
    beta  += biq * biq;
    gamma += bip * biq;
}
```

For a 200×20 matrix (20 columns, 20×19/2=190 Jacobi pairs per sweep), each inner loop reads 200 elements from two N-strided columns. With N=20, every pair of reads is in a different cache line.

The same issue applies to the column-rotation loops (lines 108–121): two column reads + two column writes per rotation, all stride-N.

**Note:** This is structurally inherent to one-sided Jacobi operating on a column-major or column-oriented workload on a row-major matrix. For m < ~64, cache misses are cold-start only; the issue matters for m in the hundreds.

**Practical fix:** Store U in column-major order internally for the SVD workspace, or copy columns p and q into temporary vectors before the inner loop, reducing strided loads to two O(m) copies followed by contiguous dot products. The rotations become column writes back.

---

### 1.5 `resample2DInto` Pass 2 — column-strided reads of scratch

**File:** `OP/ResampleOP.fProxy.cs` lines 294–313  
**Severity: LOW**

```csharp
for (int c = 0; c < dstN; c++) {          // outer: column
    for (int i = 0; i < dstM; i++) {
        fProxy pos = (fProxy)i * vScale;
        dst[i, c] = sampleColAt(in scratch, c, pos, interp, edge);  // stride-dstN
    }
}
```

`sampleColAt` accesses `scratch[r, col]` with varying `r` at stride `dstN`. For large `dstN`, this creates cache pressure on the vertical pass. A transpose of the scratch matrix before the vertical pass would make both passes fully row-major, at the cost of an extra O(srcM × dstN) buffer. For typical game-sized textures/signals (up to 1024 samples), the current approach is acceptable.

---

### 1.6 `SolveUpperTriangularTransposed` — inherently column-strided

**File:** `OP/Cholesky.fProxy.cs` lines 370–381  
**Severity: LOW (unavoidable)**

```csharp
for (int r = n - 1; r >= 0; r--) {
    for (int c = r + 1; c < n; c++)
        sum += L[c, r] * x[c];   // L[c,r] = Data[c*n+r], stride-n as c varies
```

Reading `L^T` from a lower-triangular matrix without materializing the transpose is inherently stride-n. For the small n typical in linear system solves (< 100), this is a minor concern. Materializing L^T first would trade an O(n²) copy for better cache behavior in the solve, which is only worthwhile for very large n.

---

## Section 2 — SIMD and Vectorization

### 2.1 `vecMatDot` — prevents vectorization (same as 1.1)

The stride-N inner loop cannot be autovectorized by Burst. The loop-swap fix described in §1.1 makes the inner `c` loop unit-stride and reduces to a scaled-vector-add (AXPY), which Burst vectorizes with float4/double2 SIMD automatically.

---

### 2.2 Jacobi eigendecomposition — branch in hot inner loop

**File:** `OP/Eigen.fProxy.cs` lines 282–293  
**Severity: MED**

```csharp
for (int i = 0; i < n; i++) {
    if (i == p || i == q) continue;   // branch prevents vectorization
    fProxy aip = A[i, p];
    fProxy aiq = A[i, q];
    ...
    A[i, p] = newAip;  A[p, i] = newAip;
    A[i, q] = newAiq;  A[q, i] = newAiq;
}
```

The early-exit branch prevents Burst from auto-vectorizing the sweep over all i. For n ≤ ~30 (typical symmetric Jacobi use-cases), this matters little; for n = 100+ it is measurable.

**Fix:** handle `i = p` and `i = q` outside the loop (they are handled separately anyway in the diagonal update on lines 277–280), then remove the guard:

```csharp
// (diagonal A[p,p] and A[q,q] updated above; A[p,q]=A[q,p]=0 set above)
for (int i = 0; i < n; i++) {
    if (i == p || i == q) continue;   // remove: handle separately
    ...
}
// Instead:
// Process i != p, i != q branchlessly over [0,n):
for (int i = 0; i < n; i++) {
    fProxy aip = A[i, p], aiq = A[i, q];
    fProxy newAip = c * aip - s * aiq;
    fProxy newAiq = s * aip + c * aiq;
    A[i, p] = newAip; A[p, i] = newAip;
    A[i, q] = newAiq; A[q, i] = newAiq;
}
// Then fix up the diagonal elements that were overwritten:
A[p, p] = app - t * apq;   // correct the p,p and q,q entries
A[q, q] = aqq + t * apq;
A[p, q] = 0; A[q, p] = 0;
```

The branchless form lets Burst apply SIMD to the row/column updates, at the cost of two extra fixup writes. Valid since the diagonal values were cached before the loop.

---

### 2.3 `math.sincos` not used in Box-Muller

**File:** `OP/RandomOP.fProxy.cs` lines 539–541  
**Severity: LOW (template-constrained)**

```csharp
fProxy sinVal = math.sin(angle);
fProxy cosVal = math.cos(angle);
```

Two separate transcendental calls. `math.sincos(angle, out sin, out cos)` would produce both for the cost of one evaluation. The code comment on line 525 acknowledges this: the `out`-parameter form is unavailable through the type-proxy template mechanism, so this cannot be fixed without code-generation changes.

**Impact:** Halves Box-Muller throughput for the double-proxy expansion (doubles have slower trig). For float, the hardware typically fuses or schedules sin/cos efficiently. Flag for if the template mechanism ever gains sincos support.

---

### 2.4 `normalizeLP` — `math.pow` per element, non-vectorizable

**File:** `OP/UnsafeOP.fProxy.cs` lines 365–393  
**Severity: LOW**

```csharp
for (int i = 0; i < n; i++)
    sum += math.pow(math.abs(target[i]), p);
```

`math.pow` is transcendental and not SIMD-friendly. For common integer exponents (p=1,2,3,4) the call site could dispatch to direct multiplication. Since `p` is a runtime float, no compile-time specialization is possible without an enum dispatch. Acceptable as a niche operation.

---

### 2.5 DFT does not hoist `baseAng * k` outside the inner loop

**File:** `OP/FFT.fProxy.cs` lines 196–210  
**Severity: LOW**

```csharp
for (int k = 0; k < n; k++) {
    for (int t = 0; t < n; t++) {
        fProxy ang = baseAng * (fProxy)k * (fProxy)t;  // 2 multiplies per inner iteration
```

The factor `baseAng * k` is loop-invariant with respect to `t`. Hoisting it saves one FP multiply per inner iteration:

```csharp
for (int k = 0; k < n; k++) {
    fProxy kAng = baseAng * (fProxy)k;   // hoist
    for (int t = 0; t < n; t++) {
        fProxy ang = kAng * (fProxy)t;   // 1 multiply
```

Burst's LICM pass may already do this. Impact is negligible for the O(n²) DFT (the cos/sin cost dominates).

---

## Section 3 — Algorithmic Optimality

### 3.1 Manual zeroing loops should use `UnsafeUtility.MemClear`

**Files and Severity: MED**

Three locations use double-nested assignment loops to zero a matrix:

| Location | Code |
|---|---|
| `Cholesky.fProxy.cs` lines 154–156 | `for(i) for(j) L[i,j] = 0;` |
| `SVD.Solvers.fProxy.cs` lines 220–222 | `for(r) for(c) Aplus[r,c] = 0;` |
| `RandomMatrixOP.fProxy.cs` lines 315–317 | `for(r) for(c) US[r,c] = 0;` |

Each is O(n²) scalar stores. Replace with:

```csharp
unsafe {
    UnsafeUtility.MemClear(L.Data.Ptr,
        (long)L.Data.Length * UnsafeUtility.SizeOf<fProxy>());
}
```

Note: the OP.Dot.fProxy.cs wrappers already use `UnsafeUtility.MemClear` correctly (lines 79, 109, 156). This is an inconsistency rather than a systemic failure.

---

### 3.2 `standardizeRows` does 3 passes over A when 2 suffice

**File:** `Statistics/StatsOP.fProxy.cs` lines 771–785  
**Severity: LOW**

```csharp
rowMean(in A, ref s0);       // 1 pass over A
rowStdDev(in A, ref s1);     // calls rowVariance → 2 passes over A
                              // total: 3 passes
```

A Welford-style single-pass mean+variance computation per row would reduce to 2 passes (one forward pass for mean+variance, one application pass). The savings matter only if called per-frame on large matrices, which is an unusual pattern for a standardization op.

---

### 3.3 `range()` calls `max()` then `min()` — two full passes

**File:** `Statistics/StatsOP.fProxy.cs` lines 174–177  
**Severity: LOW**

```csharp
public static fProxy range<T>(in T x) ...
    => max(x) - min(x);
```

`max` and `min` are separate O(n) passes. The `meanMinMaxRange` function (line 179) already does a single-pass min+max+sum. Callers needing only `range` should prefer `meanMinMaxRange(...).Range` when they also need other stats, or a dedicated single-pass `minMax` helper. Low priority since `range` is clearly a utility function, not a hot path.

---

### 3.4 `Cholesky.choleskyDecompositionPivot` copies A into W with a nested loop

**File:** `OP/Cholesky.fProxy.cs` lines 161–166  
**Severity: LOW**

```csharp
for (int i = 0; i < n; i++)
for (int j = 0; j <= i; j++) {
    fProxy v = A[i, j];
    W[i, j] = v;
    W[j, i] = v;
}
```

The upper-triangle mirror write (`W[j, i] = v`) with outer `i` and inner `j <= i` writes to row `j` with column `i`, which is a scattered write into the upper triangle. This is a minor init-only cost, not repeated in the hot loop.

---

### 3.5 LU inner loop — already optimal

**File:** `OP/LU.fProxy.cs` lines 54–60  
**Severity: POSITIVE**

```csharp
for (int i = k + 1; i < m; i++) {
    U[j, i] -= Ljk * U[k, i];   // inner i is contiguous (row j, row k)
}
```

With `j` fixed and `i` varying, both `U[j, i]` and `U[k, i]` walk contiguous rows. This is the correct `jki` (outer j, middle k, inner i) Gaussian elimination loop order for row-major storage. The inner loop is a row-AXPY; Burst should vectorize it.

---

## Section 4 — Allocations and GC

### 4.1 `covarianceInto` allocates means into the arena temp pool

**File:** `Statistics/StatsOP.fProxy.cs` line 638  
**Severity: MED**

```csharp
// Temp vector for column means (reclaimed by ClearTemp, not persistent).
var means = A.tempfProxyVec(N);
```

`tempfProxyVec` adds the vector to `arena.tempfProxyVectors` — it is **not** disposed when the function returns. Each call to `covarianceInto` (and by extension `covariance`, `correlation`) adds one N-element vector to the arena's temp pool. Callers in warm loops need a `ClearTemp` cadence.

Contrast with `colVariance` (line 481), which correctly uses `new fProxyN(A.N_Cols, Allocator.Temp)` + `means.Dispose()` — a function-local allocation freed on return. `covarianceInto` should use the same pattern.

---

### 4.2 Arena temp accumulation from per-frame convenience wrappers

**Files:** `OP/Solvers.fProxy.cs` lines 234–237; `OP/MatrixMetrics.fProxy.cs` lines 37, 59  
**Severity: MED**

The allocating wrappers for several warm-path operations add to the arena temp pool on every call:

| Operation | Arena temp allocations per call |
|---|---|
| `conjugateGradient` (allocating) | 3 × fProxyN (r, p, Ap) |
| `cond(A)` | 1 × fProxyN + SVD workspace |
| `rank(A)` | 1 × fProxyN + SVD workspace |
| `matrixL2(A)` (spectral norm) | 1 × fProxyN + SVD workspace |

None of these are disposed inside the function; they live until `arena.ClearTemp()`. Callers in per-frame update loops should either:
- Use the zero-alloc primitives (hoisting workspace out of the loop), or
- Call `arena.ClearTemp()` at frame end.

The zero-alloc primitives already exist for CG (`conjugateGradient(A, b, x, r, p, Ap, ...)`) and SVD (`pinvSolve` with workspace). The `cond`/`rank`/`matrixL2` functions don't expose a zero-alloc variant — adding `cond(A, ref svdWorkspace)` overloads would address the warm-path concern.

---

### 4.3 `median` and `meanMinMaxRange_medianIQRstdDevVariance` — per-call sort allocation

**File:** `Statistics/StatsOP.fProxy.cs` lines 154, 223  
**Severity: LOW**

```csharp
var copy = new UnsafeList<fProxy>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
copy.AddRange(x.Data);
copy.Sort();
...
copy.Dispose();
```

Correctly disposed before return. Not a leak. These are `Allocator.Temp` allocations that Unity reclaims after the frame anyway. No concern unless called per-frame on large arrays.

---

### 4.4 Verified zero-alloc primitives

**Severity: POSITIVE**

The following confirm the "zero-alloc" claims on the primitive overloads:

- `fProxyOP.dot(in A, in x, ref result)` — MemClear + UnsafeOP.matVecDot, no `new`.
- `fProxyOP.dot(in a, in b, ref c)` — same pattern.
- `OrthoOP.qrDecomposition(ref Q, ref R, ref u)` — no allocation; the allocating wrapper is clearly separated.
- `SVD.svdDecomposition(ref U, ref S, ref V, ...)` — explicitly documented "Does not allocate."
- `Eigen.powerIteration(...)` — explicitly documented "Does not allocate."
- `Eigen.eigenDecomposition(...)` — explicitly documented "Does not allocate."
- `Solvers.conjugateGradient(A, b, x, r, p, Ap, ...)` — no allocation.
- `fProxyResampleOP.sampleAt/sampleAtInto/resampleInto` — all allocation-free.
- `resample2DInto` — exactly one `Allocator.Temp` scratch (disposed before return); documented.

---

## Section 5 — Memory Management and Leaks

### 5.1 `choleskyPivotSolve` — multiple Temp allocs, no try/finally

**File:** `OP/Cholesky.fProxy.cs` lines 282–353  
**Severity: LOW**

The rank-deficient path allocates up to 4 `Allocator.Temp` buffers (`bt`, `g`, `G`, `GL`). All are disposed on the normal return path. The Burst/Unity constraint is that unmanaged exceptions in jobs are unrecoverable (they halt the job), so a try/finally would be non-functional inside Burst. Outside Burst (managed context), an exception from `choleskyDecomposition`'s inner calls (which are bool-returning, not throwing) or from `choleskySolve` → `SolveLowerTriangular` (which throws if not square — but GL is always square r×r here) is effectively impossible on the normal path. No real leak risk; the defensive concern is theoretical.

---

### 5.2 `W.Dispose()` called correctly on all paths in `choleskyDecompositionPivot`

**File:** `OP/Cholesky.fProxy.cs` lines 196, 213  
**Severity: POSITIVE**

```csharp
if (minDiag < -stopTol) {
    W.Dispose();    // line 196 — early return disposes correctly
    rank = k;
    return false;
}
...
if (!(maxDiag > stopTol)) {
    ...
    W.Dispose();   // line 213 — early return disposes correctly
    rank = k;
    return false;
}
...
W.Dispose();       // normal path — correct
return true;
```

All three exit paths dispose W. No leak.

---

### 5.3 Alias guards are present and correct

**Severity: POSITIVE**

- `dot(A, x, ref result)` guards `result.Data.Ptr == x.Data.Ptr` (line 75).
- `dot(a, b, ref c)` guards `c.Data.Ptr == a.Data.Ptr || c.Data.Ptr == b.Data.Ptr` (line 152).
- `trans(A, ref T)` guards `T.Data.Ptr == A.Data.Ptr` (line 189).
- `rfft` guards `im.Data.Ptr == real.Data.Ptr || im.Data.Ptr == re.Data.Ptr` (line 137).
- `Eigen.powerIteration` guards `v.Data.Ptr == w.Data.Ptr` (line 57).

No aliasing footguns found in the checked primitives.

---

## Section 6 — Burst-Specific

### 6.1 `[BurstCompile]` on two static non-job methods inconsistently

**File:** `OP/UnsafeOP.fProxy.cs` lines 416, 434  
**Severity: LOW (misleading, not harmful)**

```csharp
[MethodImpl(MethodImplOptions.NoInlining)]
[BurstCompile]
public static void swapRows([NoAlias] fProxy* target, ...) { ... }

[MethodImpl(MethodImplOptions.NoInlining)]
[BurstCompile]
public static void swapColumns([NoAlias] fProxy* target, ...) { ... }
```

`[BurstCompile]` on a static method is an entry-point declaration for `BurstCompiler.CompileFunctionPointer<...>`. It has no effect on methods that are simply called from a Burst job — Burst will compile those anyway via inlining or normal call-graph compilation. Only these two of the many `UnsafeOP` methods carry this attribute, with no corresponding `CompileFunctionPointer` usage found. The `[NoInlining]` combined with `[BurstCompile]` suggests the intent was function-pointer use, but if that's not the actual usage this is dead decoration. Remove or complete the function-pointer setup.

---

### 6.2 `UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS` defined but not used for explicit hints

**Severity: LOW (informational)**

Every template file defines `#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS` at the top. This enables the `Unity.Burst.Intrinsics.Loop` API for explicit loop hints (`Loop.ExpectVectorized()`). None of the files actually call these APIs. The define is harmless but should be kept in anticipation of future use, particularly for the core kernels (vecDot, matVecDot, matMatDot) where confirming vectorization would be high-value.

---

### 6.3 `FloatMode.Fast` hazard — already mitigated in Gallery

**Severity: POSITIVE (informational)**

The memory file notes a known footgun: `FloatMode.Fast` enables `math.pow(negativeBase, ...)` → NaN. Gallery code was already fixed (commit noted). No new instances found in the audited template files.

---

## Section 7 — Well-Optimized Areas (Confirmed)

The following patterns are correctly implemented and noted here so they are not re-litigated:

| Component | Positive finding |
|---|---|
| `matMatDot` | IKJ loop order: inner `kCols` loop is row-of-B + row-of-C, both contiguous. Burst-vectorizable. |
| `matVecDot` | Outer r, inner c: `mat[r*n+c]` and `y[r]` read contiguous rows. |
| `vecDot`, `sum`, `sumAbs`, `maxAbs` | Tight single-pass contiguous loops; `[NoAlias]` on pointers; Burst will vectorize. |
| LU elimination inner loop | `jki` order — inner `i` walks a contiguous row. |
| Power iteration matvec | Outer i, inner j: `A[i,j]` is a row-major contiguous row read. |
| KMeans assignment | GEMM-based (`X * C^T`) for O(N·D·k) with the cache-friendly `matMatDot`. |
| Arena temp pool | Batch deallocation via `ClearTemp` avoids per-element GC; no managed heap pressure. |
| Box-Muller spare caching | `hasSpare` field in `fProxyGaussian` halves trig calls over a fill loop. |
| QRCP column-norm recompute | Intentionally exact recompute (not downdating) to avoid catastrophic cancellation near rank deficiency. Correct trade-off for this library's target sizes. |
| `resample2DInto` scratch disposal | Validated before allocation, disposed before return on all paths. |
| All major solver primitives | Properly zero-alloc ref-dest forms exist for CG, QR, SVD, Cholesky, LU solvers. |
| `[NoAlias]` on UnsafeOP pointers | Enables Burst's alias-analysis optimization. Consistently applied. |

---

## Prioritized Findings Table

| # | Severity | File : Lines | Issue | Est. Impact | Suggested Fix |
|---|---|---|---|---|---|
| 1 | **HIGH** | `OP/UnsafeOP.fProxy.cs` : 102–115 | `vecMatDot` outer=cols, inner=rows → stride-N mat access, no SIMD | 2–5× on n≥64 mat-vec | Swap loops: outer r, inner c |
| 2 | **HIGH** | `Statistics/StatsOP.fProxy.cs` : 633–665 | `covarianceInto` O(N²) column-pair stride-N reads | Large for N≥20, M≥200 | Center to scratch + `matMatDotTransA` |
| 3 | **MED** | `OP/OrthoOP.fProxy.cs` : 112–134 | QR Householder application reads/writes columns at stride-N_Cols | 2–4× for tall (m≥4n) systems | Accumulate dot products in a row-major pass (see §1.3) |
| 4 | **MED** | `OP/SVD.fProxy.cs` : 75–81 | SVD Jacobi alpha/beta/gamma inner loop reads two columns stride-N | Real for m≥100 | Copy cols p,q to temp; contiguous inner loops |
| 5 | **MED** | `OP/Eigen.fProxy.cs` : 282–285 | Branch `if (i==p || i==q) continue` in Jacobi hot loop | Suppresses SIMD | Handle p,q outside loop; remove branch |
| 6 | **MED** | `Statistics/StatsOP.fProxy.cs` : 638 | `covarianceInto` uses arena temp for means (not function-local Temp) | Accumulates per call without ClearTemp | Use `new fProxyN(N, Allocator.Temp)` + `Dispose()` like `colVariance` |
| 7 | **MED** | `OP/Solvers.fProxy.cs` : 234–237; `OP/MatrixMetrics.fProxy.cs` : 37, 59 | `conjugateGradient` alloc wrapper, `cond`, `rank`, `matrixL2` add to arena temp every call | Arena bloat in per-frame use | Add zero-alloc workspace overloads for `cond/rank/matrixL2`; document ClearTemp cadence |
| 8 | **MED** | `OP/Cholesky.fProxy.cs` : 154–156; `OP/SVD.Solvers.fProxy.cs` : 220–222; `OP/RandomMatrixOP.fProxy.cs` : 315–317 | Double-loop zeroing instead of MemClear | Negligible for small n; real for n≥50 | `UnsafeUtility.MemClear(ptr, length * sizeof)` |
| 9 | **LOW** | `OP/ResampleOP.fProxy.cs` : 294–313 | `resample2DInto` Pass 2 reads scratch at stride-dstN | Matters only for large (>512) dstN | Transpose scratch before vertical pass (extra alloc) |
| 10 | **LOW** | `OP/RandomOP.fProxy.cs` : 539–541 | `fProxyGaussian` calls `math.sin` + `math.cos` separately | ~50% trig cost overhead in Gaussian fill | Use `math.sincos` when template mechanism allows out-params |
| 11 | **LOW** | `OP/FFT.fProxy.cs` : 202 | DFT doesn't hoist `baseAng * k` outside inner t-loop | 1 FP mul per inner iter; Burst LICM may already catch it | `fProxy kAng = baseAng * k;` before inner loop |
| 12 | **LOW** | `Statistics/StatsOP.fProxy.cs` : 771–785 | `standardizeRows` does 3 passes over A (mean + 2 for variance) | Real for large m×n matrices per-frame | Merge to 2 passes: online mean+variance in one row scan |
| 13 | **LOW** | `OP/UnsafeOP.fProxy.cs` : 416, 434 | `[BurstCompile]` on `swapRows/swapColumns` without `CompileFunctionPointer` usage | Misleading, not harmful | Remove attribute or add the matching function-pointer registration |
| 14 | **LOW** | `OP/Cholesky.fProxy.cs` : 370–381 | `SolveUpperTriangularTransposed` reads L^T with stride-n | Unavoidable without transpose materialization; small n only | Acceptable as-is; document for callers needing multi-RHS |

---

## Appendix — Files Reviewed

```
OP/UnsafeOP.fProxy.cs         — low-level kernels (vecDot, matVecDot, vecMatDot, matMatDot, matTrans, swap*)
OP/OP.Dot.fProxy.cs           — safe wrappers: dot, outerDot, trans
OP/LU.fProxy.cs               — LU with/without pivoting, compact inplace LU, determinant
OP/Cholesky.fProxy.cs         — Cholesky + pivoted Cholesky + solvers
OP/OrthoOP.fProxy.cs          — QR Householder, QRCP, qrDirectSolve, qrcpDirectSolve
OP/SVD.fProxy.cs              — one-sided Jacobi SVD
OP/SVD.Solvers.fProxy.cs      — pinvSolve, pseudoInverse
OP/Eigen.fProxy.cs            — power iteration, symmetric Jacobi, QR eigenvalues (general)
OP/Solvers.fProxy.cs          — triangular solvers, SolveQR, conjugateGradient
OP/MatrixMetrics.fProxy.cs    — trace, cond, rank
OP/NormsOP.fProxy.cs          — L1/L2/Linf, NormalizeRows/Columns, matrixL1/L2/Linf
OP/QueryOP.fProxy.cs          — argMaxAbs, rowArgMin/Max, colArgMin/Max, kNearest, kFarthest, radius search
OP/RandomOP.fProxy.cs         — uniform fill, Box-Muller Gaussian, ICDF samplers, weightedPick
OP/RandomMatrixOP.fProxy.cs   — multivariateNormal, randomOrthogonal, randomSPD, conditioned/ranked matrices
OP/FFT.fProxy.cs              — radix-2 Cooley-Tukey FFT, direct DFT, magnitude/phase/power
OP/ResampleOP.fProxy.cs       — sampleAt, resampleInto, resample2DInto (separable bicubic/bilinear/nearest)
Statistics/StatsOP.fProxy.cs  — sum/mean/variance/stdDev, row*/col* reductions, covariance, correlation, standardize, softmax
ML/KMeans.fProxy.cs           — Lloyd k-means with GEMM assignment + k-means++ seeding
Realtime/RollingWindow.fProxy.cs — ring-buffer sliding window
Arena/Arena.cs                — bump allocator with persistent + temp pools
```
