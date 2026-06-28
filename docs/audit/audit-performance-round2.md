# Performance Audit — Round 2

**Date:** 2026-06-28
**Reviewer:** Claude Sonnet 4.6 (second-pass; random-seed order 2914)
**Scope:** `Assets/LinearAlgebra/CodeGen/TemplateSource/` — template source only
**Traversal order:** ML → FFT/ResampleOP/HistogramOP → QueryOP/Predicate → Realtime → SVD/Eigen → Cholesky/Solvers/OrthoOP → Dot/UnsafeOP/StatsOP

---

## Executive Summary

This second pass discovers **8 net-new performance findings** not present in round 1, corrects one round-1 count error, and refutes one round-1 finding as moot. The highest-priority new issue is an algorithmic redundancy in KMeans: the final-sync block unconditionally repeats the full O(N·D·k) GEMM even on the convergence-break path where it is a guaranteed no-op. The next tier covers four column-strided apply passes in StatsOP (standardizeColumns/centerColumns/rescaleColumns/maxAbsColumns), a column-strided mean in RollingWindow, and a double temp-pool leak in RollingWindow.Covariance. A secondary finding corrects round 1's pass-count for standardizeRows from 3 to 4.

**Round-1 high/medium findings are confirmed and not repeated below.** Cross-references are given where this audit adds nuance.

---

## Round-1 Verification

### Confirmed findings

| R1 # | File:lines | Finding | Status |
|---|---|---|---|
| 1.1 | `UnsafeOP.fProxy.cs`:102–115 | `vecMatDot` outer=col inner=row → stride-N | **Confirmed. Hot.** |
| 1.2 | `StatsOP.fProxy.cs`:633–665 | `covarianceInto` O(N²) column-pair stride-N reads | **Confirmed. Hot for N≥20.** |
| 1.3 | `OrthoOP.fProxy.cs`:112–134 | QR Householder outer=col inner=row → stride-N | **Confirmed.** |
| 1.4 | `SVD.fProxy.cs`:75–81, 108–121 | Jacobi fused-loop + column rotation stride-N | **Confirmed.** |
| 1.5 | `ResampleOP.fProxy.cs`:294–313 | `resample2DInto` pass-2 column-strided reads | **Confirmed LOW.** |
| 1.6 | `Cholesky.fProxy.cs`:370–381 | `SolveUpperTriangularTransposed` stride-n (unavoidable) | **Confirmed LOW.** |
| 2.2 | `Eigen.fProxy.cs`:282–285 | Jacobi `if (i==p||i==q) continue` blocks SIMD | **Confirmed.** |
| 2.3 | `RandomOP.fProxy.cs`:539–543 | Box-Muller: `math.sin`+`math.cos` called separately | **Confirmed LOW.** |
| 3.2 | `StatsOP.fProxy.cs`:771–785 | `standardizeRows` multi-pass | **Confirmed — but count is WRONG; see §NEW-6 below.** |
| 3.3 | `StatsOP.fProxy.cs`:174–177 | `range()` calls max() then min() = two passes | **Confirmed LOW.** |
| 4.1 | `StatsOP.fProxy.cs`:638 | `covarianceInto` adds means to arena temp pool | **Confirmed MED.** |
| 4.2 | `Solvers.fProxy.cs`:234–237 | CG allocating wrapper adds 3 vecs to arena temp | **Confirmed MED.** |

### Refuted / overstated finding

**R1 §2.5 — DFT `baseAng * k` not hoisted (LOW)**

File: `FFT.fProxy.cs`:202. The claim is that `baseAng * (fProxy)k` is not hoisted from the inner `t` loop. In practice, `baseAng` and `k` are both loop-invariant with respect to `t`, and Burst's LICM (loop-invariant code motion) pass handles exactly this case. Even if it were missed, the saved operation is one FP multiply against two transcendental calls (`math.cos` + `math.sin`) per inner iteration — well under 1% of inner-loop cost. **This finding is moot; the DFT is dominated by trig cost, not the multiply.**

---

## NET-NEW Findings

### NEW-1 — KMeans: final-sync GEMM is unconditionally repeated on convergence path

**File:** `ML/KMeans.fProxy.cs` lines 239–258
**Severity: HIGH**

The `kmeans` function always executes a final-sync block regardless of exit path:

```csharp
// FIX 3: Final sync — ...
{
    for (int j = 0; j < k; j++) { ... ws.CentNormSq[j] = s; }  // centroid norms: O(k·D)
    fProxyOP.trans(in centroids, ref ws.Ct);                     // transpose: O(k·D)
    fProxyOP.dot(in X, in ws.Ct, ref ws.Gram);                   // GEMM: O(N·D·k)
    for (int n = 0; n < N; n++)
        for (int j = 0; j < k; j++)
            ws.Gram[n, j] = ws.CentNormSq[j] - (fProxy)2 * ws.Gram[n, j];  // patch: O(N·k)
    fProxyQueryOP.rowArgMin(in ws.Gram, ref assignment);          // argmin: O(N·k)
    ...
}
```

On the **convergence-break path** (`changes == 0`, line 175), the comment at line 234 correctly states: _"Convergence-break path: … the redo is a true no-op (same scores, same argmin, same inertia)."_ However, the code runs it anyway. The full O(N·D·k) GEMM plus O(N·k) patch plus O(N·k) argmin executes a second time even though the Gram matrix was just computed with the identical centroids in the same iteration.

**Impact:** For N=10,000, D=128, k=50 (a typical medium-scale run), the GEMM alone is ~64M FMA operations, duplicated on every converging call. k-means commonly converges in 10–50 iterations and convergence is the hot path.

**Fix:** Track convergence and skip the sync:

```csharp
bool convergedEarly = false;
...
if (changes == 0) { convergedEarly = true; break; }
...
// Final sync — skip if converged (Gram+assignment are already consistent)
if (!convergedEarly)
{
    // re-compute centroid norms, trans, dot, patch, argmin
    ...
}
// Inertia computation is cheap (O(N)) and can always run
fProxy sse = 0;
for (int n = 0; n < N; n++)
    sse += ws.PointNormSq[n] + ws.Gram[n, assignment[n]];
inertia = math.max(sse, (fProxy)0);
```

---

### NEW-2 — Four StatsOP "Columns" apply passes are column-strided

**Files:** `Statistics/StatsOP.fProxy.cs`
**Severity: MED**

The following functions compute per-column statistics in row-major order (correct), then apply results in a column-strided apply pass (incorrect):

| Function | Lines | Column-strided apply |
|---|---|---|
| `standardizeColumns` | 808–812 | `for c: for r: A[r,c] = ...` |
| `centerColumns` | 939–943 | `for c: for r: A[r,c] -= m` |
| `rescaleColumns` | 885–892 | `for c: for r: A[r,c] = lo+...` |
| `maxAbsColumns` | 984–992 | `for c: for r: A[r,c] /= mAbs` |

Example from `centerColumns`:
```csharp
for (int c = 0; c < A.N_Cols; c++)      // outer: column
{
    fProxy m = s0[c];
    for (int r = 0; r < A.M_Rows; r++) A[r, c] -= m;  // A[r,c] with fixed c, varying r = stride-N_Cols
}
```

The _compute_ passes in all four functions are already row-major; only the apply is mis-ordered. The fix is trivial — swap the loop order in the apply:

```csharp
// centerColumns apply FIX (row-major):
for (int r = 0; r < A.M_Rows; r++)
    for (int c = 0; c < A.N_Cols; c++)
        A[r, c] -= s0[c];      // A[r,c] with fixed r, varying c = contiguous row ✓
```

Same restructuring applies to `standardizeColumns` and `rescaleColumns`. For `maxAbsColumns`, both the find-max pass and the apply pass are column-strided; the fix is two separate row-major passes (accumulate per-column max, then divide):

```csharp
// maxAbsColumns FIX:
for (int r = 0; r < A.M_Rows; r++)
    for (int c = 0; c < A.N_Cols; c++)
        colMax[c] = math.max(colMax[c], math.abs(A[r, c]));   // pass 1, row-major
for (int r = 0; r < A.M_Rows; r++)
    for (int c = 0; c < A.N_Cols; c++)
        if (colMax[c] > 0) A[r, c] /= colMax[c];              // pass 2, row-major
```

Note: `softmaxColumns` (`StatsOP.fProxy.cs` lines 1035–1043) is also column-strided but requires per-column state across three inner loops; reordering to row-major requires two temporary vectors (per-column max and expSum) and three row-major passes. This is fixable but less straightforward; flag separately. The other four are trivial reorderings.

**Impact:** For a 500×50 matrix (e.g., rolling covariance input), each column-strided apply reads/writes 50 columns of 500 elements with stride-50. With cache lines holding 16 floats, each element load is a potential cache miss. The row-major reorder is a free speedup.

---

### NEW-3 — RollingWindow.Mean is column-strided

**File:** `Realtime/RollingWindow.fProxy.cs` lines 140–147
**Severity: MED**

```csharp
public void Mean(ref fProxyN dest)
{
    for (int c = 0; c < _features; c++)          // outer: feature / column
    {
        fProxy sum = (fProxy)0;
        for (int i = 0; i < _count; i++)
            sum += _buffer[RingRow(i), c];        // _buffer[row, c] with fixed c, varying row = stride-_features
        dest[c] = sum / (fProxy)_count;
    }
}
```

`_buffer[RingRow(i), c]` = `_buffer.Data[RingRow(i) * _features + c]`. With fixed `c` and varying `i`, this steps by `_features` elements per iteration. For `_features = 64` and `_count = 256` (a typical rolling PCA window), each feature accumulation makes 256 stride-64 reads.

**Fix:** Accumulate in row order (outer=count, inner=feature):

```csharp
// zero dest
for (int c = 0; c < _features; c++) dest[c] = (fProxy)0;
// accumulate row-major
for (int i = 0; i < _count; i++)
{
    int row = RingRow(i);
    for (int c = 0; c < _features; c++)
        dest[c] += _buffer[row, c];   // contiguous row read ✓
}
// divide
for (int c = 0; c < _features; c++) dest[c] /= (fProxy)_count;
```

The allocating `Mean()` wrapper (line 153) calls this and returns the vector to the arena temp pool — that allocation pattern is deliberate and unchanged.

---

### NEW-4 — RollingWindow.Covariance leaks two temps to arena per call

**File:** `Realtime/RollingWindow.fProxy.cs` lines 163–174
**Severity: MED**

```csharp
public void Covariance(ref fProxyMxN dest)
{
    ...
    var m = _buffer.tempfProxyMat(_count, _features);  // → arena.tempfProxyMats: NOT disposed on return
    AsMatrix(ref m);
    fProxyStatsOP.covarianceInto(in m, ref dest);       // → adds means vec to arena.tempfProxyVectors (R1 §4.1)
}
```

Each call to `Covariance()` adds:
1. A `_count × _features` matrix to the arena temp pool (via `tempfProxyMat`).
2. A `_features`-length vector to the arena temp pool (via `covarianceInto` → `tempfProxyVec`, R1 §4.1).

Neither is a function-local `Allocator.Temp`. The documentation comments "Pair with Eigen.eigenDecomposition for realtime PCA / dominant motion" — strongly implying per-frame use — but per-frame use accumulates two growing temp allocations per frame until `ClearTemp`.

**Fix:** Use function-local `Allocator.Temp` for the intermediate matrix (analogous to R1's fix for covarianceInto):

```csharp
public void Covariance(ref fProxyMxN dest)
{
    ...
    var m = new fProxyMxN(_count, _features, Allocator.Temp);
    AsMatrix(ref m);
    fProxyStatsOP.covarianceInto(in m, ref dest);
    m.Dispose();
}
```

This also makes the fix for R1 §4.1 (`covarianceInto` means vector) more urgent on this path.

---

### NEW-5 — choleskyDecompositionPivot Schur update: column reads of L, column writes of W

**File:** `OP/Cholesky.fProxy.cs` lines 233–239
**Severity: MED**

```csharp
for (int i = k + 1; i < n; i++) {
    fProxy Lik = L[i, k];
    for (int j = k + 1; j <= i; j++) {
        W[i, j] -= Lik * L[j, k];   // L[j,k] with fixed k, varying j = stride-n column read
        W[j, i] = W[i, j];          // W[j,i] with fixed i, varying j = stride-n column write
    }
}
```

`L[j, k]` = `L.Data[j * n + k]` with `j` varying → stride-n column read.
`W[j, i]` = `W.Data[j * n + i]` with `j` varying → stride-n column write (the symmetric mirror).

`W[i, j]` with fixed outer `i` and varying inner `j` is contiguous (unit-stride row write, correct). The two column accesses are the problem.

This is the hot inner loop for pivoted Cholesky: it runs O(n²) times total across all k steps. Round 1 (§3.4) identified the zeroing loop and the initialization copy as minor, but did not analyze the Schur update column accesses — which are the structurally dominant cost for n ≥ 64.

**Root cause:** `L[j, k]` could be loaded into a temporary array `colL[j] = L[j, k]` before the j-loop (one stride-n scan replaced by a sequential scan + array reference). The symmetric write `W[j, i] = W[i, j]` has no clean row-major alternative without materializing the transpose; however, symmetry maintenance could be deferred to a post-step copy.

**Practical fix:** pre-extract the k-th column of L:

```csharp
// extract column k of L into a temp (one stride-n scan, replaced by unit-stride array)
// (for small n, use stackalloc or a pre-allocated scratch vector)
for (int j = k + 1; j < n; j++) colL[j] = L[j, k];

for (int i = k + 1; i < n; i++) {
    fProxy Lik = L[i, k];
    for (int j = k + 1; j <= i; j++) {
        W[i, j] -= Lik * colL[j];   // contiguous j-loop read of colL ✓
        W[j, i] = W[i, j];          // column write W[j,i] still strided — unavoidable for symmetry
    }
}
```

Eliminating the `L[j,k]` stride-n read halves the cache pressure per inner iteration. The symmetric mirror write remains strided but requires W (not L), which fits in cache better for moderate n.

---

### NEW-6 — standardizeRows: round-1 pass count is WRONG (4 passes, not 3)

**File:** `Statistics/StatsOP.fProxy.cs` lines 771–785
**Severity: LOW-MED** (corrects R1 §3.2)

Round 1 states "3 passes over A (mean + 2 for variance)". The actual count is **4**:

1. `rowMean` → 1 pass over A (computes per-row means into s0)
2. `rowStdDev` → `rowVariance`: **2 passes over A** (rowVariance computes its own per-row means internally, then accumulates squared deviations — lines 445–458)
3. Apply pass (lines 778–783): 1 pass over A (row-major; correct)

Total = **4 passes**. Worse: row means are computed **twice** — once in `rowMean` (step 1) and once inside `rowVariance` (step 2). The separate `rowMean` call is entirely redundant because `rowVariance` already computes means internally without exposing them.

**Fix (minimum effort):** Remove the `rowMean` call and retrieve per-row means from `rowVariance`'s internal computation. Exposing the mean from `rowVariance` would require a signature change:

```csharp
// rowVariance overload that also returns per-row means:
// static void rowVarianceWithMeans(in fProxyMxN A, ref fProxyN varDest, ref fProxyN meanDest)
```

This reduces from 4 to 3 passes (1 mean+variance + 1 apply). An inline Welford-per-row reduces to 2 passes (1 online mean+variance + 1 apply).

---

### NEW-7 — rescaleRows and rescaleColumns each make two bounds passes

**File:** `Statistics/StatsOP.fProxy.cs`
**Severity: LOW**

`rescaleRows` (lines 857–873) calls `rowMin` then `rowMax` — two separate full passes over A. Same issue as R1 §3.3 (`range()` calls `max()` then `min()`), not caught there because `rescaleRows` was outside the reviewed scope.

`rescaleColumns` (lines 882–895) similarly calls `colMin` then `colMax` — two passes. The existing `meanMinMaxRange` utility at line 179 already demonstrates a single-pass combined min/max pattern; a `rowMinMax` helper could give the same benefit here.

**Fix:** A single-pass row min+max (outer=r, inner=c) accumulating both simultaneously, like `meanMinMaxRange`, would halve the number of passes. Impact is proportional to matrix size but remains LOW priority.

---

### NEW-8 — FFT radix-2 butterfly: loop-carried twiddle dependency blocks vectorization

**File:** `OP/FFT.fProxy.cs` lines 86–110
**Severity: LOW**

The inner `k` loop over butterflies within each stage carries a sequential dependency through `curRe` and `curIm`:

```csharp
for (int k = 0; k < half; k++)
{
    ...
    // cur *= w  (loop-carried dependency: each k depends on k-1)
    fProxy nRe = curRe * wRe - curIm * wIm;
    curIm = curRe * wIm + curIm * wRe;
    curRe = nRe;
}
```

Each butterfly must wait for the previous one's twiddle factor update. With ~5-cycle FP latency, throughput is bounded by roughly 1 butterfly per 5 cycles regardless of issue width. Burst cannot auto-vectorize the `k` loop.

The standard fix — precomputing all `half` twiddle factors into a temporary array before the `k` loop — breaks the dependency chain and allows Burst to vectorize. For a radix-2 FFT of N=1024, the final stage (len=1024, half=512) dominates, and precomputed twiddles would allow float4/double2 vectorization over the 512-element butterfly sweep.

**Trade-off:** Precomputed twiddles require an allocation of up to N/2 complex values per stage, or a pre-allocated scratch buffer passed by the caller. For the library's typical sizes (N ≤ 1024 per frame) and one-shot FFT calls, the allocation overhead may outweigh the gain. Flag for the hot-path zero-alloc overload if one is ever added.

---

## Prioritized Findings Table

Round-1 HIGH/MED findings are reproduced in abbreviated form for context; net-new findings are marked **NEW**.

| # | Severity | File : Lines | Issue | Est. Impact | Fix |
|---|---|---|---|---|---|
| R1-1 | **HIGH** | `UnsafeOP.fProxy.cs`:102–115 | `vecMatDot` stride-N inner loop | 2–5× on n≥64 | Swap loops (R1 §1.1) |
| R1-2 | **HIGH** | `StatsOP.fProxy.cs`:633–665 | `covarianceInto` O(N²) stride-N reads | Large for N≥20 | Center to scratch + `matMatDotTransA` (R1 §1.2) |
| **NEW-1** | **HIGH** | `ML/KMeans.fProxy.cs`:239–258 | Final-sync GEMM re-runs on convergence path | 2× O(N·D·k) on every converging call | Skip sync when `changes==0` |
| **NEW-2** | **MED** | `StatsOP.fProxy.cs`:808–812, 939–943, 885–892, 984–992 | `standardizeColumns`/`centerColumns`/`rescaleColumns`/`maxAbsColumns` apply pass column-strided | Stride-N_Cols writes on hot transforms | Reorder apply to outer=r, inner=c |
| R1-3 | **MED** | `OrthoOP.fProxy.cs`:112–134 | QR Householder stride-N application | 2–4× for m≥4n | Row-major AXPY rewrite (R1 §1.3) |
| R1-4 | **MED** | `SVD.fProxy.cs`:75–81 | Jacobi alpha/beta/gamma stride-N cols | Real for m≥100 | Copy cols p,q before inner loop (R1 §1.4) |
| R1-5 | **MED** | `Eigen.fProxy.cs`:282–285 | Jacobi `if (i==p||i==q)` in hot loop | Suppresses SIMD | Handle p,q outside loop (R1 §2.2) |
| **NEW-3** | **MED** | `Realtime/RollingWindow.fProxy.cs`:140–147 | `RollingWindow.Mean` column-strided (outer=feature, inner=ring-row) | Stride-_features misses for large feature counts | Accumulate in row order (outer=count, inner=feature) |
| **NEW-4** | **MED** | `Realtime/RollingWindow.fProxy.cs`:163–174 | `RollingWindow.Covariance` adds two temps to arena per call (matrix + means-vec) | Per-frame leak without ClearTemp | Function-local `Allocator.Temp` for intermediate matrix; fix R1 §4.1 for means-vec |
| R1-6 | **MED** | `StatsOP.fProxy.cs`:638 | `covarianceInto` means in arena temp pool | Accumulates per call | `new fProxyN(N, Allocator.Temp)` + Dispose (R1 §4.1) |
| **NEW-5** | **MED** | `Cholesky.fProxy.cs`:233–239 | Schur update inner loop: `L[j,k]` column read + `W[j,i]` column write | Stride-n for n≥64 | Pre-extract column k of L; defer symmetric mirror |
| R1-7 | **MED** | `Solvers.fProxy.cs`:234–237; `MatrixMetrics.fProxy.cs`:37,59 | CG/cond/rank alloc wrappers add to arena temp | Arena bloat per-frame | Zero-alloc overloads + ClearTemp cadence (R1 §4.2) |
| R1-8 | **MED** | `Cholesky.fProxy.cs`:154–156; `SVD.Solvers.fProxy.cs`:220–222; `RandomMatrixOP.fProxy.cs`:315–317 | Double-loop zeroing vs MemClear | Real for n≥50 | `UnsafeUtility.MemClear` (R1 §3.1) |
| **NEW-6** | **LOW-MED** | `StatsOP.fProxy.cs`:771–785 | R1 §3.2 count WRONG: 4 passes (not 3); row means computed twice (rowMean + inside rowVariance) | 1 redundant full pass | Expose means from rowVariance or use inline Welford |
| R1-9 | **LOW** | `ResampleOP.fProxy.cs`:294–313 | `resample2DInto` pass-2 column-strided | Moderate for dstN≥512 | Transpose scratch before vertical pass (R1 §1.5) |
| R1-10 | **LOW** | `RandomOP.fProxy.cs`:539–543 | Box-Muller: separate sin+cos calls | ~50% trig overhead in Gaussian fill | `math.sincos` when template allows out-params (R1 §2.3) |
| **NEW-7** | **LOW** | `StatsOP.fProxy.cs`:857–895 | `rescaleRows`/`rescaleColumns` call rowMin+rowMax / colMin+colMax separately | 2× bounds passes | Single-pass combined min+max helper |
| R1-11 | **LOW** | `StatsOP.fProxy.cs`:771–785 | `standardizeRows` redundant mean computation | 1 extra full pass | Inline Welford or expose mean from rowVariance |
| R1-12 | **LOW** | `Cholesky.fProxy.cs`:370–381 | `SolveUpperTriangularTransposed` stride-n | Small n only; unavoidable | Acceptable; document for multi-RHS callers |
| **NEW-8** | **LOW** | `FFT.fProxy.cs`:86–110 | Butterfly inner loop: loop-carried twiddle dependency blocks vectorization | ~1 butterfly/5 cycles; not SIMD-vectorized | Precomputed twiddle table (requires allocation) |
| R1-13 | **LOW** | `UnsafeOP.fProxy.cs`:416,434 | `[BurstCompile]` on non-job static methods | Misleading, not harmful | Remove attribute (R1 §6.1) |
| R1-14 MOOT | ~~LOW~~ | `FFT.fProxy.cs`:202 | DFT `baseAng*k` not hoisted | **Moot: Burst LICM handles it; dominated by trig cost** | No action needed |

---

## Appendix — Additional Notes

### softmaxColumns (StatsOP.fProxy.cs lines 1035–1043)

All three inner `r` loops (find-max, exp+sum, divide) are column-strided. This is harder to fix than NEW-2 because all three loops must operate on the same column sequentially. A row-major rewrite requires two N_Cols-length scratch vectors (per-column max, per-column expSum) and three row-major passes. Feasible but more invasive; not included in NEW-2 to keep that finding focused on the trivially-fixable cases.

### RollingWindow.AsMatrix fast-path opportunity

When the ring has not yet wrapped (`_count < _capacity`), `OldestRow == 0` and all `_count` rows are contiguous from row 0 — a single `UnsafeUtility.MemCopy` of `_count * _features * sizeof(fProxy)` bytes would replace the nested loop. The wrapped case cannot be bulk-copied without a sort. This is a micro-optimization since AsMatrix is typically called once per frame; flagged for completeness.

### V/eigenvalues initialization with branch in inner loop

`SVD.fProxy.cs` lines 56–58 and `Eigen.fProxy.cs` lines 231–233 initialize the identity matrix with `(i == j) ? 1 : 0` in the inner loop. A MemClear followed by a diagonal-fill loop (n writes) is faster for n ≥ ~16. This is already covered implicitly by R1 §3.1 (zeroing loops) but the conditional form is slightly worse than a pure zero loop. Severity: LOW, same tier as R1 §3.1.

---

## Files Reviewed (this pass, in traversal order)

```
ML/KMeans.fProxy.cs
OP/FFT.fProxy.cs
OP/ResampleOP.fProxy.cs
Statistics/HistogramOP.fProxy.cs
OP/QueryOP.fProxy.cs
OP/QueryOP.Predicate.fProxy.cs
Realtime/RollingWindow.fProxy.cs
OP/SVD.fProxy.cs
OP/Eigen.fProxy.cs
OP/Cholesky.fProxy.cs
OP/Solvers.fProxy.cs
OP/OrthoOP.fProxy.cs
OP/UnsafeOP.fProxy.cs
Statistics/StatsOP.fProxy.cs
OP/RandomOP.fProxy.cs
```
