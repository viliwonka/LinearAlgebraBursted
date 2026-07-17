# Spec: raw-pointer `.Data.Ptr` hoist pass

Status: ready for implementation. Scan 2026-07-15, all sites re-verified against current
templates 2026-07-17. Line numbers below are CURRENT as of this verification; re-confirm the
exact loop text before editing (git may have moved things again).

## 1. Goal and mechanism

Loops that read/write a `fProxyN` / `fProxyMxN` through the struct indexer (`vec[i]`,
`A[r, c]`) inside Burst-compiled code go through `Data.ElementAt(...)` (plus
`ENABLE_UNITY_COLLECTIONS_CHECKS` bounds asserts) — opaque to the auto-vectorizer, so the
loop runs scalar (~8x penalty measured on DetMath; see `docs/dev/perf-vectorization-lessons.md`).
Fix: hoist a raw pointer before the loop and index it directly:

```csharp
unsafe {
    fProxy* ap = A.Data.Ptr;
    fProxy* row = ap + (long)r * A.N_Cols;   // (long) cast on row offsets — match LU siblings
    for (int c = 0; c < n; c++) sum += row[c];
}
```

The kernel layer (UnsafeOP, fProxyComp, Blas.Fused/Triangular/ColumnScaling, operator
overloads) is already disciplined. This pass fixes the mid/high-level solver/stats/ML layer.

## 2. Binding rules

1. **Templates only.** All edits go to `Assets/LinearAlgebra/CodeGen/TemplateSource*`.
   Never touch `Assets/LinearAlgebra/Source/` or other generated trees. Regenerate with
   `Tools/regen.ps1` after each batch. fProxy templates emit both float and double outputs.
2. **Unit-stride only.** Only hoist loops whose INNER index walks contiguous memory
   (row-major: inner index = column, or flat). Strided column walks
   (`for c { for r { A[r,c] } }`) are excluded — they will not SIMD hoisted or not.
   Every site listed below is verified unit-stride; do not add new sites without checking.
3. **Bit-identity is mandatory.** The default transform is a *pure hoist*: same operations,
   same order, only the addressing changes. That is bit-identical by construction — no
   reassociation, no reordering. Consequences:
   - Reduction loops (dots, sums, variance accumulations, min/max scans): pure hoist ONLY.
     Do **not** reroute them through `UnsafeOP.vecDot` / `sum` / `sumAbs` / `maxAbs` — those
     use multi-accumulator SIMD folds with a different summation order and are NOT
     bit-identical to the scalar loop.
   - Elementwise axpy-shaped loops (`y[i] -= a * x[i]`): may be routed through
     `UnsafeOP.axpy(y, x, -a, n)` where non-aliasing is provable (see per-site notes).
     `y[i] + (-a)*x[i] == y[i] - a*x[i]` exactly in IEEE 754, so this is bit-identical
     (the same argument is already stated at `OP/LU.fProxy.cs:149-154`).
4. **`[NoAlias]` ruling.** Never drop `[NoAlias]` from an existing hot kernel without an A/B
   benchmark. This pass does not modify UnsafeOP signatures. When calling
   `UnsafeOP.axpy` (both pointers `[NoAlias]`), the two pointers must be provably distinct
   regions — each such call below has a per-site legality note.
5. **Comment policy.** No perf narration or history in code comments. Where a hoist needs
   any comment at all, one contract-level line in the style of the existing LU comment is
   the maximum. Batch outcomes, benchmark numbers, and any reverted sites go to the folder's
   `DEVLOG.md` (newest first, date + one line).
6. **Expectation setting (for the A/B step, not for code comments).** Elementwise loops
   (transforms, axpys, matrix builds) get the full SIMD win. Sum-reduction loops keep their
   serial dependency chain under Burst's default float mode (no reassociation), so their gain
   is addressing/bounds-check removal and unrolling — smaller. min/max reductions may or may
   not vectorize. The A/B benchmark decides; if a site measures a regression, revert that
   site and log it in DEVLOG (precedent: prefetch/Eisenstat reverts).
7. **Verification loop per batch:** capture benchmark baseline → edit templates →
   `Tools/regen.ps1` → `Tools/run-tests.ps1` (full suite green) → rerun benchmark →
   record before/after in the batch DEVLOG entry. Optional: spot-check vectorization with
   `docs/dev/burst-disasm-recipe.md`.

## 3. Canonical transforms

**T1 — row reduction (pure hoist):**
```csharp
// before
for (int r = 0; r < A.M_Rows; r++) {
    fProxy sum = 0f;
    for (int c = 0; c < A.N_Cols; c++) sum += A[r, c];
    dest[r] = sum;
}
// after
unsafe {
    fProxy* ap = A.Data.Ptr; fProxy* dp = dest.Data.Ptr;
    int nc = A.N_Cols;
    for (int r = 0; r < A.M_Rows; r++) {
        fProxy* row = ap + (long)r * nc;
        fProxy sum = 0f;
        for (int c = 0; c < nc; c++) sum += row[c];
        dp[r] = sum;
    }
}
```

**T2 — elementwise / accumulate-into-dest (pure hoist):** same shape; hoist every buffer
touched in the inner loop (`A`, `dest`, scratch vectors like `means`), keep the body
verbatim on the hoisted pointers.

**T3 — two-row axpy routed to the kernel (only where marked):**
```csharp
// before:  for (int i = k+1; i < m; i++) U[j, i] -= Ljk * U[k, i];
// after (mirrors OP/LU.fProxy.cs:155-173):
unsafe {
    fProxy* up = U.Data.Ptr;
    fProxy* rowK = up + (long)k * m;
    ...
    fProxy* rowJ = up + (long)j * m;
    UnsafeOP.axpy(rowJ + (k + 1), rowK + (k + 1), -Ljk, len);
}
```

Whole-buffer copies of identical shape may instead use `X.Data.CopyFrom(Y.Data)` (memcpy,
bit-identical) — marked per site.

Hoist placement: hoist `.Data.Ptr` once per method (or once per outer loop) — never inside
the inner loop. Methods already containing an `unsafe` block extend it; others gain one.

## 4. Batches and verified sites

### Batch 1 — `LU.decompNoPivot` (proof batch, do first, alone)

File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LU.fProxy.cs`

| Site | Lines | Loop | Transform |
|---|---|---|---|
| elimination inner axpy | 45-66, inner **59-61** | `U[j, i] -= Ljk * U[k, i]` | T3, exactly the sibling pattern at 155-173 (`decomp` small path). `j > k` always, so rowJ/rowK are distinct rows — `[NoAlias]` legal. Keep `L[j,k] = Ljk` (57) and `U[j,k] = 0` (64) as-is or on hoisted ptrs; O(n²), taste call. |

Bit-identity: guaranteed (T3 argument; already documented for the identical sibling).

**LU/LUP split question (user floated; evaluate, not required):** recommendation — do NOT
split. `decompNoPivot` is already a separate public entry point with its own contract; after
this batch it shares the same vectorized axpy kernel as the pivoted paths, so a file/type
split buys no performance and adds codegen churn plus API surface risk pre-v1.0. Record the
recommendation and rationale in `OP/DEVLOG.md`; leave the decision to the user.

Benchmark obligation: `TemplateSourceBenchmarks/LUBenchmark.fProxy.cs` currently measures
only `LU.decomp` (line 27) — **gap**. Add a `decompNoPivot` job + result rows to that
benchmark template (same harness pattern as the existing job) BEFORE applying the hoist, so
before/after numbers exist. Run via `Tools/benchmark.ps1`. Expected: approach the pivoted
`decomp` time at the same n (siblings are the ceiling).

### Batch 2 — StatsCore row-major family

File: `Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsCore.fProxy.cs`
All matrix sites below are the ref-dest primitives; the allocating wrappers forward to them
and are covered automatically. All are T1/T2 pure hoists; all bit-identical. Hoist `A`,
`dest`, and any scratch vector (`means`, `s0`, `s1`, `mAbsArr`) used in the inner loop.

| Method | Lines (inner loop) |
|---|---|
| rowSum | 268-274 (271-272) |
| colSum | 289-294 (zero 289-290, accumulate 292-294; `dest[c]` and `A[r,c]` both unit-stride in c) |
| rowMin / rowMax | 341-347 / 364-370 |
| colMin / colMax | 387-392 / 409-414 |
| rowVariance | 433-447 (two passes per row) |
| colVariance | 470-487 (means 470-472, zero 475-476, accumulate 478-483) |
| rowStdDev / colStdDev | 500-501 / 514-515 (sqrt map over dest; small, include for uniformity) |
| rowNormL1 / rowNormL2 | 530-536 / 551-557 |
| colNormL1 / colNormL2 | 572-577 / 592-600 |
| covarianceInto | zero-fill 630-632, means 640-644, centered build 649-651, scale 660-662 (scale may run flat over N*N) |
| standardizeRows | 773-783 |
| standardizeColumns | pass 2 796-805, pass 3 807-813 (keep the per-element `!(sd > 0)` branch verbatim) |
| rescaleRows / rescaleColumns | 858-867 / 882-889 |
| centerRows / centerColumns | 916-920 / 931-933 |
| maxAbsRows / maxAbsColumns | 959-965 / 974-980 |
| softmaxRows | 1007-1014 (DetMath.Exp call stays; exp loop still vectorizes per DetMath precedent) |

Excluded (verified strided or non-vectorizable — do not touch): `softmaxColumns` 1018-1029
(column-inner), flat `argmin`/`argmax` 76-111 (index capture), `median`/percentile paths.
Already hoisted (skip): flat `standardize`/`rescale`/`center`/`maxAbs`/`softmax`, `sum`
(routes to UnsafeOP). Optional, low value: flat `variance`/`varianceSample` (38-42, 61-65)
use the `UnsafeList` indexer `x.Data[i]`; include only if free.

Note: `StatsCore.iProxy.cs` has integer twins of some of these; out of scope for this pass
(no float SIMD claim). Log as a possible follow-up in the Statistics DEVLOG.

Benchmark obligation: **gap — no benchmark covers the matrix stats family**
(`KernelBenchmark.fProxy.cs` measures only vector Norms/Stats.sum, already pointer-path).
Add a matrix-stats section — either a new `TemplateSourceBenchmarks/StatsBenchmark.fProxy.cs`
or a section in KernelBenchmark — measuring at minimum `rowSum`, `colSum`, `rowVariance`,
`standardizeRows`, `softmaxRows` at ~1024x1024, before applying the batch.

### Batch 3 — LP dense tableau simplex

File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs` (`simplexCore` + helpers)

| Site | Lines | Transform |
|---|---|---|
| `Pivot` normalize row | 587 (`T[prow, j] *= inv`) | T2 on `T` row pointer |
| `Pivot` elimination | 590-597, inner **595** (`T[i, j] -= f * T[prow, j]`) | T3 allowed: `i == prow` is skipped at 592, rows distinct — `[NoAlias]` legal. Or pure-hoist T2; either is bit-identical. |
| `Pivot` cost rows | 599-602 (`cost1[j] -= f1 * T[prow, j]`, same for cost2) | T3 allowed: `cost1`/`cost2` are separate `fProxyN` buffers from `T` — distinct allocations. |
| `simplexCore` initial pricing | 469-477, inner **475-476** (same axpy shape) | same as cost rows |

Excluded: `RatioTest` 568-577 (`T[i, enter]` — strided column walk).

Benchmark obligation: `TemplateSourceBenchmarks/LPBenchmark.fProxy.cs` section 1 runs the
tableau-simplex backend of `LP.solve` on the same problems as the other backends — covered.

### Batch 4 — KMeans

File: `Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.fProxy.cs`
All pure hoists (T1/T2), bit-identical. In `fit`, hoist `X`, `centroids`, `ws.*` pointers
once near the top of the method (after guards) and reuse.

| Site | Lines (inner) |
|---|---|
| PointNormSq precompute | 93-98 (96) |
| CentNormSq per iter | 130-135 (133); final-sync twin 231-236 (234) |
| Gram patch | 143-145 (unit-stride j over `ws.Gram` row + `ws.CentNormSq`); final-sync twin 238-240 |
| zero accumulators | 172-176 (174) |
| accumulate points | 179-184 (183) — row base of `ws.NewCentroids` from `assignment[n]`, inner f unit-stride |
| D2Weights refill | 192-193 (contains `Gram[n, assignment[n]]` gather — hoist ptrs, expect no SIMD; keep) |
| reseed copy | 211 |
| divide to centroids | 217-221 (220) |
| `SeedKMeansPlusPlus` first-centroid copy | 353 |
| `SeedKMeansPlusPlus` init D2 | 358-367 (361-365) |
| `SeedKMeansPlusPlus` centroid copy | 390 |
| `SeedKMeansPlusPlus` incremental D2 update | 396-405 (398-404) |

Excluded: inertia sums 163-166 / 243-248 (`Gram[n, assignment[n]]` gather-dominated —
optional hoist only), farthest-point scan 203-210 (index capture), `SeedUniform` 419-434
(cold, O(N·D) once).

Benchmark obligation: `TemplateSourceBenchmarks/KMeansBenchmark.fProxy.cs` — covered.

### Batch 5 — Kalman UKF

File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.UKF.fProxy.cs`
All pure hoists, bit-identical.

| Site | Lines |
|---|---|
| `GenerateSigmaPoints` X row 0 fill | 69 |
| `GenerateSigmaPoints` diff zero | 73 |
| `GenerateSigmaPoints` sigma build | 79-84 (hoist `x`, `cache.diff`, `cache.X`; two row bases per k) |
| `ukfPredict` row copy in/out | 115, 117 |
| `ukfPredict` xPred zero + weighted mean | 120, 121-125 (124) |
| `ukfPredict` centered+scaled build | 131-140 (134-139; hoist `Y`, `X`, `xPred`) |
| `ukfUpdate` row copy | 182 |
| `ukfUpdate` Z write | 184 |
| `ukfUpdate` zPred zero + weighted mean | 189, 190-194 (193) |
| `ukfUpdate` dX / dZ / WdZ build | 205-215 (208, 209-214) |

Excluded: `GenerateSigmaPoints` 76-77 (scatter through `cache.Piv[i]`).

Benchmark obligation: `TemplateSourceBenchmarks/KalmanBenchmark.fProxy.cs` (ukf predict+update
jobs at 219-220 and 272-273) — covered.

### Batch 6 — Secondary sites

All pure hoists unless noted; bit-identical. One sub-batch per file is fine; run the full
suite once per sub-batch minimum.

**`OP/QueryOP.fProxy.cs` — `argMaxRowNorm`** 264-303: L1 274-280 (277-278), L2 284-290
(287-288), Linf 294-300 (297-298). Row base per r; compare/branch stays scalar.
Excluded: `argMaxColNorm` 309+ (strided). Benchmark: none — **gap**; suite-green only, or
piggyback a row on the Batch-2 stats benchmark.

**`OP/NormsOP.fProxy.cs`** — `normalizeRows` 103-140 (norm loops 115 / 122 / 129, apply 138);
`matrixLInf` 200-212 (206-207). Excluded: `normalizeColumns` 143-180, `matrixL1` 185-197
(both strided). Benchmark: none — **gap**, same handling as QueryOP.

**`OP/LOBPCG.fProxy.cs`** — `FillGramSub` 894-905 (dot 900), `FillHSub` 914-925 (dot 920),
`Deflate` 937-954 (coeff dot 945, triple axpy 947-952). All row-row unit-stride over n.
Pure hoist ONLY — the dots are reductions (rule 3); do not reroute to `UnsafeOP.vecDot`.
The Deflate axpys may be routed to `UnsafeOP.axpy` per row only if V/AV/BV rows are provably
distinct from Against/AgainstA/AgainstB rows at every call site — if that audit is not
trivial, pure-hoist and move on. Benchmark: `LOBPCGBenchmark.fProxy.cs` — covered.

**`OP/Eigen.fProxy.cs` — lanczos body**: vCur copy 602-603, `w -= beta*V[jj-1,:]` 611-612,
alpha dot 617-618, `w -= alpha*vCur` 621-622, reorth passes 625-633 (proj dot 628-629, axpy
630-631), wNormSq 636-637, next-row scale 650-651. Hoist `ws.V`, `ws.w`, `ws.vCur` once.
Pure hoist only (dots are reductions). Benchmark: `LargeSparseBenchmark.fProxy.cs` (lanczos
over BSR) — covered.

**`OP/Control.fProxy.cs` (class LQR)** — `RiccatiStep`: `Rbar += R` 78-80, K seed copy 91-93
(same-shape full copy — `K.Data.CopyFrom(BSA.Data)` allowed, bit-identical),
`Snext = Q + AtSA - BSATK` 103-105. `lqrSchedule` Kschedule block copy 314-316 (row-offset
writes, inner c unit-stride). Benchmark: `LQRBenchmark.fProxy.cs` — covered.

**`OP/Riccati.fProxy.cs` (class Riccati)** — `dare` SDA accumulations `GkNext += Gk` 185-187,
`HkNext += Hk` 192-194. Excluded: diagonal bumps 171 (`[i,i]`), `SymmetrizeInPlace` 88-97
(mirrored strided write), `FrobeniusNorm`/`FrobeniusNormDiff` 57-80 (already hoisted).
Benchmark: LQRBenchmark (dare path) — covered.

**`OP/Optimize.fProxy.cs` — `ladIRLS`**: zero pass 256, residual dot 262, weighted
normal-equations accumulate 264-269 (Gram row update 268, `rhs` 267), prevx copy 273,
x copyback 281, final L1 residual 294-299 (297). Excluded: lower-triangle mirror 271
(strided write). Pure hoist only (262/268/297 involve reductions or run under them).
Benchmark: `FittingBenchmark.fProxy.cs` (ladIRLS job, line 59) — covered.

### Batch 7 — LOWER tier (optional; do only if batches 1-6 land clean)

- `ML/PCA.fProxy.cs`: `BuildWorkingCopy` centered build 135-137, standardized build 155-157;
  `transform` Xs build ~588-590. Excluded: `ComputeSampleStd` 110-115 and totalVariance
  140-145 (column-inner, strided), `ApplySignConvention` 95-96 (strided).
- `Arena/ArenaExtensions.fProxy.cs`: `fProxyHouseholderMat` rank-1 update 248-255.
- `OP/MPC.State.fProxy.cs`: Acond build 383-385, Phi block writes 405-407, Gamma block
  copies 416-428 (all inner-j unit-stride block copies).

No benchmark obligation; suite green suffices. Bit-identical (pure hoists).

## 5. Sites dropped after verification (do NOT implement)

- **`OP/SVD.Solvers.fProxy.cs` back-transform** (scan listed 448/491; current 441-449,
  485-492, plus single-RHS 84-90 / 133-139 and `pinv` 266-269): the inner loops walk ROWS
  of row-major `U`/`M`/`B`/`X` (stride = N_Cols) — strided, will not SIMD. Dropped.
- **Riccati Frobenius norms** (`OP/Riccati.fProxy.cs:57-80`): already raw-pointer. Dropped.
- Re-confirmed exclusions from the scan: `softmaxColumns`, `normalizeColumns`, `matrixL1`,
  `argMaxColNorm`, `RatioTest`, Dot.householderInPlace column walk, diagonal `[i,i]` bumps,
  argmin/argmax index-capture, UnsafeOP/fProxyComp/Blas.* (already disciplined).

## 6. Batch order and acceptance criteria

Order: 1 (LU proof) → 2 (StatsCore) → 3 (LP) → 4 (KMeans) → 5 (UKF) → 6 (secondary,
per-file sub-batches) → 7 (optional). One commit per batch, full suite green before each
commit.

Acceptance:
1. `Tools/regen.ps1` clean; full headless suite green via `Tools/run-tests.ps1` after every
   batch (no new failures vs current baseline).
2. Numerical outputs bit-identical: every transform in this spec is either a pure hoist or
   an IEEE-exact axpy rewrite — any test-tolerance change or golden-value change is a bug in
   the batch, not an acceptable drift. Revert and investigate.
3. Measurable speedup on the relevant benchmark for **Batch 1 (LU)** and **Batch 2
   (StatsCore)** — these two are hard requirements. Other batches: record before/after; a
   flat result is acceptable, a regression on any site is reverted (log in DEVLOG).
4. Benchmark gaps closed as specified: LUBenchmark gains a `decompNoPivot` case; a matrix
   stats benchmark section exists. QueryOP/NormsOP remain unbenchmarked (accepted gap,
   noted here).
5. `Tools/check-doc-leaks.ps1` clean; no perf/history comments added to code; each batch
   adds one dated DEVLOG line in the touched template folder (including the LU/LUP split
   recommendation from Batch 1).
