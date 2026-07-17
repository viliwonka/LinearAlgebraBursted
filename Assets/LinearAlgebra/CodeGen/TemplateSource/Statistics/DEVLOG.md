# DEVLOG — Statistics
Code comments state contracts only; history lives here (see CLAUDE.md).

## StatsCore rowMin/rowMax + Stats.min/max: reroute to UnsafeOP.min/max
- 2026-07-17 | The row/vector max/min reductions were scalar (`m = math.min(m, x[i])` — a carried reduction,
  no auto-vectorisation). Rerouted `rowMin`/`rowMax` (per-row, contiguous) and the generic `min<T>`/`max<T>`
  (whole array) to the new `UnsafeOP.min`/`max` kernels, mirroring the shipped rowNormL1/L2 → sumAbs/vecDot
  per-row pattern. Bit-identical on finite data (max/min exact; the MaxValue/MinValue seed was neutral).
  LEFT AS-IS: `colMin`/`colMax` already use unit-stride `dp[c] = math.min(dp[c], row[c])` (the colSum-trick
  form, auto-vectorises); `meanMinMaxRange` stays a single fused min+max+sum pass (rerouting = 3 passes).

## StatsCore.fProxy.cs — row reductions routed to SIMD kernels
- 2026-07-17 | Follow-up to the hoist below. Under FloatMode.Strict (== Default in Burst), a plain
  `sum += row[c]` row reduction CANNOT auto-vectorise (lane-splitting reorders the sum, which Strict
  forbids), so post-hoist rowSum/rowNormL1/rowNormL2 were still scalar (~1.5× = de-index only) while
  colSum — an elementwise accumulate into a per-column vector, no reassociation needed — vectorised to
  32×. Rerouted the three pure row reductions to the frozen-tree SIMD kernels: rowSum→`UnsafeOP.sum`,
  rowNormL1→`UnsafeOP.sumAbs`, rowNormL2→`sqrt(UnsafeOP.vecDot(row,row))`. These use fixed 2×fProxy4/
  fProxyW accumulator trees that are cross-arch deterministic (the frozen numeric contract in
  UnsafeOP.fProxy.cs) but NOT bit-identical to the prior serial left-to-right sum — a deliberate
  pre-1.0 baseline change (owner-approved: no bit-compat obligation before release). Suite 6317/6317
  stays green (existing tolerances absorb the last-ULP reorder). N=1024 float rowSum 0.35→0.035 ms
  (10× beyond the hoist, ~15× over the original indexer), now colSum-tier (~30 GFLOP/s). rowVariance/
  standardizeRows kept their stable two-pass scalar deviation sum (no single kernel; the one-pass
  sum-of-squares reformulation would risk catastrophic cancellation) — candidates for an explicit
  fProxyW two-pass later if needed.

## StatsCore.fProxy.cs — raw-pointer hoist (spec-raw-pointer-hoist-pass batch 2)
- 2026-07-17 | The O(M·N) row-major reduction/transform family (rowSum, colSum, row/colMin,
  row/colMax, row/colVariance, row/colStdDev, row/colNormL1/L2, covarianceInto, standardizeRows,
  standardizeColumns, rescale/center/maxAbs Rows+Columns, softmaxRows) was looping through the
  `fProxyMxN`/`fProxyN` struct indexer — opaque to Burst's auto-vectoriser, so the inner loops ran
  scalar. Hoisted `A.Data.Ptr` (per-row base `ap + (long)r*nc`) and the dest/scratch pointers before
  the loops; bodies kept verbatim on the raw pointers (pure hoist, no reassociation → bit-identical;
  suite 6317/6317 unchanged). Per-element NaN-safe guards (`!(sd>0)`, `!(rng>0)`) and the softmax
  max-scan branch left as-is (the branch-free rewrite is the separate math.select pass). Measured
  (9950X3D, N=1024, float): colSum 0.96→0.030 ms (32×), standardizeRows 2.14→0.79 ms (2.7×),
  rowVariance 1.27→0.73 ms (1.7×), rowSum 0.51→0.35 ms (1.5×), softmaxRows 4.56→3.12 ms (1.5×, exp-
  bound). Added StatsBenchmark (was no matrix-stats coverage). Column-inner variants (softmaxColumns,
  normalizeColumns via strided writes) correctly left on the indexer — strided, won't SIMD.
- 2026-07-17 | Follow-up (not done): `StatsCore.iProxy.cs` has integer twins of several of these
  (rowSum/colSum/min/max) still on the indexer — out of scope for this pass (no float-SIMD claim, and
  the long-accumulator contract differs); revisit if an integer-stats hot path appears.

## StatsCore.iProxy.cs
- 2026-07-12 | The `long`-accumulator sum contract (int/short always safe, long can wrap) is
  pinned by StatsTests.iProxy.cs's SumAccumulatorOwnOverflow: the same 2-element/MaxValue-filled
  input is correct-and-widened for int/short but silently wraps for long. (was StatsCore.iProxy.cs:27-29)
