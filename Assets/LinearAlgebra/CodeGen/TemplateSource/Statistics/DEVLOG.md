# DEVLOG — Statistics
Code comments state contracts only; history lives here (see CLAUDE.md).

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
