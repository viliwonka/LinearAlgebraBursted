# Stats

`Stats` (float/double family) + a narrower `int`/`short`/`long` family (`uint` excluded — the doc
comment calls its stats "unsigned-hostile", e.g. an unsigned range has no signed-overflow-safe
`range` reduction).

## Whole-array & per-axis reductions

Whole-array (`floatN`, or a `floatMxN` treated flat): `sum`, `mean`, `variance`/`varianceSample`,
`stdDev`/`stdDevSample`, `argmin`/`argmax`, `min`/`max`, `median`, `range`, plus two bundled-result
shortcuts that compute several of the above together in as few passes as possible: `meanMinMaxRange(x)`
and the full-summary bundle `meanMinMaxRange_medianIQRstdDevVariance(x)`.

Per-axis, each with a `ref floatN dest` zero-alloc form and an allocating form: `rowSum`/`colSum`,
`rowMean`/`colMean`, `rowMin`/`rowMax`/`colMin`/`colMax`, `rowVariance`/`colVariance`,
`rowStdDev`/`colStdDev`, `rowNormL1`/`rowNormL2`/`colNormL1`/`colNormL2`.

## Covariance & correlation

`covarianceInto(in A, ref C)` / `covariance(in A)` / `correlation(in A)` — computed via the Gram
formulation (center once into a scratch, then `centeredᵀ · centered` through
[`Blas`](la-primitives.md)'s `matMatDotTransA`), not the naive O(N²) column-pair loop. Degrades to a zero
matrix (not NaN) when `M < 2` for the `ref`-dest primitive; the allocating wrappers still throw.

## Transforms

Whole-array and per-axis (`Rows`/`Columns`) in-place variants of `standardize`, `rescale` (+
`(lo,hi)` overload), `center`, `maxAbs`, `softmax`.

## Integer stats — widened-return convention

`sum → long` (overflow headroom); `mean`/`variance`/`stdDev`/`varianceSample`/`stdDevSample`/`median
→ double` (need a fractional result); `min`/`max → int` (same type); `argmin`/`argmax → int` (index).
Whole-array only — no per-axis reductions, covariance, or in-place transforms for the integer family.

## Realtime

[`RollingWindow`](realtime.md) reuses `covarianceInto` for a moving covariance over a ring buffer.

## Histogram & resampling

Two smaller features that live alongside Stats (design doc:
[spec-histogram-resample.md](../dev/spec-histogram-resample.md)):

- **`Histogram`** — `histogramInto(in data, lo, hi, ref Indices counts)` (+ an auto-range overload
  that finds finite min/max in one pass), `densityInto`, `cdfInto` (monotone, last bin pinned to
  exactly 1.0), `histogram2DInto`. **Out-of-range and NaN samples are always dropped, never thrown**
  (the closed upper edge `x == hi` lands in the last bin, not dropped). `ref`-dest only, no allocating
  wrapper.
- **`Resample`** — `sampleAt(in data, float pos, Interp, EdgeMode) : float` (continuous-position
  evaluation: `Nearest`/`Linear`/`Cubic` Catmull-Rom over 4 taps), `resampleInto` (1D endpoint-
  preserving resize, point-resampling, no anti-alias prefilter), `resample2DInto` (separable 2-pass).
  `EdgeMode{Clamp, Wrap, Mirror}`.

## Performance

`covarianceInto` uses the row-major Gram formulation above, and the `standardizeColumns` /
`centerColumns` / `rescaleColumns` / `maxAbsColumns` apply passes are row-major (contiguous, not
column-strided).
