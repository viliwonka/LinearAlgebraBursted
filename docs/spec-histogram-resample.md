# Spec — Histogram + Resampling

Status: **SPEC (coder-ready)** · 2026-06-27 · fProxy-only (float/double)

Two small, self-contained features that compose with the existing Random + Stats + Gen layers.
Both written as codegen template sources (`fProxy` → float/double), zero-alloc `ref`-dest
primitives + optional allocating Arena wrappers, `ArgumentException("method: msg")` at API entry.

---

## Part 0 — Research / motivation

### Histogram — game & simulation use cases
1. **Telemetry / analytics** — frame-time / FPS buckets, latency, session length, damage-dealt spread.
2. **Procgen validation** — terrain-height / biome / loot-rarity distribution sanity checks.
3. **Balancing** — verify the shape of a damage or drop-rate formula (pairs with the Random layer).
4. **Heatmaps (2D)** — bin player positions / deaths / activity over the map → spatial density matrix;
   weighted variant for intensity. This is the headline 2D use.
5. **AI / RL** — discretize a continuous state/sensor space into bins (tabular Q-tables).
6. **Color / tone** — luminance histograms for auto-exposure, histogram equalization for contrast.
7. **Audio / DSP** — spectral binning (pairs with the existing FFT), loudness histograms.
8. **Physics sims** — particle velocity / energy distributions (Maxwell–Boltzmann checks), Monte-Carlo
   aggregation, empirical density estimation.
9. **Sampling FROM data** — a histogram IS an empirical PMF. Raw counts feed the existing
   `fProxyRandomOP.weightedPick` directly (it normalizes internally), so "sample from a histogram" =
   `histogram` → `weightedPick` (bin) → uniform-or-interpolated within bin. The `cdf` output enables
   inverse-transform sampling; **resampling the CDF gives smooth continuous sampling** — the exact
   "turn a vector into a smooth function to sample from" the user described.

### Resampling / interpolation — does the library have any? **No.**
`fProxyGenOP.sample<F>` samples a *functor* (formula); nothing resamples *existing data* to a new
resolution, and nothing evaluates a data vector at an arbitrary continuous position. Genuine gap.

Use cases: animation-curve / camera-path smoothing (Catmull–Rom), heightmap / texture resize (LOD),
audio sample-rate conversion, time-series alignment to a fixed cadence, signal up/down-sampling around
the FFT, and turning a histogram/PDF/CDF into a smooth function for sampling.

Methods: **Nearest** (`round`), **Linear** (`lerp` of 2 neighbors), **Cubic** = Catmull–Rom (interpolating,
C1, local 4-point stencil — the standard game cubic; passes through the data, needs no derivative input).
Boundaries handled by an edge mode: **Clamp** (repeat edge), **Wrap** (periodic), **Mirror** (reflect).
2D (matrix resize) is separable: bilinear = two 1D linears, bicubic = two 1D cubics.

---

## Part 1 — Placement decisions (aesthetics)

- **Histogram → new `fProxyHistogramOP`**, file `Statistics/HistogramOP.fProxy.cs`, namespace
  `LinearAlgebra.Stats` (sibling to `fProxyStatsOP`). Rationale: StatsOP is scalar/per-axis *reductions*;
  histogram is *binning / distribution estimation* — a distinct family (counts / density / cdf / 2D). One
  concept per OP, matching QueryOP / NormsOP / RandomOP / GenOP. Lives in the Stats folder & namespace so
  it's grouped with statistics.
- **Resampling → new `fProxyResampleOP`**, file `OP/ResampleOP.fProxy.cs`, namespace `LinearAlgebra`
  (like GenOP). Sibling to GenOP (`sample<F>` samples a formula; `ResampleOP` samples data). Two enums in
  one singular file `OP/ResampleEnums.cs`: `Interp { Nearest, Linear, Cubic }`, `EdgeMode { Clamp, Wrap, Mirror }`.

### Codegen constraints (load-bearing — see memory)
- Templates compile in firstpass: proxy types + Unity.Mathematics available, **generated concretes
  (`intN`/`intMxN`) are NOT**. → **1D integer counts use the shared `Indices` type** (read/write int buffer),
  never `intN`. The **2D heatmap returns `fProxyMxN`** (counts as float-valued matrix) — sidesteps the
  missing int-matrix type AND is the more useful form (normalize / resample / feed matrix ops directly).
- Use `(fProxy)0` / `(fProxy)1` casts, never `0f/1f`. No `const fProxy`.
- `math.lerp`, `math.floor`, `math.clamp`, `math.round` are all available.

---

## Part 2 — `fProxyHistogramOP` API

All bin layouts: **K equal-width bins** over `[lo, hi)`, width `w = (hi - lo) / K`; bin index
`b = (int)floor((x - lo) / w)`. **Out-of-range policy = DROP** (numpy convention): a value with `b < 0`
or `b >= K` is skipped; the single exception is `x == hi`, which lands in the **last bin** `K-1`
(so the closed upper edge is counted). Document this on every method.

1. `void histogramInto<T>(in T data, fProxy lo, fProxy hi, ref Indices counts)`
   - `K = counts.N`. Validates `K >= 1`, `hi > lo` (else throw). **Zeroes `counts` first** (caller buffer
     may hold garbage), then accumulates. Returns void; total counted may be < data.Length (drops).
2. `void histogramInto<T>(in T data, ref Indices counts)` — **auto-range** overload: one pass for
   `min`/`max`, then `lo = min`, `hi = max`. With `hi == min` (constant data) → all mass in bin 0
   (special-case: put every in-range sample in bin 0 to avoid div-by-zero). Guarantees no drops.
3. `void densityInto<T>(in T data, fProxy lo, fProxy hi, ref fProxyN dest)`
   - `dest[b] = count_b / (N * w)` so `Σ dest[b] * w == 1` (a proper density integrating to 1, numpy
     `density=True`). `N` = data.Length. `dest.N` = K. Empty data → throw. (PMF form is just `density*w`;
     and raw counts already feed `weightedPick`, so we expose density as the normalized statistical form.)
4. `void cdfInto<T>(in T data, fProxy lo, fProxy hi, ref fProxyN dest)`
   - Cumulative **normalized over the in-range samples**: `dest[b] = (Σ_{i<=b} count_i) / inRangeTotal`,
     monotone non-decreasing, `dest[K-1] == 1` when any sample is in range (if all dropped → all zeros).
     This is the inverse-transform / smooth-sampling feeder.
5. `void histogram2DInto<TX, TY>(in TX dataX, in TY dataY, fProxy loX, fProxy hiX, fProxy loY, fProxy hiY, ref fProxyMxN counts)`
   - Joint heatmap. `dataX.Data.Length == dataY.Data.Length` (paired points) else throw. `counts` is
     `Kx × Ky` (**rows = X bins, cols = Y bins** — document). Float-valued counts (each in-range pair adds
     1). Zeroes `counts` first. Same per-axis drop + closed-upper-edge rule on each axis independently.

Generic constraint everywhere: `where T : unmanaged, IUnsafefProxyArray` (so vec + matrix both work —
a matrix histograms its flat data, exactly like StatsOP scalar reductions).

(Arena allocating wrappers are **out of scope for v1** — keep surface minimal; ref-dest primitives only.)

---

## Part 3 — `fProxyResampleOP` API

Continuous index coordinate convention: a length-`N` vector spans positions `[0, N-1]`; `pos` need not be
integral. Edge mode resolves indices outside `[0, N-1]` for all four cubic taps / two linear taps.

1. `fProxy sampleAt(in fProxyN data, fProxy pos, Interp interp, EdgeMode edge)`
   - The core kernel — evaluate the data vector as a continuous function at `pos`. `data.N >= 1` else throw.
   - Nearest: `data[idx(round(pos))]`. Linear: `lerp(data[idx(i0)], data[idx(i1)], frac)`, `i0=floor(pos)`.
     Cubic: Catmull–Rom over taps `i0-1,i0,i0+1,i0+2` with parameter `frac`:
     `0.5 * ((2p1) + (-p0+p2)t + (2p0-5p1+4p2-p3)t² + (-p0+3p1-3p2+p3)t³)`.
   - `idx(i)` applies the edge mode: Clamp → `clamp(i,0,N-1)`; Wrap → `((i % N) + N) % N`; Mirror →
     reflect into `[0,N-1]` (period `2N-2`, the standard no-edge-repeat reflection; for `N==1` → 0).
2. `void sampleAtInto(in fProxyN data, in fProxyN positions, ref fProxyN dest, Interp interp, EdgeMode edge)`
   - Gather: `dest[j] = sampleAt(data, positions[j], …)`. `dest.N == positions.N` else throw.
     Alias-safe note: `dest` may not alias `data` (reads of `data` must be stable) — document; no guard.
3. `void resampleInto(in fProxyN src, ref fProxyN dst, Interp interp, EdgeMode edge)`
   - Resize `src.N → dst.N`. Endpoint-preserving map: `pos(j) = j * (src.N - 1) / (dst.N - 1)` for
     `dst.N > 1`; `dst.N == 1` → `dst[0] = src[0]`. `src.N >= 1` else throw. (So upsample & downsample
     both hit `src[0]` and `src[N-1]` exactly.) Downsampling note: this is point-resampling (no
     anti-alias prefilter) — document that decimating noisy data may alias; callers smooth first
     (the existing `gaussianKernel` / convolution path).
4. `void resample2DInto(in fProxyMxN src, ref fProxyMxN dst, Interp interp, EdgeMode edge)`
   - Matrix resize `src.(M×N) → dst.(M'×N')` via **separable** interpolation along rows then columns
     (NN / bilinear / bicubic per `interp`). Endpoint-preserving on each axis independently. `src` at
     least `1×1`. Implementation may use one `Allocator.Temp` scratch (intermediate `M × N'` or row
     buffers); dispose before return; validate args BEFORE allocating so the throw path can't leak.
     Single-row or single-column `src` degenerates gracefully (that axis returns the edge value).

Cubic = Catmull–Rom in both 1D and 2D. Mirror/Wrap/Clamp identical semantics in 2D, applied per axis.

---

## Part 4 — Tests (test-writer, after review)

Histogram:
- Known small data → exact bin counts (hand-computed); closed-upper-edge (`x == hi` → last bin);
  out-of-range drop; auto-range covers full span with no drops; constant data → all in bin 0.
- `densityInto`: `Σ dest*w ≈ 1`. `cdfInto`: monotone, last == 1, matches cumulative counts.
- 2D: paired points land in the right `(xbin, ybin)` cell; row/col convention asserted; mismatched
  lengths throw.
- Feed counts → `weightedPick` smoke test (the sampling bridge).
- Float + double generated variants.

Resample:
- `sampleAt` at integer positions returns exact data values (all three interps). Linear midpoint =
  mean of neighbors. Cubic reproduces a cubic/parabola polynomial exactly on a uniform grid (Catmull–Rom
  property). Edge modes: Clamp/Wrap/Mirror produce the documented out-of-range taps.
- `resampleInto`: identity when `dst.N == src.N` (Linear/Cubic reproduce input on aligned grid);
  endpoints preserved on up & downsample; upsample of a linear ramp stays linear (Linear interp).
- `resample2DInto`: resize of a separable / planar field; endpoints preserved; bilinear of a bilinear
  field is exact; `M'==M, N'==N` identity.
- Throw paths: empty input, mismatched dest lengths.
- Float + double generated variants.

---

## Part 5 — Build order
1. Coder A: `ResampleEnums.cs` + `fProxyResampleOP` (no deps).
2. Coder B: `fProxyHistogramOP` (no deps; `Indices` already exists).
   (A & B touch disjoint files — can run in parallel; neither runs regen.)
3. One `Tools/regen-and-test.ps1` pass (single codegen run — avoid parallel regen race).
4. 3 review agents (code-review-1/2/3) in parallel → fix CRITICAL/MAJOR.
5. test-writer → regen+test green.
6. README feature line + memory note. Commit when the user asks.
