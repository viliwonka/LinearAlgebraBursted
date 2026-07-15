# FFT / DFT

`FFT`. 1D transforms over **split real/imaginary** arrays (`floatN re`, `floatN im`) — there is no
complex type. Convention: forward `X[k] = Σ x[n]·exp(-2πi·kn/N)` (no forward scaling); the inverse
divides by N, so `ifft(fft(x)) == x`.

## Entry points

- **`fft(ref re, ref im, in ws)` / `ifft(...)`** — in-place power-of-two transform (throws otherwise —
  use `dft` for arbitrary N). Dispatches to table-indexed radix-4 or mixed-radix (one radix-2 stage +
  two radix-4 sub-FFTs), covering every power-of-two. Needs a workspace (`arena.floatFFTCache(n)`),
  built once and reused.
- **`rfft(in real, ref re, ref im[, in ws])` / `irfft(...)`** — real input, packs N samples into an
  N/2-point complex FFT and unpacks the half spectrum (`re`/`im` length **N/2+1**; `im[0]`/`im[N/2]`
  always zero).
- **`dft(in inRe, in inIm, ref outRe, ref outIm)` / `idft(...)`** — direct O(N²), works for
  **arbitrary N** — the fallback when N isn't a power of two.
- `magnitude`/`powerSpectrum`/`phase(in re, in im, ref dest)` — spectrum post-processing.

Workspace `floatFFTCache` (twiddle tables + rfft/mixed-radix scratch) is built once via
`arena.floatFFTCache(n)`, persistent, disposed with the arena; single-use-at-a-time (one per thread
for parallel transforms, FFTW "plan" semantics).

## Usage

All FFT/IFFT/rfft lengths must be a **power of two** — use `dft` for arbitrary N. Build a workspace
once (its twiddle-table build amortizes after ~1–3 transforms) and reuse it for every transform of
that size:

```csharp
var ws = arena.floatFFTCache(1024);   // builds the twiddle table on creation
for (int f = 0; f < frames; f++)
    FFT.fft(ref re, ref im, in ws);   // zero-alloc, reuses the plan
```

## Accuracy

The workspace builds its twiddle table at double precision, so there is no recurrence drift even at
very large `float` N. The transforms are validated by a stability suite: fft-vs-dft cross-check (both
dispatch paths, N up to 2048), Parseval energy, linearity, analytic transforms
(impulse/constant/exponential/shift), rfft↔fft consistency, and large-N round-trips.

## Cross-platform stability

The **workspace path** (`fft(ws)`/`ifft(ws)`/`rfft(ws)`/`irfft(ws)`) is bit-for-bit reproducible
across CPU architectures for a fixed Burst version, provided the caller's job is compiled under
`FloatMode.Strict`. Its twiddle table is built from roots of unity using only `+ - * sqrt` (no
`sin`/`cos`), and the butterflies are only `+ - *`; `sqrt` is IEEE-754 correctly-rounded (identical
on every platform) and `+ - *` do not reassociate under `Strict`, so there is no architecture- or
library-dependent rounding anywhere in the path. This is what a deterministic lockstep sim needs.

**`dft`/`idft`** (the arbitrary-N fallback) compute their twiddles with `math.sin`/`math.cos`. Burst
only guarantees those bit-identical under `FloatMode.Deterministic` (opt-in, 64-bit only) — under
`Strict` they are *not* cross-architecture reproducible. The power-of-two workspace transforms are
deterministic; `dft`/`idft` are not.

## Performance

The transforms use an in-place mixed-radix (radix-4/2) core over a twiddle-table workspace, built once.

Ryzen 9 9950X3D, single-thread Burst, median of 4. N=1,048,576 (2²⁰,
`Benchmarks/FFTBenchmark.cs`); this size is memory-bandwidth-bound, so absolute ms varies a few % with
machine memory traffic:

| Path | dtype | med(ms) |
|---|---|---|
| `fft(ws)` (twiddle-table workspace) | float | 6.56 |
| `fft(ws)` | double | 7.25 |
| `rfft(ws)` (twiddle-table workspace) | float | 3.62 |
| `rfft(ws)` | double | 4.10 |
| `irfft(ws)` (twiddle-table workspace) | float | 4.04 |
| `irfft(ws)` | double | 4.66 |
