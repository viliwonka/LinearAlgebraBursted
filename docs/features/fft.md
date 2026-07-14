# FFT / DFT

`FFT`. 1D transforms over **split real/imaginary** arrays (`floatN re`, `floatN im`) — there is no
complex type. Convention: forward `X[k] = Σ x[n]·exp(-2πi·kn/N)` (no forward scaling); the inverse
divides by N, so `ifft(fft(x)) == x`.

## Entry points

- **`fft(ref re, ref im)` / `ifft(...)`** — in-place, zero-alloc, no workspace. Power-of-two only
  (throws otherwise — use `dft` for arbitrary N). Dispatches to a zero-alloc radix-4 recurrence for
  power-of-4 lengths, radix-2 recurrence otherwise.
- **`fft(ref re, ref im, in ws)` / `ifft(...)`** — workspace (twiddle-table plan) form, fastest.
  Dispatches to true table-indexed radix-4 or mixed-radix (one radix-2 stage + two radix-4
  sub-FFTs), covering every power-of-two. ~1.3–1.9× faster than the no-workspace path; the table
  build amortizes after ~1–3 transforms, so build one (`arena.floatFFTCache(n)`) for any repeated use.
- **`rfft(in real, ref re, ref im[, in ws])` / `irfft(...)`** — real input, packs N samples into an
  N/2-point complex FFT and unpacks the half spectrum (`re`/`im` length **N/2+1**; `im[0]`/`im[N/2]`
  always zero).
- **`dft(in inRe, in inIm, ref outRe, ref outIm)` / `idft(...)`** — direct O(N²), works for
  **arbitrary N** — the fallback when N isn't a power of two.
- `magnitude`/`powerSpectrum`/`phase(in re, in im, ref dest)` — spectrum post-processing.

Workspace `floatFFTCache` (twiddle tables + rfft/mixed-radix scratch) is built once via
`arena.floatFFTCache(n)`, persistent, disposed with the arena; single-use-at-a-time (one per thread
for parallel transforms, FFTW "plan" semantics).

## Which one to use

All FFT/IFFT/rfft lengths must be a **power of two** — use `dft` for arbitrary N.

- **One-shot or a few transforms → no-workspace `fft(re, im)`.** Table-free, zero-allocation, nothing
  to set up (dispatches to a radix-4 recurrence for power-of-4 lengths, radix-2 otherwise).
- **Many transforms of the same size → build a workspace once and reuse it.** The twiddle-table build
  amortizes after only ~1–3 transforms, so this is the right default for any repeated use:

```csharp
var ws = arena.floatFFTCache(1024);   // builds the twiddle table on creation
for (int f = 0; f < frames; f++)
    FFT.fft(ref re, ref im, in ws);   // zero-alloc, reuses the plan
```

## Accuracy

The workspace builds its twiddle table at double precision (no recurrence drift); the no-workspace
recurrence accumulates a small twiddle drift at very large `float` N — negligible for typical use, but
prefer the workspace if you need maximum accuracy on huge float transforms. The transforms are
validated by a stability suite: fft-vs-dft cross-check, Parseval energy, linearity, analytic
transforms (impulse/constant/exponential/shift), rfft↔fft consistency, and large-N round-trips.

## Cross-platform stability

The **workspace path** (`fft(ws)`/`ifft(ws)`/`rfft(ws)`/`irfft(ws)`) is bit-for-bit reproducible
across CPU architectures for a fixed Burst version, provided the caller's job is compiled under
`FloatMode.Strict`. Its twiddle table is built from roots of unity using only `+ - * sqrt` (no
`sin`/`cos`), and the butterflies are only `+ - *`; `sqrt` is IEEE-754 correctly-rounded (identical
on every platform) and `+ - *` do not reassociate under `Strict`, so there is no architecture- or
library-dependent rounding anywhere in the path. This is what a deterministic lockstep sim needs.

The **no-workspace path** (`fft`/`ifft`/`rfft`/`irfft` without `ws`) and **`dft`/`idft`** compute
their twiddles with `math.sin`/`math.cos` on the fly. Burst only guarantees those bit-identical under
`FloatMode.Deterministic` (opt-in, 64-bit only) — under `Strict` they are *not* cross-architecture
reproducible. If you need determinism, use the workspace path.

## Performance

The transforms use an in-place mixed-radix (radix-4/2) core. The twiddle-table workspace is ~1.3–1.9×
faster than the no-workspace path but must be built once (see *Which one to use* above).

Ryzen 9 9950X3D, single-thread Burst, median of 9. N=1,048,576 (2²⁰,
`Benchmarks/FFTBenchmark.cs`); this size is memory-bandwidth-bound, so absolute ms varies a few % with
machine memory traffic:

| Path | dtype | med(ms) |
|---|---|---|
| `fft` (no workspace, in-place) | float | 24.39 |
| `fft` (no workspace) | double | 25.20 |
| `fft(ws)` (twiddle-table workspace) | float | 12.91 |
| `fft(ws)` | double | 18.55 |
| `rfft` (real input, no workspace) | float | 17.87 |
| `rfft` (no workspace) | double | 19.22 |
| `rfft(ws)` (twiddle-table workspace) | float | 11.27 |
| `rfft(ws)` | double | 12.95 |
