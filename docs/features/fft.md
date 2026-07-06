# FFT / DFT

`FFT`. 1D transforms over **split real/imaginary** arrays (`floatN re`, `floatN im`) — there is no
complex type. Convention: forward `X[k] = Σ x[n]·exp(-2πi·kn/N)` (no forward scaling); the inverse
divides by N, so `ifft(fft(x)) == x`. A deeper usage write-up (workspace tradeoffs, accuracy notes)
lives in [docs/fft.md](../fft.md) — written before the `FFT`/`floatFFTCache` naming landed, so read
past `floatFFT_OP`/`floatFFT_WS` as the same classes under their current names.

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

## Performance

The transforms use an in-place mixed-radix (radix-4/2) core. The twiddle-table workspace is ~1.3–1.9×
faster than the no-workspace path but must be built once — see [docs/fft.md](../fft.md) for when each
is the right default.

Ryzen 9 9950X3D, single-thread Burst, median of 9. N=1,048,576 (2²⁰,
`Benchmarks/FFTBenchmark.cs`); this size is memory-bandwidth-bound, so absolute ms varies a few % with
machine memory traffic:

| Path | dtype | med(ms) |
|---|---|---|
| `fft` (no workspace, in-place) | float | 23.52 |
| `fft` (no workspace) | double | 25.12 |
| `fft(ws)` (twiddle-table workspace) | float | 12.58 |
| `fft(ws)` | double | 16.32 |
| `rfft` (real input, no workspace) | float | 18.28 |
| `rfft` (no workspace) | double | 18.72 |
| `rfft(ws)` (twiddle-table workspace) | float | 9.83 |
| `rfft(ws)` | double | 11.76 |
