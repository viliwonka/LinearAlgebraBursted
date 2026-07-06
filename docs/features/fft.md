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

## Benchmarks

Single-thread, this machine, `Burst IJob.Run`, each row a distinct optimization (cited by commit):

| Change | Size | Result | Source |
|---|---|---|---|
| DFT: reduce twiddle argument mod N | N=2048, float | 287 → 95.3ms (3.0×), float now ≤ double | `b46bf7c` |
| rfft: real two-for-one packing vs. full `fft` | N=1M, float | 24.0 → 15.9ms (1.5×); ~1.7× at 64K–256K | `dc3bd3f` |
| Radix-4, in-place mixed-radix rewrite | — | 1.7–2.0× vs. the recursive version | `7d81eed` |
| Radix-4 wired into rfft/irfft's inner FFT | large N | 1.6–2.0× | `103c720` |
| Zero-alloc radix-4 recurrence (no-workspace path) | — | ~1.35× vs. the old radix-2 recurrence | `6bc2adb` |

Workspace-vs-no-workspace (~1.3–1.9×) is a design tradeoff, not a bug fix — see
[docs/fft.md](../fft.md) for when each is the right default.

Current absolute numbers, N=1,048,576 (2²⁰, `Benchmarks/FFTBenchmark.cs`). AMD Ryzen 9 9950X3D,
single CCD pinned (non-V-Cache), median of 9, 2026-07-06 (consolidated `AllBenchmarks` run), Unity
Editor batchmode (checks likely on). N=1M FFT is memory-bandwidth-bound, so absolute ms drifts a few
% with machine memory traffic between runs:

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
