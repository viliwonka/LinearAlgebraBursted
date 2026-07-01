# FFT / DFT — usage

1D Fourier transforms over **split real/imaginary** arrays (`fProxyN re`, `fProxyN im`) —
there is no complex type. Available for `float` and `double` via `floatFFT_OP` / `doubleFFT_OP`.

Convention: forward `X[k] = Σ x[n]·exp(-2πi·kn/N)` (no forward scaling); the inverse divides by N,
so `ifft(fft(x)) == x`.

## Entry points

| Method | Notes |
|---|---|
| `fft(ref re, ref im)` · `ifft(…)` | no-workspace, in-place, **zero-alloc**. Power-of-two only. |
| `fft(ref re, ref im, in ws)` · `ifft(…)` | workspace (plan), **fastest**. Power-of-two only. |
| `rfft(in real, ref re, ref im [, in ws])` · `irfft(…)` | real input → half spectrum (length N/2+1). |
| `dft(in inRe, in inIm, ref outRe, ref outIm)` · `idft(…)` | direct O(N²), works for **any N**. |
| `arena.fProxyMagnitude / fProxyPowerSpectrum / fProxyPhase(in re, in im)` | spectrum post-processing. |

## Which one to use

All FFT/IFFT/rfft lengths must be a **power of two** — use `dft` for arbitrary N.

- **One-shot / a few transforms → no-workspace `fft(re, im)`.** Table-free and zero-allocation.
  Internally it dispatches to a radix-4 recurrence for power-of-4 lengths and a radix-2 recurrence
  otherwise. Nothing to set up.
- **Many transforms of the same size → build a workspace once and reuse it.** It auto-dispatches
  radix-4 (power-of-4) or mixed-radix (other powers of two), ~1.3–1.9× faster than the no-ws path.
  The twiddle-table build amortizes after only ~1–3 transforms, so this is the right default for
  any repeated use.

```csharp
var ws = arena.floatFFT_WS(1024);     // builds the twiddle table on creation
for (int f = 0; f < frames; f++)
    floatFFT_OP.fft(ref re, ref im, in ws);    // zero-alloc, reuses the plan
```

## Workspace notes

- Built on creation by the arena factory (`arena.floatFFT_WS(n)`); disposed with the arena —
  no manual `Dispose`. This matches every other workspace in the library (`floatSVD_WS`, …).
- Holds the twiddle tables plus the rfft/mixed-radix scratch, so repeated `fft/ifft/rfft/irfft(ws)`
  allocate nothing.
- **Single-use-at-a-time**: the scratch is shared, so use one workspace per thread for parallel
  transforms (FFTW "plan" semantics).
- Sized for exactly `n`; pass data with `re.N == n` (or `real.N == n` for `rfft`).

## rfft / irfft

- `rfft` packs N real samples and returns the half spectrum (`re`/`im` of length N/2+1);
  `im[0]` and `im[N/2]` are always zero (DC and Nyquist are real).
- `irfft` inverts it back to N real samples: `irfft(rfft(x)) == x`.

## Notes

- The workspace path builds its twiddle table at double precision (accurate, no recurrence drift).
  The no-ws recurrence accumulates a small twiddle drift at very large `float` N — negligible for
  typical use; prefer the workspace if you need maximum accuracy on huge float transforms.
- Validated by a stability suite: fft-vs-dft cross-check, Parseval energy, linearity, analytic
  transforms (impulse/constant/exponential/shift), rfft↔fft consistency, and large-N round-trips.
