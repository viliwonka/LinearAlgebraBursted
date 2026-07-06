# Spec: vector/matrix generators (easings, kernels, windows, wavetables)

Procedural builders for vectors/matrices. Goal: **simple, one-call, games-friendly.**

## Conventions (match the rest of the library)
- **Two forms each** (like the Stats ref overloads): a zero-alloc primitive `fProxyGen_OP.xxx(ref fProxyN dest, …)` that fills a caller vector (length taken from `dest.N`), and an ergonomic `arena.fProxyXxx(n, …)` extension that allocates + returns. Use the ref form in per-frame loops.
- **Functors are the Burst "lambda"** — reuse the existing `IfProxyScalarFunction { fProxy Eval(fProxy x); }` (same one the optimizers use). No managed lambdas in jobs.
- **fProxy-only** (float/double). Kernels are **normalized to sum 1**. Easings map **t∈[0,1]**.

---

## 1. linspace / arange  — the axis
```
fProxyGen_OP.linspace(ref dest, a, b)        // dest[i] = a + (b-a)*i/(N-1);  N=1 → {a}
fProxyGen_OP.arange  (ref dest, start, step) // dest[i] = start + i*step
arena.fProxyLinspace(a, b, n) / arena.fProxyArange(start, step, n)   // allocating
```
`linspace(0,1,N)` is the canonical input to `sample`. Example: `var t = arena.fProxyLinspace(0f, 1f, 64);`

## 2. sample\<F\>  — fill from any curve
```
fProxyGen_OP.sample<F>(ref F f, ref dest, t0 = 0, t1 = 1) where F : struct, IfProxyScalarFunction
    // dest[i] = f.Eval(t0 + (t1-t0)*i/(N-1))
arena.fProxySample(ref f, n, t0, t1)        // allocating
```
This is `linspace` piped through a functor — the whole point. Works with the built-in easings/waves **or the user's own struct**.
Example (precompute a tween LUT once, sample cheaply per frame):
```
var lut = arena.fProxyEasingLUT(new Easing.SmoothStep(), 64);   // = fProxySample over [0,1]
float y = lut[(int)(t * 63)];
```

## 3. Easing functor library  (`namespace Easing`, each a tiny struct : IfProxyScalarFunction)
All map t∈[0,1]→[0,1] (back/elastic overshoot). Usable standalone (`new Easing.SmoothStep().Eval(0.3f)`) or via `sample`.
```
Linear  SmoothStep  SmootherStep
EaseIn/Out/InOut × { Quad, Cubic, Quart, Sine, Expo }
EaseOutBounce   EaseInElastic / EaseOutElastic   EaseInBack / EaseOutBack   // overshoot
```
Convenience: `arena.fProxyEasingLUT(ref ease, n)` == `fProxySample(ref ease, n, 0, 1)`.

## 4. Wavetables  (`namespace Wave`, periodic functors with optional fields)
```
Wave.Sine { Cycles=1, Phase=0 }   Wave.Saw   Wave.Square { Duty=0.5 }   Wave.Triangle
```
`Eval(t)` over t∈[0,1] = `Cycles` periods. Build a table: `var tbl = arena.fProxySample(new Wave.Sine{Cycles=1}, 256, 0, 1);`
(Range is [-1,1]; use for oscillators/LFOs. Any `sample` range works.)

## 5. Kernels  (normalized, symmetric, centered)
```
fProxyGen_OP.gaussianKernel (ref dest, sigma)   // 1D, dest[i]=exp(-(i-c)²/2σ²) then ÷Σ ;  c=(N-1)/2
fProxyGen_OP.boxKernel      (ref dest)          // 1D uniform 1/N
fProxyGen_OP.tentKernel     (ref dest)          // 1D triangular, ÷Σ
fProxyGen_OP.gaussianKernel2D(ref destMat, sigma)   // N×N separable = outer(g,g), ÷Σ (reuses outerDot)
arena.fProxyGaussianKernel(n, sigma) / arena.fProxyBoxKernel(n) / arena.fProxyTentKernel(n)
arena.fProxyGaussianKernel2D(n, sigma)
```
Use: blur / smoothing weights / convolution. The 2D Gaussian is the outer product of the 1D one (separable).

## 6. DSP window functions  (index-based, depend on N — single enum entry point)
```
enum WindowType { Hann, Hamming, Blackman, Box }
fProxyGen_OP.window(ref dest, WindowType)        // dest[i] = w(i, N)
arena.fProxyWindow(n, WindowType)               // allocating
```
Formulas (i over 0..N-1): Hann `0.5(1−cos 2πi/(N−1))`, Hamming `0.54−0.46 cos(...)`, Blackman `0.42−0.5cos(...)+0.08cos(4πi/(N−1))`, Box `1`.
Use: pre-FFT windowing (pairs with the FFT candidate), tapering.

## 7. 2D generators  (from curves & 2D functions)
The 2D analog of `sample`. **One new interface** + one core; gradients/rank-1/diagonals all fall out of it.
```
interface IfProxyBivariateFunction { fProxy Eval(fProxy x, fProxy y); }   // the 2D "lambda"

fProxyGen_OP.sample2D<F>(ref F f, ref destMat, x0,x1, y0,y1)
    // M[i,j] = f.Eval( lerp(x0,x1, i/(M_Rows-1)),  lerp(y0,y1, j/(N_Cols-1)) )    — meshgrid + eval
arena.fProxySample2D(ref f, rows, cols, x0,x1,y0,y1)                              // allocating
```
**Rank-1 / two curves (separable, fast):**
```
fProxyGen_OP.outer   (in u, in v, ref destMat)   // M[i,j] = u[i]*v[j]   (reuses outerDot)
fProxyGen_OP.outerSum(in u, in v, ref destMat)   // M[i,j] = u[i]+v[j]   (additive fields)
```
`gaussianKernel2D` = `outer(g, g)`. Any separable field = `outer(sampleU, sampleV)`.

**Built-in bivariate functors (`namespace Field2`, each : IfProxyBivariateFunction) — gradients & fields:**
```
Field2.PlaneX (→x)   PlaneY (→y)   Diagonal (→x+y)   Radial { Cx,Cy } (→√((x-Cx)²+(y-Cy)²))
Field2.Gaussian2D { Sigma }   Ripple { Freq }(=sin(Freq·r))   Checker { Sx,Sy }
```
So a radial gradient is `sample2D(new Field2.Radial(), …)`; a horizontal gradient is `PlaneX`; a height field is the user's own struct. **Diagonal from a curve** needs no new API — `arena.fProxyDiagonalMat(arena.fProxySample(ref ease, n))`.

> Separable (`outer` of two 1D samples) is O(rows·cols) with two cheap 1D passes; full `sample2D` evaluates the functor per cell. Use `outer` when the field factorizes, `sample2D` otherwise.

## 8. Future: Fourier (FT / FFT)  — flagged, separate effort
Pairs with the windows (§6) and wavetables (§4). Plan when built:
- **`dft` / `idft`** — direct O(N²), exact, ANY N. Simple, good baseline & for small N.
- **`fft` / `ifft`** — radix-2 Cooley–Tukey, O(N log N), N a power of 2.
- **No complex TYPE** — use split real/imag arrays (the `Eigen.valuesQR` precedent): `fft(ref re, ref im)` in-place, or `rfft(in real, ref re, ref im)` for real input. Helpers: `magnitude`/`phase`/`powerSpectrum` (re,im → vector).
- Typical games/DSP pipeline: rolling-window samples → `window(Hann)` → `rfft` → `powerSpectrum` (beat/pitch/feature detection). Own spec doc when scheduled.

---

## Placement
- `fProxyGen_OP` (new static class) — the ref-dest primitives + `sample<F>` / `sample2D<F>` / `outer`.
- `Arena` extensions (`ArenaExtensions.fProxy.cs`) — the allocating `fProxyXxx` wrappers, beside `fProxyIdentityMat`/`fProxyDiagonalMat`.
- `IfProxyBivariateFunction` — new, beside `IfProxyScalarFunction` (Optimize.fProxy.cs).
- `Easing` / `Wave` (univariate) and `Field2` (bivariate) — small structs implementing the functor interfaces.

## Suggested v1 (compounding, smallest→most useful)
1. **`linspace` + `sample<F>` + Easing lib** — the elegant core; tween LUTs, curves, user functors. Reuses `IfProxyScalarFunction`.
2. **`sample2D<F>` + `outer` + Field2 (gradients/radial)** — 2D fields, gradients, separable kernels. One new interface; gradients & rank-1 fall out.
3. **Gaussian/box/tent kernels** (1D + 2D=outer(g,g)) — blur/smoothing.
4. **Wavetables + DSP windows** — oscillators + pre-FFT.
5. **FT/FFT** (§8) — its own spec; consumes windows.

Tests: known-value oracles (`linspace(0,1,5)=={0,.25,.5,.75,1}`, `SmoothStep(0.5)=0.5`, gaussian sums to 1 & symmetric, Hann endpoints=0, `outer(u,v)[i,j]==u[i]*v[j]`, `sample2D(PlaneX)` rows equal), + `sample`==manual-loop equivalence.
