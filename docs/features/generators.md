# Generators

Procedural vector/matrix builders. Design doc: [spec-generators.md](../spec-generators.md). Each has
a zero-alloc `ref dest` primitive (`Generate.xxx`) and an ergonomic allocating `arena.floatXxx(...)`
wrapper (`ArenaExtensions.Generators.float.cs`) — use the `ref` form in per-frame loops.

- **Linear** — `Generate.linspace(ref dest, a, b)`, `arange(ref dest, start, step)`.
- **Functor sampling** — `Generate.sample<F>(ref F f, ref dest[, t0, t1]) where F : IfloatScalarFunction`
  evaluates any easing/wave functor over a domain — the "Burst-native lambda" pattern used by both
  families below.
- **Easing** — `floatEasing`: `Linear`, `SmoothStep`/`SmootherStep`, `EaseIn/Out/InOutQuad/Cubic/
  Quart/Sine/Expo`, `EaseInBounce/EaseOutBounce/EaseInOutBounce`, `EaseIn/Out/InOutElastic`,
  `EaseIn/Out/InOutBack` — each a tiny struct functor mapping `t ∈ [0,1] → [0,1]` (Back/Elastic
  overshoot beyond that range).
- **Wave / LFO** — `floatWave`: `Sine`/`Saw`/`Square`/`Triangle{Cycles, Phase[, Duty]}`, output range
  `[-1,1]`, `t ∈ [0,1]` spans `Cycles` periods. `Cycles = 0`/`Duty = 0` are silently treated as
  defaults (1 / 0.5) — you can't request a literal zero.
- **Kernels & windows** — `gaussianKernel`/`boxKernel`/`tentKernel(ref dest, ...)` (normalized, sum =
  1), `gaussianKernel2D` (separable outer of the 1D kernel), `window(ref dest, WindowType)`
  (`Box`/`Hann`/`Hamming`/`Blackman`).
- **Rank-1 builders** — `outer(in u, in v, ref floatMxN dest)` (thin wrapper over
  [`Blas.outerDot`](blas.md)), `outerSum(in u, in v, ref dest)`.

## Benchmarks

Not benchmarked — these are setup-time/low-frequency builders, not hot-loop kernels.
