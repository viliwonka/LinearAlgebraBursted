# Generate

Procedural vector/matrix builders. Each has a zero-alloc `ref dest` primitive (`Generate.xxx`) and an ergonomic allocating `arena.floatXxx(...)`
wrapper. 

- **Linear**:
  - `linspace(ref dest, a, b)`, 
  - `arange(ref dest, start, step)`
- **Functor sampling**: 
  - `sample<F>(ref F f, ref dest, t0, t1) where F : IfloatScalarFunction` lambda pattern, evaluates any easing/wave functor over a domain
- **Easing** 
  - `Linear`, 
  - `SmoothStep`/`SmootherStep`, 
  - `EaseIn/Out/InOutQuad/Cubic/Quart/Sine/Expo`, 
  - `EaseInBounce/EaseOutBounce/EaseInOutBounce`, 
  - `EaseIn/Out/InOutElastic`,
  - `EaseIn/Out/InOutBack` - each a tiny struct functor mapping `t ∈ [0,1] → [0,1]`
- **Wave / LFO**:
  - `Sine`/`Saw`/`Square`/`Triangle`, output range `[-1,1]`, `t ∈ [0,1]` . 
- **Kernels & windows** - 
  - Kernels: `gaussianKernel`,`boxKernel`,`tentKernel`, 
  - Window: `Box`,`Hann`,`Hamming`,`Blackman`.
- **Rank-1 builders** - `outer(in u, in v, ref floatMxN dest)`, `outerSum(in u, in v, ref dest)`.

## Performance

Not benchmarked — these are setup-time/low-frequency builders, not hot-loop kernels.
 