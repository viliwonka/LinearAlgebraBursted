# Random

`Rand`. Sampler-struct pattern: `IfloatSampler { float Next(ref Random rng); }`, implemented by
`floatUniform`/`floatExponential`/`floatRayleigh`/`floatWeibull`/`floatCauchy`/`floatLogistic`/
`floatPareto`/`floatTriangular`/`floatGaussian` (stateful Box-Muller — pass `ref`, it caches a spare
value between calls).

## Filling vectors/matrices

- `Rand.randomInPlace<S>(ref Random rng, ref floatN dest, ref S sampler)` — generic over any sampler
  struct (vector and matrix overloads).
- `Rand.nextUniformInPlace(ref Random, ref dest[, min, max])` — uniform shortcut.
- Integer/bool refill: `nextUniformInPlace(ref Random, ref intN dest, min, max)`,
  `nextBernoulliInPlace(ref Random, ref boolN dest, float p)`, `nextBoolInPlace` (fair coin).

## Picking & shuffling

`weightedPick(in weights, ref rng) : int`, `weightedPickInPlace(in weights, ref Indices dest, ref
rng)`, `randomPermutationInPlace(ref Pivot, ref rng)`, `shuffleInPlace(ref Indices, ref rng)`,
`sampleKWithoutReplacementInPlace(ref Indices dest, int n, ref rng)`.

## Multivariate normal & structured matrices

`multivariateNormalInPlace(ref rng, in cholL, in mean, ref dest)` (+ row-batch
`multivariateNormalRowsInPlace`) — caller factors Σ once via `CHO.decomp`, then
draws as many samples as needed. Property-matrix generators (all validate before allocating Temp
scratch): `randomOrthogonalInPlace` (Haar-uniform, Mezzadri sign-fixed QR),
`randomSpdInPlace(..., minEig, maxEig)`, `randomMatrixWithConditionInPlace(..., cond)`,
`randomMatrixWithRankInPlace(..., rank)`, `randomStochasticInPlace`.

## Benchmarks

Not benchmarked. One known, template-constrained inefficiency: Box-Muller Gaussian sampling calls
`math.sin` and `math.cos` separately instead of `math.sincos` (which computes both for the cost of
one evaluation) — the codegen template mechanism doesn't currently support an `out`-parameter method
across the proxy substitution, so this is left as-is (roughly doubles the trig cost of Gaussian fills
specifically; every other sampler is unaffected).
