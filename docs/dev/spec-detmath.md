# Spec — DetMath: in-house cross-platform-deterministic transcendentals

Status: draft. Internal dev spec (not shipped). Builds on the FFT twiddle precedent
(`sqrt`-based deterministic roots of unity) and the determinism survey in the
`stats-random-determinism` memory.

## 1. Goal

Provide `sin`, `cos`, `exp`, `log`, `atan` (and everything derived from them) as functions
that are **bit-identical across CPU architectures** for a fixed set of coefficients, so that
every feature depending on a transcendental can be made cross-platform deterministic — the same
lockstep-simulation guarantee the core factorizations and the workspace FFT already carry.

Today `Unity.Mathematics.math.sin/cos/exp/log/pow/...` compile to platform-specific
implementations (hardware instructions or a vendored math library). `+ - * / sqrt` are IEEE-754
correctly-rounded and therefore reproducible on every platform; transcendentals are **not**
required to be correctly rounded (table-maker's dilemma), so each CPU/mathlib/Burst version ships
its own ~0.5–1 ULP approximation and those differ across architectures. Burst only makes the
intrinsics bit-identical under `FloatMode.Deterministic`, which is opt-in **and double-only** — so
`float` transcendentals cannot be made deterministic that way at all.

## 2. Determinism principle

A transcendental evaluated as a **fixed sequence of IEEE-754 `+ - * / sqrt` operations on
baked constants** is deterministic across architectures *by construction*: every platform runs the
identical rounding steps and gets identical bits. It does **not** need to be correctly rounded —
it only needs to be the *same computation everywhere*.

Hard requirements for every DetMath routine:

- Only `+ - * / sqrt` and integer/bit ops (for `frexp`/`ldexp`-style exponent handling). No call
  to any platform libm / `math.sin` etc. anywhere in the path.
- **No FMA contraction and no reassociation.** Under Burst `FloatMode.Default` (which *is* `Strict`)
  `a*b+c` is two rounding steps and expression order is preserved — this is what we rely on. FMA and
  reassociation only appear under `FloatMode.Fast`, which DetMath call sites must never use. An
  *explicit* `math.fma` is deterministic (single defined op), but we avoid it to keep the two-rounding
  model uniform and portable; if used it must be used identically on every path.
- Any lookup table must be a **baked source constant** computed offline at high precision. Building a
  table at runtime with the platform libm re-introduces the exact divergence we are removing. (The FFT
  twiddle table is runtime-built but only from `sqrt`, hence exempt.)
- SIMD is allowed and encouraged: lane-wise `+ - * /` is bit-identical to scalar under Strict
  (Strict forbids *reassociation*, not *vectorization*). No horizontal/cross-lane reductions inside a
  routine.

## 3. Scope

**Core primitives (5):** `exp`, `log`, `sin`, `cos`, `atan`.

**Derived (compose from the core, no new approximations):**
- `pow(x,y) = exp(y·log x)` (with sign/zero/edge handling), `exp2/exp10/log2/log10` = scaled `exp`/`log`
- `tan = sin/cos`, `sincos` = one reduction feeding both
- `asin/acos/atan2` via `atan`
- `sinh/cosh/tanh` via `exp`
- `expm1`/`log1p` want their own small polynomials for accuracy near 0 (Phase 3)

**Already deterministic — explicitly OUT of scope:** `sqrt` (IEEE correctly-rounded), `rcp = 1/x`,
`rsqrt = 1/sqrt(x)` (Unity.Mathematics defines them as plain `/` and `sqrt`; only `FloatMode.Fast`
substitutes the approximate RCPPS/RSQRTPS, which the library never uses). Uniform PRNG (integer
xorshift) is already deterministic; only the uniform→distribution transforms need DetMath.

**Consumers (survey: 85 transcendental calls across 14 template files):**
- **`UnsafeMathOP` / `Comp.*`** — user-facing element-wise `sin cos tan asin acos atan atan2 sinh
  cosh tanh exp exp2 log log2 log10 pow` (22 calls). Highest value: one wide DetMath gives every
  downstream consumer both determinism and SIMD.
- **`RandomOP` samplers** — Exponential/Rayleigh (`log`), Weibull/Pareto (`log`,`pow`), Cauchy
  (`tan`), Logistic (`log`), Normal/Box-Muller (`log`,`sin`,`cos`).
- **`StatsCore` softmax / softmaxRows / softmaxCols** (`exp`) — the only Stats offender.
- **FFT `dft`/`idft`** — arbitrary-N twiddles (`sin`,`cos`,`atan2`). The workspace FFT is already
  deterministic; `dft` is the last transcendental holdout after the no-workspace FFT removal.
- **`GenOP` windows/kernels** (`exp`,`cos`), **`Easing`** (`sin`,`cos`,`pow`), **`Wave`** (`sin`),
  **`ArenaExtensions` rotation** (`sin`,`cos`), **`Gallery.Special`**, **`Analysis` log-det** (`log`),
  robust-loss residual (`log`), **`UnsafeOP` Lp norm** (`pow`), cond-number generator (`pow`).

Note: dropping niche probability models does NOT shrink DetMath — `exp/log/sin/cos` are each already
required by non-sampler features. Samplers are just consumers.

## 4. Approach decision

Two license-clear routes (both verified non-copyleft, unlike the GPL-tainted `ladBR`/`ladFN`):

- **RLIBM-ALL (Rutgers, MIT license).** Coefficients generated so the polynomial's result is the
  *correctly-rounded* nearest float for all inputs → bit-identical across platforms by definition, at
  libm-grade (0-ULP for `float`) accuracy, ~1.05× glibc cost. LLVM libc has adopted its
  Log/Log2/Log10/Exp/Exp2. Best when we want maximum accuracy and the exact analytic inverse-CDF
  (heavy tails intact). arXiv 2108.06756; github rutgers-apl/rlibm-all.
- **SLEEF (Boost Software License 1.0).** Vectorized reference with 1-ULP and 3.5-ULP variants;
  straightforward to port `xexp/xlog/xsin/xcos/xatan` into a wide `fProxyW` form. A few ULP is plenty
  for sampling and element-wise math.

**Recommendation:** start from **RLIBM** coefficients for `float` `exp`/`log` (0-ULP, drop-in
determinism, cleanest correctness story) and **SLEEF-style** minimax polynomials for `sin`/`cos`/`atan`
and the `double` variants, evaluated with explicit Horner/Estrin under Strict. Accept a few-ULP target
where correctly-rounded coefficients aren't readily available. Bake all coefficients as `const`
source; add a Third-Party-Notices line (same treatment as the settled HiGHS/MIT dependency).

## 5. API surface

New static class `DetMath` in `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DetMath.fProxy.cs`,
codegen-templated `fProxy → float/double`. Contract-only XML docs (accuracy budget + domain per fn).

- Scalar: `DetMath.Sin(fProxy x)`, `Cos`, `SinCos(x, out s, out c)`, `Exp`, `Log`, `Atan`, plus the
  derived `Pow/Tan/Atan2/Asin/Acos/Exp2/Log2/...`.
- Wide (SIMD): `DetMath.Sin(fProxyW x)` etc., pure `fProxyW` `+ - * /` chains (Estrin regroup for ILP),
  bit-identical to the scalar path.
- Coefficient tables live per precision (float vs double get different minimax coefficients) — use the
  existing shared-type / `//+choose[...]` codegen mechanisms, never hand-duplicate the two dtypes.

DetMath is the mechanism. Whether a consumer *routes through it by default* vs *behind an opt-in* is a
per-consumer product decision (§9), not part of the primitive.

## 6. Algorithm sketches (per primitive)

Standard argument-reduction + polynomial, all reductions in exact arithmetic:

- **`exp`**: reduce `x = k·ln2 + r`, `|r| ≤ ln2/2`, via `k = round(x/ln2)`; `ln2` split hi/lo
  (Cody-Waite) so `x − k·ln2_hi − k·ln2_lo` is computed without cancellation error; evaluate
  `exp(r)` by minimax poly; scale by `2^k` through direct exponent-bit assembly (`ldexp` as integer
  bit ops, deterministic). Handle overflow→+inf, underflow→0, subnormal boundary.
- **`log`**: `x = m·2^e`, `m ∈ [1,2)` via exponent/mantissa bit split (`frexp`); reduce `m` around 1
  (`s = (m−1)/(m+1)`, series in `s²`) or a minimax poly; `log x = e·ln2 + poly`, `ln2` split hi/lo.
  Handle `x ≤ 0` → NaN, `x = 0` → −inf, `x = +inf` → +inf.
- **`sin`/`cos`/`sincos`**: reduce `x = k·(π/2) + r`, `|r| ≤ π/4`; `π/2` split into several exact
  hi/mid/lo parts (Cody-Waite; full Payne-Hanek only if we must support very large arguments — see
  Risks); select sin/cos poly by `k mod 4` with the right sign. Odd/even minimax polys in `r²`.
- **`atan`**: fold to `[0,1]` via `atan(x)=π/2−atan(1/x)` for `|x|>1` and sign symmetry; minimax poly
  in `x²`. `atan2` from `atan` + quadrant logic. `asin/acos` via `atan` of the half-angle form.

## 7. SIMD & sampler integration

- Polynomials are pure `+−×` chains → vectorize with `fProxyW` exactly like the FFT butterfly / GEMV
  reductions. Throughput, not latency: a wide DetMath shines on **array/batch** work.
- **Element-wise `Comp.*`** consumers are already array-shaped → wide DetMath drops in directly and
  gives both determinism and a speedup over scalar `math.*`.
- **Samplers**: prefer a **batch/array-fill API** (`Rand.normal(destSpan)`) so all 8 lanes are used; a
  one-at-a-time `normal()` leaves 7 lanes idle. If we keep the scalar signature, carry an 8-lane buffer
  + cursor in the RNG state and refill via one wide compute every 8th call (numpy PCG / MKL VSL
  pattern) — deterministic because the refill pre-draws uniforms in a fixed order.
- Prefer **branch-free Box-Muller / inverse-CDF** over **ziggurat** for the wide path: ziggurat's
  per-lane rejection breaks lane-lockstep. Box-Muller is branch-free (8 uniforms → `log/sqrt/sin/cos`
  → normals) and reuses the wide DetMath.

## 8. Testing

- **Accuracy**: golden vectors generated OFFLINE at high precision (e.g. mpmath/`System.Math` double
  reference), checked into `SourceTests`, compared within a documented per-function ULP budget. Cover
  edge cases: 0, ±inf, NaN, subnormals, reduction boundaries (`±π/4·k`, `k·ln2`), large arguments,
  monotonic segments, and identities (`sin²+cos²=1`, `exp(log x)=x`, `log(exp x)=x`) within tolerance.
- **Determinism-by-construction audit**: a source check that DetMath calls no libm/`math.` transcendental
  and no `FloatMode.Fast` — the guarantee is structural, not something we can prove by running on one arch.
- **Perf**: A/B native `math.*` vs DetMath (scalar and wide) on both throughput and ULP, per the
  "bench both perf and accuracy" rule. Wide DetMath should beat scalar `math.*` on batch element-wise.

## 9. Migration

- **`dft`/`idft`**: route to `DetMath.Sin/Cos` → the arbitrary-N FFT fallback becomes deterministic,
  closing the last FFT transcendental gap. (Removes the "dft is not cross-arch deterministic" caveat
  added when the no-workspace FFT was deleted.)
- **`softmax`**: route to `DetMath.Exp`.
- **Samplers**: route through DetMath; decide default-on vs opt-in. Recommendation: make the DetMath
  path the default once accuracy is validated (samplers don't need libm-grade accuracy), so
  determinism is the out-of-the-box behavior consistent with the rest of the library.
- **`Comp.*` element-wise**: the highest-value migration; gives users deterministic + SIMD transcendentals.
  Likely default-on with the old `math.*` behavior gone (breaking, cheap pre-1.0) OR a documented switch.
- Update the README determinism section and `docs/features/*` once the deterministic paths land.

## 10. Phasing

1. **Phase 1 — `exp`, `log`, `sin`, `cos` (double then float), scalar + wide.** Unblocks softmax,
   `dft`/`idft`, Normal/Exponential samplers, windows/kernels. Highest value; ship first.
2. **Phase 2 — `atan` + derived `pow`, `tan`, `atan2`, `asin`, `acos`, `exp2/log2/...`.** Completes the
   `Comp.*` element-wise surface and the remaining samplers (Cauchy/Weibull/Pareto/Logistic).
3. **Phase 3 — `sinh/cosh/tanh`, `expm1`, `log1p`, `erf`.** For future NN activations / stats CDFs; and
   the batch sampler API + buffered-RNG SIMD path.

## 11. Risks & open questions

- **Large-argument range reduction.** Cody-Waite hi/lo splits are accurate to moderate magnitudes;
  `sin(1e20)` needs Payne-Hanek (many bits of 2/π) or a double-double reduction — heavy. Decide the
  supported domain (document a max `|x|` for full accuracy, or invest in Payne-Hanek). Most consumers
  (FFT twiddles, sampler transforms, windows) have small, well-bounded arguments.
- **Default-on vs opt-in per consumer** — product decision (accuracy vs libm parity vs determinism).
- **Per-function accuracy target** — 0-ULP (RLIBM) where available vs a few ULP (SLEEF) elsewhere; pin
  and document each.
- **Licensing** — add a Third-Party-Notices entry for RLIBM (MIT) and/or SLEEF (Boost); both are
  non-copyleft and safe, but attribution is required.
- **Coefficient provenance** — bake coefficients from the chosen reference; record source + generation
  method in the DEVLOG so they can be regenerated.

## 12. References

- Sorensen, Jones, Heideman, Burrus, "Real-valued fast Fourier transform algorithms," IEEE ASSP 1987
  (context for the parallel real-FFT determinism story).
- RLIBM: Lim & Nagarakatte, "One Polynomial Approximation to Produce Correctly Rounded Results...,"
  arXiv 2108.06756; github rutgers-apl/rlibm-all (MIT).
- SLEEF: Shibata, "SLEEF: A Portable Vectorized Library of C Standard Mathematical Functions"
  (Boost Software License 1.0).
- Cody & Waite, *Software Manual for the Elementary Functions* (argument reduction).
- Payne & Hanek, "Radian reduction for trigonometric functions" (large-argument reduction).
- The `stats-random-determinism` memory and the FFT.Workspace DEVLOG (sqrt-based deterministic twiddle
  precedent) in this repo.
