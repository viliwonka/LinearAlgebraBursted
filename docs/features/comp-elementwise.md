# Comp — element-wise operations

`floatComp`/`doubleComp`/`intComp`/`shortComp`/`longComp`/`uintComp`/`boolComp` are per-type static
classes (kept split, not merged — see [naming-style-guide](../dev/naming-style-guide.md)'s split/merge
checklist) of generic `<T> where T : unmanaged, IUnsafe*Array` methods, so one body serves both
`floatN` and `floatMxN` alike. All mutating methods use the **`InPlace`** suffix (not `Inpl`).

## float/double (`floatComp`)

- **Arithmetic:** `addInPlace`/`subInPlace`/`mulInPlace`/`divInPlace`/`modInPlace`, each with a
  `(T, float s)` scalar form and a `(T, T)` buffer form; `addScaledInPlace(y, a, x)` (axpy, `y += a·x`),
  `scaleAddInPlace(y, a, x)` (`y = a·y + x`); `clampInPlace(x, lo, hi)`; `signFlipInPlace`.
- **Math functions:** `absInPlace`, `signInPlace`, `sqrtInPlace`/`rsqrtInPlace`, the full trig set
  (`sin/cos/tan/asin/acos/atan/sinh/cosh/tanh/acosh`, all `InPlace`), `expInPlace`/`exp2InPlace`/
  `exp10InPlace`, `logInPlace`/`log2InPlace`/`log10InPlace`, `ceilInPlace`/`floorInPlace`/
  `roundInPlace`, `reluInPlace`, `powInPlace(int exponent)`, `rcpInPlace`, `fracInPlace`,
  `saturateInPlace`, `degreesInPlace`/`radiansInPlace`.
- **Two-buffer/interpolation:** `lerpInPlace(a,b,t)`, `unlerpInPlace`, `smoothstepInPlace`,
  `stepInPlace(edge)`, `madInPlace(a,b,c)`, `remapInPlace(oldMin,oldMax,newMin,newMax)`,
  `atan2InPlace(y,x)`, `minInPlace`/`maxInPlace`, `sincos(x, sin, cos)` (no `InPlace` — writes two
  separate outputs, doesn't mutate `x`).

Allocating sugar for all of the above lives on the operator overloads (`+ - * / %`) on `floatN`/
`floatMxN` — see [dense-types](dense-types.md).

## bool (`boolComp`)

Logic ops over `boolN`/`boolMxN`: `notInPlace`, `andInPlace`/`orInPlace`/`xorInPlace` (buffer and
scalar-`bool` forms), `equalsInPlace`/`notEqualsInPlace`. `Analysis.any`/`Analysis.all` reduce a
`boolN`/`boolMxN` to one bool (vacuous-truth empty semantics: `any(empty) == false`, `all(empty) ==
true`, matching `Unity.Mathematics.math.any/all`).

## Integer bit ops

`intComp`/`shortComp`/`longComp`/`uintComp` add bitwise ops beyond the float set — see
[select-bits](select-bits.md).

## Performance

Not independently benchmarked — `Comp` isn't measured standalone. The same axpy-shaped kernel
(`y += a·x`, independent across elements ⇒ full SIMD) that backs `addScaledInPlace` is also what the
vectorized LU/Cholesky/QR sweeps in [decompositions](decompositions.md) are built on.
