# Select & bit operations

Two small, unrelated primitive groups that don't fit under [Comp](comp-elementwise.md) or
[Query](query.md).

## Select

`Select` — **element-wise ternary select**, not order-statistics/quickselect (the name is easy to
misread as the latter): `select(in a, in b, in boolN c, ref dest)` computes `dest[i] = c[i] ? b[i] :
a[i]` (vector and matrix overloads), plus scalar-`bool`-condition overloads that copy one whole
operand wholesale. Both a zero-alloc `ref`-dest primitive and an allocating form exist.

## Integer bit intrinsics

`intComp`/`shortComp`/`longComp`/`uintComp` add bitwise ops on top of the arithmetic set shared with
`floatComp` (see [comp-elementwise.md](comp-elementwise.md)): `bitwiseAndInPlace`/`bitwiseOrInPlace`/
`bitwiseXorInPlace`/`bitwiseComplementInPlace` (buffer and scalar forms), `bitwiseLeftShiftInPlace`/
`bitwiseRightShiftInPlace`, `rorInPlace`/`rolInPlace` (rotate), and the count/scan family
`countbitsInPlace` (popcount), `tzcntInPlace`/`lzcntInPlace` (trailing/leading zero count),
`reversebitsInPlace`, `ceilpow2InPlace` — thin forwards to `Unity.Mathematics.math`'s per-lane
intrinsics, applied element-wise across a whole `intN`/`intMxN`.

Bool logic (`boolComp`) is covered in [comp-elementwise.md](comp-elementwise.md), alongside `Comp`'s
other per-type variants.

## Benchmarks

Not benchmarked — these are thin per-element forwarders to `Unity.Mathematics.math` intrinsics with
no dedicated hot-path optimization work done.
