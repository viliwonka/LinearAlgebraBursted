# Select & bit operations

Two small, unrelated primitive groups that don't fit under [Comp](comp-elementwise.md) or
[Query](query.md).

## Select

`Select` — **element-wise ternary select**, not order-statistics/quickselect (the name is easy to
misread as the latter): `select(in a, in b, in boolN c, ref dest)` computes `dest[i] = c[i] ? b[i] :
a[i]` (vector and matrix overloads), plus scalar-`bool`-condition overloads that copy one whole
operand wholesale. Both a zero-alloc `ref`-dest primitive and an allocating form exist.

## Performance

These are thin forwarders to Unity.Mathematics intrinsics; not benchmarked or separately optimized.
