# Hash

`Hash` — xxHash32 (public-domain, Yann Collet), chosen over ad-hoc mixing for a well-specified,
avalanche-vetted, Burst-friendly reference implementation. All hashes return `uint`.

- `Hash.hash(in floatN v, uint seed = 0) : uint` / `hash(in floatMxN v, uint seed = 0)`.
- `rowHashes(in A, ref uintN dest, uint seed = 0)` (+ allocating overload); `colHashes(...)` is slower
  (strided gather-then-hash per column — hashing is naturally row-major-friendly, same access-pattern
  concern as everywhere else in the library).
- `Hash.combine(uint a, uint b)` — order-sensitive fold, for combining hashes of several buffers into
  one checksum.

## Caveats (documented on the class itself)

This is a **bit-exact** hash, not a value-equality hash:

- `-0.0f` and `+0.0f` compare equal (`==`) but hash differently — they differ only in the sign bit,
  and the hash reads raw bits, not IEEE value.
- Two `NaN`s with different bit payloads hash differently, even though all NaNs compare unequal to
  everything (including each other) under IEEE 754.

The flagship use case is **lockstep-multiplayer desync detection**: hash each client's authoritative
state every tick and compare checksums across the network. There, bit-exact behavior is *correct* —
a real divergence (even one only in a sign bit or NaN payload) is exactly what you want caught, not
masked by value-equality. See [determinism](../../README.md#determinism) for the Burst `FloatMode`
side of making that state reproducible in the first place.

## Performance

Not benchmarked.
