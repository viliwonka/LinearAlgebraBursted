# Hash

`Hash` - xxHash32 provides a well-specified, avalanche-vetted hashing function. All hashes return `uint`.

- `hash(in floatN v..)`,
- `hash(in floatMxN A..)`,
- `rowHashes(in floatMxN A..)`, 
- `colHashes(in floatMxN A..)`,

## Caveats

This is a **bit-exact** hash, not a value-equality hash:

- `-0.0f` and `+0.0f` compare equal (`==`) but hash differently - they differ only in the sign bit,
  and the hash reads raw bits, not IEEE value.
- Two `NaN`s with different bit payloads hash differently, even though all NaNs compare unequal to
  everything (including each other) under IEEE 754.

Usecase is fast comparing for multiplayer desync detection.