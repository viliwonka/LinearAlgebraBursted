using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Type-agnostic byte-stream hash kernel shared by every element-type family (float/double/
    // int/short/long/uint/bool). A single, non-templated file (mirrors OpHelpers.Shared.cs /
    // Query.Shared.cs): the per-type wrapper files (one for float/double, one for the int family,
    // one for bool - see the sibling files in this same folder) all partial-merge into this SAME
    // `Hash` class, and this kernel would otherwise have to live twice (once per merged fragment)
    // and collide as CS0111/CS0101 - so it lives here instead, emitted exactly once. See
    // docs/dev/naming-style-guide.md's "Split vs merge safety" and the identical rationale on
    // Query.Shared.cs's decodeIndex.
    //
    // NOTE FOR EDITORS: this file must never contain the literal proxy-token substrings that name
    // the sibling per-type files above (spelled out here only as "fP" + "roxy" / "iP" + "roxy" to
    // avoid the codegen parser tripping over its own trigger words) - either one appearing ANYWHERE
    // in this file's text, even inside a comment, flips TemplateConverter's singular-file detection
    // (it checks file CONTENT, not just the filename) and this file would silently get copy-mangled
    // once per int-family type instead of emitted once verbatim. Refer to the sibling files by
    // description ("the float/double file", "the int-family file") instead of by name.
    //
    // ALGORITHM CHOICE: xxHash32 (Yann Collet, public domain), picked over a Unity.Mathematics-style
    // hand-rolled multiply-xor scheme (e.g. what math.hash's internal mixing resembles) because it is
    // a well-specified, widely-vetted, non-cryptographic hash with excellent avalanche behavior and a
    // Burst-friendly inner loop (pure integer add/xor/multiply/rotate, no data-dependent branching
    // inside the hot 16-byte block loop) - "cleanest" in the sense of being a documented, externally
    // checkable reference algorithm rather than an ad hoc mix invented for this file, and "fastest" in
    // the sense of processing 4 independent accumulator lanes per 16 input bytes before a final
    // combine (lets Burst pipeline/vectorize the 4 lanes). There is explicitly NO output-compatibility
    // requirement with Unity.Mathematics' math.hash (a different, unspecified internal scheme) or with
    // any other xxHash32 implementation's byte-order assumptions beyond this file's own reads (see
    // ReadLE32 below, which fixes a canonical read order independent of host endianness) - this class
    // only promises internal determinism/consistency (same input + seed -> same output, forever,
    // within this library), which is exactly what the flagship use case (see the float/double file's
    // class doc in this same folder) needs.
    public static partial class Hash
    {
        private const uint Prime1 = 2654435761u;
        private const uint Prime2 = 2246822519u;
        private const uint Prime3 = 3266489917u;
        private const uint Prime4 = 668265263u;
        private const uint Prime5 = 374761393u;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint RotL(uint x, int r) => (x << r) | (x >> (32 - r));

        // Reads 4 bytes as a little-endian uint32 via an explicit byte combine (NOT an unaligned
        // pointer cast): this is xxHash32's own canonical read order, an algorithm detail that must
        // stay fixed regardless of the host machine's actual endianness, and a manual combine also
        // sidesteps any strict-alignment concern for buffers whose element stride isn't a multiple of
        // 4 (e.g. a bool buffer, or a tail slice not starting on a 4-byte boundary).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint ReadLE32(byte* p) =>
            (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);

        // The core xxHash32 mixing step: folds one 32-bit input word into a running accumulator.
        // Reused verbatim as the public `combine(a, b)` scalar helper below - deliberately NOT
        // commutative (the second argument is scaled by Prime2 before being added to the first, which
        // is not), which is exactly the non-commutativity `combine` documents.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Round(uint acc, uint input)
        {
            acc += input * Prime2;
            acc = RotL(acc, 13);
            acc *= Prime1;
            return acc;
        }

        /// <summary>
        /// Core byte-stream kernel: xxHash32 over <paramref name="byteLength"/> raw bytes starting at
        /// <paramref name="data"/>, seeded with <paramref name="seed"/>. Every typed hash/rowHashes/
        /// colHashes entry point in this class (see the sibling per-type files in this folder) is a
        /// thin, <see cref="MethodImplOptions.AggressiveInlining"/> wrapper around this one
        /// <see cref="MethodImplOptions.NoInlining"/> loop kernel - the same split UnsafeOP's raw
        /// pointer kernels use. <paramref name="byteLength"/> == 0 is well-defined and deterministic
        /// (returns a seed-dependent constant; <paramref name="data"/> is never dereferenced in that
        /// case, so a null/dangling pointer is safe for a zero-length call).
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static unsafe uint hash(byte* data, int byteLength, uint seed = 0)
        {
            byte* p = data;
            byte* bEnd = data + byteLength;
            uint h32;

            if (byteLength >= 16)
            {
                byte* limit = bEnd - 16;
                uint v1 = seed + Prime1 + Prime2;
                uint v2 = seed + Prime2;
                uint v3 = seed;
                uint v4 = seed - Prime1;

                do
                {
                    v1 = Round(v1, ReadLE32(p)); p += 4;
                    v2 = Round(v2, ReadLE32(p)); p += 4;
                    v3 = Round(v3, ReadLE32(p)); p += 4;
                    v4 = Round(v4, ReadLE32(p)); p += 4;
                } while (p <= limit);

                h32 = RotL(v1, 1) + RotL(v2, 7) + RotL(v3, 12) + RotL(v4, 18);
            }
            else
            {
                h32 = seed + Prime5;
            }

            h32 += (uint)byteLength;

            while (p + 4 <= bEnd)
            {
                h32 += ReadLE32(p) * Prime3;
                h32 = RotL(h32, 17) * Prime4;
                p += 4;
            }

            while (p < bEnd)
            {
                h32 += (uint)(*p) * Prime5;
                h32 = RotL(h32, 11) * Prime1;
                p++;
            }

            h32 ^= h32 >> 15;
            h32 *= Prime2;
            h32 ^= h32 >> 13;
            h32 *= Prime3;
            h32 ^= h32 >> 16;

            return h32;
        }

        /// <summary>
        /// Combines two hash values into one, order-sensitive: <c>combine(a, b) != combine(b, a)</c>
        /// for typical a/b (the second argument is scaled by a different prime than the first before
        /// mixing - see <see cref="Round"/>, which this reuses verbatim). Useful for folding several
        /// independently-computed hashes (e.g. one per field of a larger game-state record) into a
        /// single checksum without re-hashing their combined bytes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint combine(uint a, uint b) => Round(a, b);
    }
}
