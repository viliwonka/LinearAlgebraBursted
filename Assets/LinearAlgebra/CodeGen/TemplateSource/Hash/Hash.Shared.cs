using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Type-agnostic byte-stream hash kernel shared by every element-type family (float/double/
    // int/short/long/uint/bool). A single, non-templated file: the per-type wrapper files all
    // partial-merge into this SAME `Hash` class, so this kernel lives here once instead of
    // colliding as CS0111/CS0101 across merged fragments.
    //
    // Codegen hazard: this file must never contain the sibling per-type files' proxy-token name
    // spellings -- see Hash/DEVLOG.md. Refer to the sibling files by description ("the
    // float/double file", "the int-family file") instead of by name.
    //
    // Algorithm: xxHash32 (Yann Collet, public domain); non-cryptographic. No output-compatibility
    // requirement with Unity.Mathematics' math.hash or any other xxHash32 implementation -- only
    // internal determinism (same input + seed -> same output, forever, within this library).
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
