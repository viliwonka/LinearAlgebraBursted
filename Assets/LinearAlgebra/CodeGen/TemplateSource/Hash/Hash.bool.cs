using System.Runtime.CompilerServices;

namespace BULA
{
    public static partial class Hash
    {
        /// <summary>
        /// Hashes a bool vector's raw bytes (1 byte per element in Burst - see the float/double
        /// file's class doc in this same folder for the flagship use case and the float-only
        /// bit-pattern caveats, which do NOT apply here). Burst's bool storage has exactly two valid
        /// byte values (0/1), so equal-valued bools always hash identically - there is no
        /// -0.0/NaN-style footgun for bool.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(in boolN v, uint seed = 0)
        {
            unsafe { return hash((byte*)v.Data.Ptr, v.Data.Length * sizeof(bool), seed); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(in boolMxN v, uint seed = 0)
        {
            unsafe { return hash((byte*)v.Data.Ptr, v.Data.Length * sizeof(bool), seed); }
        }

        // rowHashes/colHashes(in boolMxN, ...) are NOT here: this file is a genuinely singular file
        // (bool has only one concrete type, so it never goes through codegen's per-type rotation),
        // but those two methods need a dest type fixed at the real generated uint vector type, which
        // can only be produced via the int-family rotation's own substitution machinery. They
        // instead live in the int-family file in this same folder, restricted (via a per-type skip
        // marker) to emit exactly once. Same public `Hash` class either way - see that file's note.
    }
}
