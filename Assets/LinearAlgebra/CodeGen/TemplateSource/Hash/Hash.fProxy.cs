using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    /// <summary>
    /// Deterministic, non-cryptographic hashing over LinearAlgebra vectors/matrices, for every
    /// element type (float/double/int/short/long/uint/bool). See Hash.Shared.cs's class doc for the
    /// chosen algorithm (xxHash32) and why.
    ///
    /// Main use case: deterministic state checksums for lockstep-multiplayer desync detection -
    /// hash each peer's authoritative simulation state every tick (or every N ticks) and compare the
    /// resulting uint across the network; a mismatch means two peers' simulations have diverged.
    ///
    /// Floating-point caveats (float/double only - does not apply to int/short/long/uint/bool):
    /// floats are hashed by their raw BITS (the buffer's actual memory, reinterpreted - free, no
    /// conversion needed), NOT by numeric value. Two consequences:
    /// <list type="bullet">
    /// <item><description><c>-0.0</c> and <c>+0.0</c> compare EQUAL (<c>-0.0f == 0.0f</c> is true)
    /// but hash DIFFERENTLY - their bit patterns differ only in the sign bit.</description></item>
    /// <item><description>Two NaNs with different bit payloads hash DIFFERENTLY even though every
    /// NaN compares unequal to everything (including itself) under IEEE 754 - hashing does not follow
    /// IEEE equality semantics, it follows raw memory.</description></item>
    /// </list>
    /// </summary>
    public static partial class Hash
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(in fProxyN v, uint seed = 0)
        {
            unsafe { return hash((byte*)v.Data.Ptr, v.Data.Length * sizeof(fProxy), seed); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(in fProxyMxN v, uint seed = 0)
        {
            unsafe { return hash((byte*)v.Data.Ptr, v.Data.Length * sizeof(fProxy), seed); }
        }

        // The row/col hash dest is always a uint buffer regardless of A's element type. "uintN"/
        // "uintMxN" are codegen outputs (the iProxy token is always chosen as uint here, the same for
        // both float and double), not types that exist in TemplateSource's own standalone compile, so
        // they are emitted via the choose marker rather than hand-written.

        /// <summary>
        /// Writes one xxHash32 value per row of <paramref name="A"/> into <paramref name="dest"/> -
        /// dest[r] equals Hash.hash of row r extracted as a standalone vector, given the same seed.
        /// Zero-alloc: rows are contiguous (row-major storage), so each row is hashed directly out of
        /// A's backing buffer with no gather/copy. <paramref name="dest"/> is a uint buffer (a
        /// uintN) sized A.M_Rows.
        /// </summary>
        public static void rowHashes(in fProxyMxN A, ref /*+choose[uintN|uintN]*/iProxyN/*-choose*/ dest, uint seed = 0)
        {
            if (dest.N != A.M_Rows)
                throw new ArgumentException("Hash.rowHashes: dest.N must equal A.M_Rows");

            unsafe
            {
                int rowBytes = A.N_Cols * sizeof(fProxy);
                byte* rowPtr = (byte*)A.Data.Ptr;
                uint* destPtr = (uint*)dest.Data.Ptr;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    destPtr[r] = hash(rowPtr, rowBytes, seed);
                    rowPtr += rowBytes;
                }
            }
        }

        /// <summary>Allocating wrapper: same as the ref-dest <c>rowHashes</c> overload, but returns a
        /// fresh arena-backed uint buffer (a uintN) instead of writing into a caller-provided one.</summary>
        public static /*+choose[uintN|uintN]*/iProxyN/*-choose*/ rowHashes(in fProxyMxN A, uint seed = 0)
        {
            var dest = A./*+choose[uintVec|uintVec]*/iProxyVec/*-choose*/(A.M_Rows);
            rowHashes(in A, ref dest, seed);
            return dest;
        }

        /// <summary>
        /// Writes one xxHash32 value per column of <paramref name="A"/> into <paramref name="dest"/> -
        /// dest[c] equals Hash.hash of column c extracted as a standalone vector, given the same
        /// seed. Columns are STRIDED (not contiguous, under row-major storage), so each column is
        /// first gathered into a reused scratch vector (drawn once from A's arena Temp pool, refilled
        /// per column) before hashing - this makes colHashes slower than rowHashes (an O(M) gather per
        /// column vs. a direct pointer slice per row), but it is required for correctness: streaming
        /// the strided bytes through xxHash32's block algorithm in column order would NOT produce the
        /// same hash as hashing a real contiguous vector of that column's values, so the gather-then-
        /// hash approach is what makes the "same result as a standalone vector" guarantee above hold.
        /// </summary>
        public static void colHashes(in fProxyMxN A, ref /*+choose[uintN|uintN]*/iProxyN/*-choose*/ dest, uint seed = 0)
        {
            if (dest.N != A.N_Cols)
                throw new ArgumentException("Hash.colHashes: dest.N must equal A.N_Cols");

            if (A.N_Cols == 0) return;

            var col = A.fProxyTempVec(A.M_Rows);
            unsafe
            {
                int byteLen = A.M_Rows * sizeof(fProxy);
                uint* destPtr = (uint*)dest.Data.Ptr;
                for (int c = 0; c < A.N_Cols; c++)
                {
                    for (int r = 0; r < A.M_Rows; r++)
                        col[r] = A[r, c];
                    destPtr[c] = hash((byte*)col.Data.Ptr, byteLen, seed);
                }
            }
        }

        /// <summary>Allocating wrapper: same as the ref-dest <c>colHashes</c> overload, but returns a
        /// fresh arena-backed uint buffer (a uintN) instead of writing into a caller-provided one.</summary>
        public static /*+choose[uintN|uintN]*/iProxyN/*-choose*/ colHashes(in fProxyMxN A, uint seed = 0)
        {
            var dest = A./*+choose[uintVec|uintVec]*/iProxyVec/*-choose*/(A.N_Cols);
            colHashes(in A, ref dest, seed);
            return dest;
        }
    }
}
