using System;
using System.Runtime.CompilerServices;


namespace LinearAlgebra
{
    public static partial class Hash
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(in longN v, uint seed = 0)
        {
            unsafe { return hash((byte*)v.Data.Ptr, v.Data.Length * sizeof(long), seed); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(in longMxN v, uint seed = 0)
        {
            unsafe { return hash((byte*)v.Data.Ptr, v.Data.Length * sizeof(long), seed); }
        }

        // See the identical note in Hash.fProxy.cs: `uintN` cannot be hand-written directly here
        // (it does not exist as a real type in TemplateSource's own standalone compile - it is a
        // codegen OUTPUT of this very file's own int/short/long/uint rotation), so the dest type is
        // always spelled via the real `longN` placeholder token but immediately CHOSEN to the fixed
        // literal "uintN" for every one of this file's 4 generated slots (int/short/long/uint) - this
        // keeps int-sourced/short-sourced/long-sourced rowHashes/colHashes correctly returning a uint
        // buffer instead of accidentally tracking A's own element type.

        /// <summary>
        /// Writes one xxHash32 value per row of <paramref name="A"/> into <paramref name="dest"/> -
        /// dest[r] equals Hash.hash of row r extracted as a standalone vector, given the same seed.
        /// Zero-alloc: rows are contiguous (row-major storage), so each row is hashed directly out of
        /// A's backing buffer with no gather/copy. <paramref name="dest"/> is a uint buffer (a
        /// uintN) sized A.M_Rows, regardless of A's own element type.
        /// </summary>
        public static void rowHashes(in longMxN A, ref uintN dest, uint seed = 0)
        {
            if (dest.N != A.M_Rows)
                throw new ArgumentException("Hash.rowHashes: dest.N must equal A.M_Rows");

            unsafe
            {
                int rowBytes = A.N_Cols * sizeof(long);
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
        public static uintN rowHashes(in longMxN A, uint seed = 0)
        {
            var dest = A.uintVec(A.M_Rows);
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
        public static void colHashes(in longMxN A, ref uintN dest, uint seed = 0)
        {
            if (dest.N != A.N_Cols)
                throw new ArgumentException("Hash.colHashes: dest.N must equal A.N_Cols");

            if (A.N_Cols == 0) return;

            var col = A.longTempVec(A.M_Rows);
            unsafe
            {
                int byteLen = A.M_Rows * sizeof(long);
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
        public static uintN colHashes(in longMxN A, uint seed = 0)
        {
            var dest = A.uintVec(A.N_Cols);
            colHashes(in A, ref dest, seed);
            return dest;
        }

        
    }
}
