using Unity.Burst;
using System.Runtime.CompilerServices;
namespace LinearAlgebra
{

    [BurstCompile]
    public static partial class Analysis {
        
        /// <summary>
        /// Returns true if bm is square and every off-diagonal element is false.
        /// Diagonal elements may be true or false.
        /// </summary>
        public static bool isDiagonal(in boolMxN bm)
        {
            if (bm.M_Rows != bm.N_Cols)
                return false;

            for (int i = 0; i < bm.M_Rows; i++)
            {
                for (int j = 0; j < bm.N_Cols; j++)
                {
                    if (i != j && bm[i, j])
                        return false;
                }
            }

            return true;
        }

        public static bool IsAllSame<T>(in T x) where T : unmanaged, IUnsafeBoolArray
        {
            for (int i = 1; i < x.Data.Length; i++)
            {
                if (x.Data[i-1] != x.Data[i])
                    return false;
            }
            return true;
        } 

        public static bool IsAllEqualTo<T>(in T x, bool y) where T : unmanaged, IUnsafeBoolArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] != y)
                    return false;
            }
            return true;
        }

        


        public static bool IsAnyEqualTo<T>(in T x, bool y) where T : unmanaged, IUnsafeBoolArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] == y)
                    return true;
            }
            return false;
        }

        // ---- any/all: mirror math.any/math.all, including empty-input semantics
        // (any(empty) == false, all(empty) == true).

        /// <summary>Returns true if any element of x is true. any(empty) == false.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool any(in boolN x) => IsAnyEqualTo(x, true);

        /// <summary>Returns true if any element of x is true. any(empty) == false.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool any(in boolMxN x) => IsAnyEqualTo(x, true);

        /// <summary>Returns true if every element of x is true. all(empty) == true.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool all(in boolN x) => IsAllEqualTo(x, true);

        /// <summary>Returns true if every element of x is true. all(empty) == true.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool all(in boolMxN x) => IsAllEqualTo(x, true);

        // --- QueryOP bridge: bool mask → scalar counts ---------------------------

        // ---- whichTrue — fills Indices from boolN/boolMxN ---

        /// <summary>
        /// Fills idx[0..count) with the flat indices of true elements in mask.
        /// Returns count. idx must be sized >= mask.N (worst case).
        /// Use countTrue first if you want to allocate an exact-sized Indices buffer.
        /// </summary>
        public static int whichTrue(in boolN mask, ref Indices idx)
        {
            if (idx.N < mask.N)
                throw new System.ArgumentException("Analysis.whichTrue: idx.N must be >= mask.N");
            int count = 0;
            for (int i = 0; i < mask.N; i++)
                if (mask.Data[i]) idx[count++] = i;
            return count;
        }

        /// <summary>
        /// Matrix overload: fills idx[0..count) with flat indices of true elements in mask.
        /// idx must be sized >= mask.M_Rows * mask.N_Cols.
        /// </summary>
        public static int whichTrue(in boolMxN mask, ref Indices idx)
        {
            int total = mask.M_Rows * mask.N_Cols;
            if (idx.N < total)
                throw new System.ArgumentException("Analysis.whichTrue: idx.N must be >= mask total size");
            int count = 0;
            for (int i = 0; i < total; i++)
                if (mask.Data[i]) idx[count++] = i;
            return count;
        }

        /// <summary>
        /// Returns the count of true elements in mask (no index buffer needed).
        /// Use whichTrue (in Analysis) when you also need the indices.
        /// </summary>
        public static int countTrue(in boolN mask)
        {
            int count = 0;
            for (int i = 0; i < mask.N; i++)
                if (mask.Data[i]) count++;
            return count;
        }

        /// <summary>
        /// Matrix overload: returns the count of true elements in mask.
        /// </summary>
        public static int countTrue(in boolMxN mask)
        {
            int total = mask.M_Rows * mask.N_Cols;
            int count = 0;
            for (int i = 0; i < total; i++)
                if (mask.Data[i]) count++;
            return count;
        }
    }

    public static partial class ArenaExtensions
    {
        // ---- whichTrue (bool → Indices) ----

        /// <summary>
        /// Count-pass + exact-alloc: fills exact-sized Indices with indices of true elements in mask.
        /// </summary>
        public static Indices WhichTrue(this ref Arena arena, in boolN mask)
        {
            int count = Analysis.countTrue(in mask);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            int written = 0;
            for (int i = 0; i < mask.N; i++)
                if (mask.Data[i]) idx[written++] = i;
            return idx;
        }

        /// <summary>
        /// Matrix overload: count-pass + exact-alloc Indices of true element flat indices.
        /// </summary>
        public static Indices WhichTrue(this ref Arena arena, in boolMxN mask)
        {
            int count = Analysis.countTrue(in mask);
            if (count == 0) return arena.Indices(0);
            int total = mask.M_Rows * mask.N_Cols;
            var idx = arena.Indices(count);
            int written = 0;
            for (int i = 0; i < total; i++)
                if (mask.Data[i]) idx[written++] = i;
            return idx;
        }
    }
}
