#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;

namespace LinearAlgebra
{
    // QueryOP.Predicate: integer scalar predicate ops (Group A only).
    // Groups B, C, and D are fProxy-only; for integer matrix row/col filtering
    // use the float or double variant.
    //
    // Group A — Flat / scalar predicate ops (generic T + P):
    //   findFirst, count, any, all, findAll.
    // Constraint: where T : unmanaged, IUnsafeshortArray
    //             where P : struct, IfshortPredicate
    public static partial class shortQueryOP
    {
        // -------------------------------------------------------------------------
        // GROUP A — FLAT / SCALAR PREDICATE OPS
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the flat index of the first element in x where pred.Test(x[i]) is true.
        /// Short-circuits on the first match. Returns -1 if none match.
        /// Empty x (length == 0) returns -1 without throwing.
        /// Generic over shortN and shortMxN (row-major flat index for matrices).
        /// </summary>
        public static int findFirst<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafeshortArray
            where P : struct, IfshortPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) return i;
            return -1;
        }

        /// <summary>
        /// Returns the count of elements in x where pred.Test(x[i]) is true.
        /// Full scan. Empty x returns 0.
        /// </summary>
        public static int count<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafeshortArray
            where P : struct, IfshortPredicate
        {
            int c = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) c++;
            return c;
        }

        /// <summary>
        /// Returns true if at least one element in x satisfies pred.
        /// Short-circuits on the first true. Empty x returns false.
        /// </summary>
        public static bool any<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafeshortArray
            where P : struct, IfshortPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) return true;
            return false;
        }

        /// <summary>
        /// Returns true if every element in x satisfies pred.
        /// Short-circuits on the first false. Empty x returns true (vacuous truth).
        /// </summary>
        public static bool all<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafeshortArray
            where P : struct, IfshortPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (!pred.Test(x.Data[i])) return false;
            return true;
        }

        /// <summary>
        /// Fills idx[0..count) with flat indices where pred.Test(x[i]) is true,
        /// in ascending scan order. Returns count.
        /// idx must have length >= x.Data.Length (worst case — all elements match).
        /// Empty x returns 0 with no writes.
        /// </summary>
        public static int findAll<T, P>(in T x, ref P pred, ref Indices idx)
            where T : unmanaged, IUnsafeshortArray
            where P : struct, IfshortPredicate
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("QueryOP.findAll: idx.N must be >= x.Data.Length");
            int c = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) idx[c++] = i;
            return c;
        }
    }
}
