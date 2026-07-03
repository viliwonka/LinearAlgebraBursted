#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Query Group A for integer types: flat / scalar predicate ops (findFirst, count, any, all,
    // findAll). Concrete-shape overloads (shortN / shortMxN) forward to the generic bodies in
    // shortQueryCore -- so the merged int/short/long partial never collides on the type-identical
    // `<T,P>(in T, ref P)` signatures (CS0111). Groups B/C/D are fProxy-only; for integer matrix
    // row/col filtering use the float or double variant.
    public static partial class Query
    {
        // -------------------------------------------------------------------------
        // GROUP A — FLAT / SCALAR PREDICATE OPS
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the flat index of the first element in x where pred.Test(x[i]) is true.
        /// Short-circuits on the first match. Returns -1 if none match. Empty x returns -1.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findFirst<P>(in shortN   x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.findFirst(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findFirst<P>(in shortMxN x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.findFirst(in x, ref pred);

        /// <summary>
        /// Returns the count of elements in x where pred.Test(x[i]) is true. Full scan. Empty x returns 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int count<P>(in shortN   x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.count(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int count<P>(in shortMxN x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.count(in x, ref pred);

        /// <summary>
        /// Returns true if at least one element in x satisfies pred. Short-circuits. Empty x returns false.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool any<P>(in shortN   x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.any(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool any<P>(in shortMxN x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.any(in x, ref pred);

        /// <summary>
        /// Returns true if every element in x satisfies pred. Short-circuits. Empty x returns true (vacuous).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool all<P>(in shortN   x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.all(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool all<P>(in shortMxN x, ref P pred) where P : struct, IshortPredicate => shortQueryCore.all(in x, ref pred);

        /// <summary>
        /// Fills idx[0..count) with flat indices where pred.Test(x[i]) is true, in ascending scan order.
        /// Returns count. idx must have length >= x.Data.Length. Empty x returns 0 with no writes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findAll<P>(in shortN   x, ref P pred, ref Indices idx) where P : struct, IshortPredicate => shortQueryCore.findAll(in x, ref pred, ref idx);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findAll<P>(in shortMxN x, ref P pred, ref Indices idx) where P : struct, IshortPredicate => shortQueryCore.findAll(in x, ref pred, ref idx);
    }
}
