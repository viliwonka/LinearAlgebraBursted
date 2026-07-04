#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Mathematics;

//alsoExpand[uint]// relational comparators (+ the ispow2 predicate below) - no signed-only ops here.

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeBoolOP
    {

        #region COMPARATORS
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLess([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreater([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqual([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqual([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqual([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqual([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessScalar([NoAlias] iProxy* a, iProxy b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterScalar([NoAlias] iProxy* a, iProxy b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqualScalar([NoAlias] iProxy* a, iProxy b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqualScalar([NoAlias] iProxy* a, iProxy b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqualScalar([NoAlias] iProxy* a, iProxy b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqualScalar([NoAlias] iProxy* a, iProxy b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b;
        }
        #endregion

        #region PREDICATES
        // ispow2 is a genuine PREDICATE (not a relational comparison against a second operand), but
        // it produces a boolN/boolMxN the same way the comparators above do, so its kernel lives here
        // rather than among the elementwise math kernels in UnsafeMathOP.iProxy.cs/UnsafeBitsOP.
        // iProxy.cs - see iProxyN.Comparators.cs/iProxyMxN.Comparators.cs for its public surface.
        // Matches Unity.Mathematics' math.ispow2 semantics per type:
        //   - int/uint: math.ispow2 exists natively (x > 0 && (x & (x-1)) == 0; 0 and negative
        //     values are never a power of two - note this means a negative int bit pattern is never
        //     reported as a power of two even if the SAME bits, read as uint, would be).
        //   - long: math.ispow2 has NO long/ulong overload in Unity.Mathematics (verified against
        //     com.unity.mathematics's math.cs), so it is defined directly here using the identical
        //     formula, widened to long.
        //   - short: math.ispow2 has no short overload either. Unlike the bit-PATTERN ops
        //     (countbits/tzcnt/lzcnt/reversebits/ror/rol - see UnsafeBitsOP.iProxy.cs), ispow2 tests
        //     the NUMERIC VALUE, and a short's value is preserved exactly under promotion to int (no
        //     reinterpretation/width correction needed) - so the same int-style formula applies
        //     directly to the promoted value, with no cast tricks required.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ispow2([NoAlias] iProxy* a, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
            {
                iProxy v = a[i];
                target[i] = /*+choose[math.ispow2(v)|v > 0 && (v & (v - 1)) == 0|v > 0L && (v & (v - 1L)) == 0L|math.ispow2(v)]*/math.ispow2(v)/*-choose*/;
            }
        }
        #endregion
    }
}
