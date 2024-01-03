#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeBoolOP
    {

        #region COMPARATORS
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLess([NoAlias] long* a, [NoAlias] long* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreater([NoAlias] long* a, [NoAlias] long* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqual([NoAlias] long* a, [NoAlias] long* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqual([NoAlias] long* a, [NoAlias] long* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqual([NoAlias] long* a, [NoAlias] long* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqual([NoAlias] long* a, [NoAlias] long* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessScalar([NoAlias] long* a, long b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterScalar([NoAlias] long* a, long b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqualScalar([NoAlias] long* a, long b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqualScalar([NoAlias] long* a, long b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqualScalar([NoAlias] long* a, long b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqualScalar([NoAlias] long* a, long b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b;
        }
        #endregion
    }
}
