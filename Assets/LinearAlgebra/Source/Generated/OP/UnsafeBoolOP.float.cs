#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeBoolOP
    {

        #region COMPARATORS
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLess([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreater([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqual([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqual([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqual([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqual([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessScalar([NoAlias] float* a, float b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterScalar([NoAlias] float* a, float b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqualScalar([NoAlias] float* a, float b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqualScalar([NoAlias] float* a, float b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqualScalar([NoAlias] float* a, float b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqualScalar([NoAlias] float* a, float b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b;
        }
        #endregion
    }
}
