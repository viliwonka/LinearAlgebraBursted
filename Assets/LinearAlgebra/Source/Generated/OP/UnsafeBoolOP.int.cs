#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeBoolOP
    {

        #region COMPARATORS
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLess([NoAlias] int* a, [NoAlias] int* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreater([NoAlias] int* a, [NoAlias] int* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqual([NoAlias] int* a, [NoAlias] int* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqual([NoAlias] int* a, [NoAlias] int* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqual([NoAlias] int* a, [NoAlias] int* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqual([NoAlias] int* a, [NoAlias] int* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessScalar([NoAlias] int* a, int b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] < b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterScalar([NoAlias] int* a, int b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] > b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprLessOrEqualScalar([NoAlias] int* a, int b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] <= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprGreaterOrEqualScalar([NoAlias] int* a, int b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] >= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprEqualScalar([NoAlias] int* a, int b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cmprNotEqualScalar([NoAlias] int* a, int b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b;
        }
        #endregion
    }
}
