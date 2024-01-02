using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra
{
    
    public static unsafe class mathUnsafeiProxy
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setAll([NoAlias] iProxy* x, int n, iProxy s)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexZero([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexOne([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)(i+1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void abs([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (iProxy)(-x[i]) : x[i];   
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void max([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > y[i]? x[i]: y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void min([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < y[i] ? x[i] : y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clamp([NoAlias] iProxy* x, int n, iProxy min, iProxy max)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > max ? max : x[i] < min ? min : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mod([NoAlias] iProxy* x, iProxy y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)(x[i] % y);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void relu([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (iProxy)0 : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mad([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] iProxy* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = (iProxy)(a[i] * b[i] + c[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxy dot([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
        {
            iProxy sum = 0;
            for (int i = 0; i < n; i++)
                sum += (iProxy)(x[i] * y[i]);

            return sum;
        }

    }
}