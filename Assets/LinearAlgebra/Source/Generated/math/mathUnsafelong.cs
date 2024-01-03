using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra
{
    
    public static unsafe class mathUnsafelong
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setAll([NoAlias] long* x, int n, long s)
        {
            for (int i = 0; i < n; i++)
                x[i] = (long)s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexZero([NoAlias] long* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (long)i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexOne([NoAlias] long* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (long)(i+1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void abs([NoAlias] long* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (long)(-x[i]) : x[i];   
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void max([NoAlias] long* x, [NoAlias] long* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > y[i]? x[i]: y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void min([NoAlias] long* x, [NoAlias] long* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < y[i] ? x[i] : y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clamp([NoAlias] long* x, int n, long min, long max)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > max ? max : x[i] < min ? min : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mod([NoAlias] long* x, long y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (long)(x[i] % y);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void relu([NoAlias] long* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (long)0 : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mad([NoAlias] long* a, [NoAlias] long* b, [NoAlias] long* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = (long)(a[i] * b[i] + c[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long dot([NoAlias] long* x, [NoAlias] long* y, int n)
        {
            long sum = 0;
            for (int i = 0; i < n; i++)
                sum += (long)(x[i] * y[i]);

            return sum;
        }

    }
}