using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra
{
    
    public static unsafe class mathUnsafeint
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setAll([NoAlias] int* x, int n, int s)
        {
            for (int i = 0; i < n; i++)
                x[i] = (int)s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexZero([NoAlias] int* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (int)i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexOne([NoAlias] int* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (int)(i+1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void abs([NoAlias] int* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (int)(-x[i]) : x[i];   
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void max([NoAlias] int* x, [NoAlias] int* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > y[i]? x[i]: y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void min([NoAlias] int* x, [NoAlias] int* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < y[i] ? x[i] : y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clamp([NoAlias] int* x, int n, int min, int max)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > max ? max : x[i] < min ? min : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mod([NoAlias] int* x, int y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (int)(x[i] % y);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void relu([NoAlias] int* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (int)0 : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mad([NoAlias] int* a, [NoAlias] int* b, [NoAlias] int* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = (int)(a[i] * b[i] + c[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int dot([NoAlias] int* x, [NoAlias] int* y, int n)
        {
            int sum = 0;
            for (int i = 0; i < n; i++)
                sum += (int)(x[i] * y[i]);

            return sum;
        }

    }
}