using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;


namespace LinearAlgebra
{

    public static unsafe class mathUnsafeuint
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setAll([NoAlias] uint* x, int n, uint s)
        {
            for (int i = 0; i < n; i++)
                x[i] = (uint)s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexZero([NoAlias] uint* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (uint)i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexOne([NoAlias] uint* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (uint)(i+1);
        }

        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void max([NoAlias] uint* x, [NoAlias] uint* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > y[i]? x[i]: y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void min([NoAlias] uint* x, [NoAlias] uint* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < y[i] ? x[i] : y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clamp([NoAlias] uint* x, int n, uint min, uint max)
        {
            for (int i = 0; i < n; i++)
                x[i] = (uint)math.max(min, math.min(max, x[i]));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mod([NoAlias] uint* x, uint y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (uint)(x[i] % y);
        }

        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mad([NoAlias] uint* a, [NoAlias] uint* b, [NoAlias] uint* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = (uint)(a[i] * b[i] + c[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint dot([NoAlias] uint* x, [NoAlias] uint* y, int n)
        {
            uint sum = 0;
            for (int i = 0; i < n; i++)
                sum += (uint)(x[i] * y[i]);

            return sum;
        }

    }
}