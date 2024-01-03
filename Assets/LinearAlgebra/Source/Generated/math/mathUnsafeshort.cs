using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra
{
    
    public static unsafe class mathUnsafeshort
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setAll([NoAlias] short* x, int n, short s)
        {
            for (int i = 0; i < n; i++)
                x[i] = (short)s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexZero([NoAlias] short* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (short)i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexOne([NoAlias] short* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (short)(i+1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void abs([NoAlias] short* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (short)(-x[i]) : x[i];   
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void max([NoAlias] short* x, [NoAlias] short* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > y[i]? x[i]: y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void min([NoAlias] short* x, [NoAlias] short* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < y[i] ? x[i] : y[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clamp([NoAlias] short* x, int n, short min, short max)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > max ? max : x[i] < min ? min : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mod([NoAlias] short* x, short y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (short)(x[i] % y);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void relu([NoAlias] short* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? (short)0 : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mad([NoAlias] short* a, [NoAlias] short* b, [NoAlias] short* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = (short)(a[i] * b[i] + c[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short dot([NoAlias] short* x, [NoAlias] short* y, int n)
        {
            short sum = 0;
            for (int i = 0; i < n; i++)
                sum += (short)(x[i] * y[i]);

            return sum;
        }

    }
}