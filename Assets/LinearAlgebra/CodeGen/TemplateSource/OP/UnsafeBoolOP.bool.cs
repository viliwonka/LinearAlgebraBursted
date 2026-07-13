using System;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra.Internal
{
    
    public static unsafe partial class UnsafeBoolOP
    {
        // ---- dedicated in-place kernels (target mutated; from must not alias target, matching
        //      the numeric comp* family's contract). The pure operators copy first (TempCopy)
        //      and then run these, so no copy-form kernels are needed. ----

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void notInPlace([NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = !target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void orInPlace([NoAlias] bool* target, [NoAlias] bool* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] |= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void orInPlace([NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] |= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void andInPlace([NoAlias] bool* target, [NoAlias] bool* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] &= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void andInPlace([NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] &= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void xorInPlace([NoAlias] bool* target, [NoAlias] bool* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] ^= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void xorInPlace([NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] ^= b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void equalsInPlace([NoAlias] bool* target, [NoAlias] bool* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = target[i] == from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void equalsInPlace([NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] = target[i] == b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void notEqualsInPlace([NoAlias] bool* target, [NoAlias] bool* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = target[i] != from[i];
        }

    }
}
