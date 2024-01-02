#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    
    public static unsafe partial class UnsafeBoolOP
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void equals([NoAlias] bool* a, [NoAlias] bool* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void equals([NoAlias] bool* a, [NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] == b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void notEquals([NoAlias] bool* a, [NoAlias] bool* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] != b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void not([NoAlias] bool* target, [NoAlias] bool* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = !from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void or([NoAlias] bool* a, [NoAlias] bool* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] | b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void or([NoAlias] bool* a, [NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] | b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void and([NoAlias] bool* a, [NoAlias] bool* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] & b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void and([NoAlias] bool* a, [NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] & b;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void xor([NoAlias] bool* a, [NoAlias] bool* b, [NoAlias] bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] ^ b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void xor([NoAlias] bool* a, [NoAlias] bool* target, int n, bool b)
        {
            for (int i = 0; i < n; i++)
                target[i] = a[i] ^ b;
        }

    }
}
