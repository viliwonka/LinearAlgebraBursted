#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    [BurstCompile]
    /// <summary>
    /// In-place boolean-buffer logic ops (not/or/and/xor/equals/notEquals), buffer×buffer and buffer×scalar.
    /// </summary>
    public static partial class Bool_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void not<T>(this T a) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe {
                UnsafeBool_OP.not(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void or<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.or(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void or<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.or(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void and<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.and(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void and<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.and(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void xor<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.xor(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void xor<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.xor(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void equals<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.equals(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void equals<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.equals(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void notEquals<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBool_OP.notEquals(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }
    }
}
