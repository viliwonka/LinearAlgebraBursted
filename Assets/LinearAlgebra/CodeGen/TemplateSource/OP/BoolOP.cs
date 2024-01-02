#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    // can add chaining here for inplace methods

    [BurstCompile]
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class BoolOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void not<T>(this T a) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe {
                UnsafeBoolOP.not(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void or<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.or(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void or<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.or(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void and<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.and(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void and<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.and(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void xor<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.xor(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void xor<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.xor(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void equals<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.equals(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void equals<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.equals(a.Data.Ptr, a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void notEquals<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.notEquals(a.Data.Ptr, b.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }
    }
}
