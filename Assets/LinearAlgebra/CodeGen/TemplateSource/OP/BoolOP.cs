using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// In-place boolean-buffer logic ops (notInPlace/orInPlace/andInPlace/xorInPlace/equalsInPlace/
    /// notEqualsInPlace), buffer×buffer and buffer×scalar. The == and != operator overloads are the
    /// only allocating counterparts left; !, |, ^ and &amp; have none -- call these directly.
    /// </summary>
    public static partial class boolComp {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void notInPlace<T>(this T a) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe {
                UnsafeBoolOP.notInPlace(a.Data.Ptr, a.Data.Length);
            }
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void orInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.orInPlace(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void orInPlace<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.orInPlace(a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void andInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.andInPlace(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void andInPlace<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.andInPlace(a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void xorInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.xorInPlace(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void xorInPlace<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.xorInPlace(a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void equalsInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.equalsInPlace(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void equalsInPlace<T>(this T a, bool b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.equalsInPlace(a.Data.Ptr, a.Data.Length, b);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void notEqualsInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeBoolArray
        {
            unsafe
            {
                UnsafeBoolOP.notEqualsInPlace(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }
    }
}
