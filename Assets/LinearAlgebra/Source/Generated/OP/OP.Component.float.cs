#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// </summary>
    public static partial class floatComp {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(T place, float s) where T : unmanaged, IUnsafefloatArray {

            unsafe {
                UnsafeOP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(T place, float s) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T place, float s) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(float s, T place) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(this T place, T from) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                // place += from. (compAdd is (target, from) — a prior reversed call mutated `from` instead.)
                UnsafeOP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T place, T fromB) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        // y += a * x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addScaledInPlace<T>(this T y, float a, T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.axpy(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        // y = a * y + x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void scaleAddInPlace<T>(this T y, float a, T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.aypx(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T place, float s) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                UnsafeOP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(float s, T place) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                UnsafeOP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of mulInPlace, matching addInPlace/subInPlace's existing pattern
        // of overloading a single name across a scalar (T, float) and a buffer (T, T) form.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(this T from, T to) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of divInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of modInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T v, float s) where T : unmanaged, IUnsafefloatArray
        {
            addInPlace(v, -s);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(float s, T v) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {                 
                UnsafeOP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInPlace<T>(this T a) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                UnsafeOP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the mathUnsafe clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInPlace<T>(in T x, float lo, float hi) where T : unmanaged, IUnsafefloatArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInPlace: lo must be <= hi");
            unsafe
            {
                mathUnsafefloat.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }
    }
}
