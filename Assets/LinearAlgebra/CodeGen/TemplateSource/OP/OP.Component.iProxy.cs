#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// </summary>
    public static partial class iProxyComp {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray {

            unsafe {
                UnsafeOP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(iProxy s, T place) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(this T place, T from) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                // place += from. (was passing the operands to compAdd reversed → mutated `from`.)
                UnsafeOP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T place, T fromB) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(iProxy s, T place) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of mulInPlace, matching addInPlace/subInPlace's existing pattern
        // of overloading a single name across a scalar (T, iProxy) and a buffer (T, T) form.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(this T from, T to) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of divInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of modInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T v, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            addInPlace(v, (iProxy)(-s));
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(iProxy s, T v) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {                 
                UnsafeOP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInPlace<T>(this T a) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe { 
                UnsafeOP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the mathUnsafe clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInPlace<T>(in T x, iProxy lo, iProxy hi) where T : unmanaged, IUnsafeiProxyArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInPlace: lo must be <= hi");
            unsafe
            {
                mathUnsafeiProxy.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseComplementInPlace<T>(this T a) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseComplement(a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInPlace<T>(this T a, iProxy value) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseAnd(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInPlace<T>(this T a, iProxy value) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseOr(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInPlace<T>(this T a, iProxy value) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseXor(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInPlace<T>(this T a, int shift) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseLeftShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInPlace<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseLeftShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInPlace<T>(this T a, int shift) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseRightShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInPlace<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseRightShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseAndComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseOrComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseXorComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseLeftShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseRightShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }
    }
}
