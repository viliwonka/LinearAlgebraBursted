#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// Inpl = inplace
    /// </summary>
    public static partial class doubleComp {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(T place, double s) where T : unmanaged, IUnsafedoubleArray {

            unsafe {
                UnsafeOP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInpl<T>(T place, double s) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(this T place, double s) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(double s, T place) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(this T place, T from) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                // place += from. (compAdd is (target, from) — a prior reversed call mutated `from` instead.)
                UnsafeOP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T place, T fromB) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        // y += a * x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addScaledInpl<T>(this T y, double a, T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.axpy(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        // y = a * y + x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void scaleAddInpl<T>(this T y, double a, T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.aypx(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(this T place, double s) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(double s, T place) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of mulInpl, matching addInpl/subInpl's existing pattern
        // of overloading a single name across a scalar (T, double) and a buffer (T, T) form.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInpl<T>(this T from, T to) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of divInpl.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of modInpl.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T v, double s) where T : unmanaged, IUnsafedoubleArray
        {
            addInpl(v, -s);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(double s, T v) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {                 
                UnsafeOP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInpl<T>(this T a) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the mathUnsafe clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInpl<T>(in T x, double lo, double hi) where T : unmanaged, IUnsafedoubleArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInpl: lo must be <= hi");
            unsafe
            {
                mathUnsafedouble.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }
    }
}
