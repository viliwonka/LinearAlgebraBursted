#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;

namespace LinearAlgebra
{

    // can add chaining here for inplace methods

    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class float_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(T place, float s) where T : unmanaged, IUnsafefloatArray {

            unsafe {
                Unsafe_OP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInpl<T>(T place, float s) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(this T place, float s) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                Unsafe_OP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(float s, T place) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                Unsafe_OP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(this T place, T from) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                // place += from. (compAdd is (target, from); the old call passed them reversed, so this
                // method actually mutated `from` instead of `place` — wrong for any direct caller.)
                Unsafe_OP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T place, T fromB) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        // y += a * x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addScaledInpl<T>(this T y, float a, T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.axpy(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        // y = a * y + x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void scaleAddInpl<T>(this T y, float a, T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.aypx(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(this T place, float s) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                Unsafe_OP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(float s, T place) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                Unsafe_OP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void compMulInpl<T>(this T from, T to) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void compDivInpl<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void compModDiv<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T v, float s) where T : unmanaged, IUnsafefloatArray
        {
            addInpl(v, -s);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(float s, T v) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {                 
                Unsafe_OP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInpl<T>(this T a) where T : unmanaged, IUnsafefloatArray
        {
            unsafe {
                Unsafe_OP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the mathUnsafe clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInpl<T>(in T x, float lo, float hi) where T : unmanaged, IUnsafefloatArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInpl: lo must be <= hi");
            unsafe
            {
                mathUnsafefloat.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }
    }
}
