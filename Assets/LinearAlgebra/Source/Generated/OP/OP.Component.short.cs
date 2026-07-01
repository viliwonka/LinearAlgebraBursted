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
    public static partial class shortElem_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(T place, short s) where T : unmanaged, IUnsafeshortArray {

            unsafe {
                Unsafe_OP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInpl<T>(T place, short s) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {
                Unsafe_OP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(this T place, short s) where T : unmanaged, IUnsafeshortArray
        {
            unsafe
            {
                Unsafe_OP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(short s, T place) where T : unmanaged, IUnsafeshortArray
        {
            unsafe
            {
                Unsafe_OP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(this T place, T from) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {
                // place += from. (was passing the operands to compAdd reversed → mutated `from`.)
                Unsafe_OP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T place, T fromB) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {
                Unsafe_OP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(this T place, short s) where T : unmanaged, IUnsafeshortArray
        {
            unsafe
            {
                Unsafe_OP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(short s, T place) where T : unmanaged, IUnsafeshortArray
        {
            unsafe
            {
                Unsafe_OP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of mulInpl, matching addInpl/subInpl's existing pattern
        // of overloading a single name across a scalar (T, short) and a buffer (T, T) form.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInpl<T>(this T from, T to) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {
                Unsafe_OP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of divInpl.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {
                Unsafe_OP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of modInpl.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {
                Unsafe_OP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T v, short s) where T : unmanaged, IUnsafeshortArray
        {
            addInpl(v, (short)(-s));
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(short s, T v) where T : unmanaged, IUnsafeshortArray
        {
            unsafe {                 
                Unsafe_OP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInpl<T>(this T a) where T : unmanaged, IUnsafeshortArray
        {
            unsafe { 
                Unsafe_OP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the mathUnsafe clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInpl<T>(in T x, short lo, short hi) where T : unmanaged, IUnsafeshortArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInpl: lo must be <= hi");
            unsafe
            {
                mathUnsafeshort.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseComplementInpl<T>(this T a) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseComplement(a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInpl<T>(this T a, short value) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseAnd(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInpl<T>(this T a, short value) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseOr(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInpl<T>(this T a, short value) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseXor(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInpl<T>(this T a, int shift) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseLeftShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInpl<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseLeftShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInpl<T>(this T a, int shift) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseRightShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInpl<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseRightShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInpl<T>(this T a, T b) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseAndComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInpl<T>(this T a, T b) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseOrComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInpl<T>(this T a, T b) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseXorComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInpl<T>(this T a, T b) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseLeftShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInpl<T>(this T a, T b) where T : unmanaged, IUnsafeshortArray {
            unsafe {
                Unsafe_OP.bitwiseRightShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }
    }
}
