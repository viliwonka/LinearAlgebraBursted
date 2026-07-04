#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{
    /// <summary>
    /// </summary>
    public static partial class uintComp {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(T place, uint s) where T : unmanaged, IUnsafeuintArray {

            unsafe {
                UnsafeOP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(T place, uint s) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T place, uint s) where T : unmanaged, IUnsafeuintArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(uint s, T place) where T : unmanaged, IUnsafeuintArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(this T place, T from) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                // place += from. (was passing the operands to compAdd reversed → mutated `from`.)
                UnsafeOP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T place, T fromB) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T place, uint s) where T : unmanaged, IUnsafeuintArray
        {
            unsafe
            {
                UnsafeOP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(uint s, T place) where T : unmanaged, IUnsafeuintArray
        {
            unsafe
            {
                UnsafeOP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of mulInPlace, matching addInPlace/subInPlace's existing pattern
        // of overloading a single name across a scalar (T, uint) and a buffer (T, T) form.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(this T from, T to) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of divInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of modInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // v - s, via a direct forward-order kernel (UnsafeOP.scalSub(target, n, s)) rather than the
        // v + (-s) negation trick a scalar-subtract could otherwise reuse from addInPlace: unsigned
        // types can't negate s, and this is bit-identical to the negation trick for signed types
        // anyway (v + (-s) == v - s under modular wraparound), so every generated type shares one
        // implementation. Callers (e.g. the vector/matrix `operator -` overloads) call this same
        // name either way - see uintN.Operators.cs.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T v, uint s) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.scalSub(v.Data.Ptr, v.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(uint s, T v) where T : unmanaged, IUnsafeuintArray
        {
            unsafe {
                UnsafeOP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        // Negation has no unsigned meaning (uint has no unary minus), so this - and every operator
        // that calls it (the vector/matrix unary `operator -`) - simply doesn't exist for uint.
        

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the UnsafeMathOP clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.
        /// Takes <paramref name="x"/> by value (<c>this T</c>), matching every other Comp wrapper in this
        /// file - a generic extension method's receiver cannot use <c>this in T</c> (CS8338: the 'in'
        /// extension-method form requires a concrete, non-generic value type). Existing callers that
        /// wrote the old static-style <c>clampInPlace(in v, ...)</c> just drop the now-illegal <c>in</c>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInPlace<T>(this T x, uint lo, uint hi) where T : unmanaged, IUnsafeuintArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInPlace: lo must be <= hi");
            unsafe
            {
                UnsafeMathOP.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseComplementInPlace<T>(this T a) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseComplement(a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInPlace<T>(this T a, uint value) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseAnd(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInPlace<T>(this T a, uint value) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseOr(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInPlace<T>(this T a, uint value) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseXor(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInPlace<T>(this T a, int shift) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseLeftShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInPlace<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseLeftShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInPlace<T>(this T a, int shift) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseRightShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInPlace<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseRightShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseAndComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseOrComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseXorComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseLeftShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInPlace<T>(this T a, T b) where T : unmanaged, IUnsafeuintArray {
            unsafe {
                UnsafeOP.bitwiseRightShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        // ---- Componentwise math, forwarding to UnsafeMathOP (mathUnsafe's former home). ----

        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void minInPlace<T>(this T x, T y) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeMathOP.min(x.Data.Ptr, y.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxInPlace<T>(this T x, T y) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeMathOP.max(x.Data.Ptr, y.Data.Ptr, x.Data.Length); }
        }

        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void madInPlace<T>(this T a, T b, T c) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeMathOP.mad(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, a.Data.Length); }
        }

        // ---- Bit-manipulation intrinsics, forwarding to UnsafeBitsOP (see UnsafeBitsOP.uint.cs
        // for the per-type width-correction details, especially short). Every one of these REPLACES
        // each element in place with the op's own result (e.g. countbitsInPlace turns each element
        // into its own population count) - the same in-place philosophy as everything else in this
        // file, just producing a differently-meaning value rather than a transformed one. Sign-
        // agnostic (they act on the bit pattern, not the numeric value) - no skipFor for uint. ----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void countbitsInPlace<T>(this T x) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.countbits(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void tzcntInPlace<T>(this T x) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.tzcnt(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lzcntInPlace<T>(this T x) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.lzcnt(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void reversebitsInPlace<T>(this T x) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.reversebits(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rorInPlace<T>(this T x, int n) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.ror(x.Data.Ptr, x.Data.Length, n); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rolInPlace<T>(this T x, int n) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.rol(x.Data.Ptr, x.Data.Length, n); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ceilpow2InPlace<T>(this T x) where T : unmanaged, IUnsafeuintArray
        {
            unsafe { UnsafeBitsOP.ceilpow2(x.Data.Ptr, x.Data.Length); }
        }
    }
}
