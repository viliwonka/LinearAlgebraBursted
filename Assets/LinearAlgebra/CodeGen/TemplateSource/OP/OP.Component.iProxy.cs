#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;

using Unity.Burst;

namespace LinearAlgebra
{
    // can add chaining here for inplace methods

    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class iProxyOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray {

            unsafe {
                UnsafeOP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInpl<T>(T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(this T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInpl<T>(iProxy s, T place) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInpl<T>(this T place, T from) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compAdd(from.Data.Ptr, place.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T place, T fromB) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compSub(fromB.Data.Ptr, place.Data.Ptr, fromB.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(this T place, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInpl<T>(iProxy s, T place) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe
            {
                UnsafeOP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void compMulInpl<T>(this T from, T to) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void compDivInpl<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void compModDiv<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {
                UnsafeOP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(this T v, iProxy s) where T : unmanaged, IUnsafeiProxyArray
        {
            addInpl(v, (iProxy)(-s));
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInpl<T>(iProxy s, T v) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe {                 
                UnsafeOP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInpl<T>(this T a) where T : unmanaged, IUnsafeiProxyArray
        {
            unsafe { 
                UnsafeOP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseComplementInpl<T>(this T a) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseComplement(a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInpl<T>(this T a, iProxy value) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseAnd(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInpl<T>(this T a, iProxy value) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseOr(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInpl<T>(this T a, iProxy value) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseXor(a.Data.Ptr, a.Data.Length, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInpl<T>(this T a, int shift) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseLeftShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInpl<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseLeftShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInpl<T>(this T a, int shift) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseRightShift(a.Data.Ptr, a.Data.Length, shift);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInpl<T>(int valueToBeShifted, T a) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseRightShift(valueToBeShifted, a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseAndInpl<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseAndComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseOrInpl<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseOrComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseXorInpl<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseXorComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseLeftShiftInpl<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseLeftShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void bitwiseRightShiftInpl<T>(this T a, T b) where T : unmanaged, IUnsafeiProxyArray {
            unsafe {
                UnsafeOP.bitwiseRightShiftComp(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }
    }
}
