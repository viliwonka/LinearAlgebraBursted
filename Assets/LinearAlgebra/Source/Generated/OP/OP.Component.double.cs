#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// </summary>
    public static partial class doubleComp {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(this T place, double s) where T : unmanaged, IUnsafedoubleArray {

            unsafe {
                UnsafeOP.scalAdd(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(this T place, double s) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.scalMul(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T place, double s) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(double s, T place) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalDiv(s, place.Data.Ptr, place.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addInPlace<T>(this T place, T from) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                // place += from.
                UnsafeOP.compAdd(place.Data.Ptr, from.Data.Ptr, from.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T place, T fromB) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compSub(place.Data.Ptr, fromB.Data.Ptr, fromB.Data.Length);
            }
        }

        // y += a * x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void addScaledInPlace<T>(this T y, double a, T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.axpy(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        // y = a * y + x
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void scaleAddInPlace<T>(this T y, double a, T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.aypx(y.Data.Ptr, x.Data.Ptr, a, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T place, double s) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalMod(place.Data.Ptr, place.Data.Length, s);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(double s, T place) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                UnsafeOP.scalMod(s, place.Data.Ptr, place.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of mulInPlace, matching addInPlace/subInPlace's existing pattern
        // of overloading a single name across a scalar (T, double) and a buffer (T, T) form.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mulInPlace<T>(this T from, T to) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, from.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of divInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void divInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compDiv(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        // (T,T) buffer-pairwise overload of modInPlace.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void modInPlace<T>(this T targetDividend, T fromDivisor) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.compMod(targetDividend.Data.Ptr, fromDivisor.Data.Ptr, targetDividend.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(this T v, double s) where T : unmanaged, IUnsafedoubleArray
        {
            addInPlace(v, -s);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void subInPlace<T>(double s, T v) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {                 
                UnsafeOP.scalSub(s, v.Data.Ptr, v.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signFlipInPlace<T>(this T a) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe {
                UnsafeOP.signFlip(a.Data.Ptr, a.Data.Ptr, a.Data.Length);
            }
        }

        /// <summary>Clamp every element of <paramref name="x"/> to [<paramref name="lo"/>, <paramref name="hi"/>] in-place.
        /// Delegates to the UnsafeMathOP clamp kernel; no allocation.</summary>
        /// <remarks>Throws <c>ArgumentException</c> if <paramref name="lo"/> is greater than <paramref name="hi"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clampInPlace<T>(this T x, double lo, double hi) where T : unmanaged, IUnsafedoubleArray
        {
            if (lo > hi)
                throw new ArgumentException("clampInPlace: lo must be <= hi");
            unsafe
            {
                UnsafeMathOP.clamp(x.Data.Ptr, x.Data.Length, lo, hi);
            }
        }

        // ---- Componentwise math, forwarding to UnsafeMathOP. Every wrapper here is a thin, non-loop
        // passthrough - [AggressiveInlining] is load-bearing (the loop itself lives in the UnsafeMathOP
        // kernel, which is [MethodImpl(NoInlining)]). ----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void absInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.abs(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void signInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.sign(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sqrtInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.sqrt(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rsqrtInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.rsqrt(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void acosInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.acos(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void asinInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.asin(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void atanInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.atan(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ceilInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.ceil(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void floorInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.floor(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void roundInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.round(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cosInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.cos(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void coshInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.cosh(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sinInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.sin(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sinhInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.sinh(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void tanInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.tan(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void tanhInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.tanh(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void expInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.exp(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void exp2InPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.exp2(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void exp10InPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.exp10(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void logInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.log(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void log2InPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.log2(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void log10InPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.log10(x.Data.Ptr, x.Data.Length); }
        }

        /// <summary>acosh(x) = log(x + sqrt(x^2 - 1)), domain x &gt;= 1.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void acoshInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.acosh(x.Data.Ptr, x.Data.Length); }
        }

        /// <summary>x[i] = max(0, x[i]).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void reluInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.relu(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void powInPlace<T>(this T x, int exponent) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.pow(x.Data.Ptr, x.Data.Length, exponent); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lerpInPlace<T>(this T a, T b, double t) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.lerp(a.Data.Ptr, b.Data.Ptr, a.Data.Length, t); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void unlerpInPlace<T>(this T a, T b, double t) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.unlerp(a.Data.Ptr, b.Data.Ptr, a.Data.Length, t); }
        }

        // (T a, T b, double t) form: a[i] = smoothstep(a[i], b[i], t) - both edges are buffers.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void smoothstepInPlace<T>(this T a, T b, double t) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.smoothstep(a.Data.Ptr, b.Data.Ptr, a.Data.Length, t); }
        }

        // (T x, double edge0, double edge1) form: x[i] = smoothstep(edge0, edge1, x[i]) - scalar edges.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void smoothstepInPlace<T>(this T x, double edge0, double edge1) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.smoothstep(x.Data.Ptr, x.Data.Length, edge0, edge1); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void stepInPlace<T>(this T x, double edge) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.step(x.Data.Ptr, x.Data.Length, edge); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void saturateInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.saturate(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void fracInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.frac(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rcpInPlace<T>(this T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.rcp(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void madInPlace<T>(this T a, T b, T c) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.mad(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, a.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void remapInPlace<T>(this T x, double oldMin, double oldMax, double newMin, double newMax) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.remap(x.Data.Ptr, x.Data.Length, oldMin, oldMax, newMin, newMax); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void degreesInPlace<T>(this T radians) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.degrees(radians.Data.Ptr, radians.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void radiansInPlace<T>(this T degrees) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.radians(degrees.Data.Ptr, degrees.Data.Length); }
        }

        /// <summary>Writes sin(x[i]) into <paramref name="sin"/> and cos(x[i]) into <paramref name="cos"/>;
        /// <paramref name="x"/> itself is left unchanged. No "InPlace" suffix - the receiver is not
        /// mutated, so that suffix would be misleading here (unlike every other wrapper in this
        /// file).</summary>
        /// <remarks><paramref name="sin"/> and <paramref name="cos"/> must not alias <paramref name="x"/>
        /// (or each other) - the kernel reads x[i] while writing sin/cos, so an aliased buffer would
        /// read back a value it just overwrote.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sincos<T>(this T x, T sin, T cos) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.sincos(x.Data.Ptr, x.Data.Length, sin.Data.Ptr, cos.Data.Ptr); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void atan2InPlace<T>(this T y, T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.atan2(y.Data.Ptr, x.Data.Ptr, y.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void minInPlace<T>(this T x, T y) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.min(x.Data.Ptr, y.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxInPlace<T>(this T x, T y) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.max(x.Data.Ptr, y.Data.Ptr, x.Data.Length); }
        }

        /// <summary>x[i] = |x[i] - y[i]|. Componentwise (per-index scalar difference), NOT a
        /// whole-vector Euclidean distance - y is left unchanged.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void absDiffInPlace<T>(this T x, T y) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.absDiff(x.Data.Ptr, y.Data.Ptr, x.Data.Length); }
        }

        /// <summary>x[i] = (x[i] - y[i])^2. Componentwise, NOT a whole-vector squared distance - y is
        /// left unchanged.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sqrDiffInPlace<T>(this T x, T y) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe { UnsafeMathOP.sqrDiff(x.Data.Ptr, y.Data.Ptr, x.Data.Length); }
        }
    }
}
