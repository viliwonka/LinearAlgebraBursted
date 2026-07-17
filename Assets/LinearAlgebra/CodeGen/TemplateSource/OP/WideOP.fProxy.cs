using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
//+deleteThis
using LinearAlgebra.mathProxies; // TEMPLATE-ONLY: fProxy4 stub (-> float4/double4)
//-deleteThis

namespace LinearAlgebra.Internal
{
    // Widest per-dtype SIMD vector for level-1 kernels: 8 float lanes (one AVX ymm register)
    // or 4 double lanes (double4 — 256 bits is already double's full AVX2 width). Kernels
    // written against fProxyW get the full register width for BOTH element types from one
    // template body. Storage is always a 32-byte v256; the double side reinterprets it as a
    // double4 (same size, zero-cost) and uses native vector math.
    //
    // Numeric contract: every operation is lane-independent except HSum, whose fold order is
    // FIXED below (halves first, then the balanced width-4 tree). A kernel's summation tree is
    // therefore identical between the AVX path, the non-AVX lane-wise fallback, and Mono —
    // bit-identical results on every path. The float fallback is a correctness path (pre-2011
    // CPUs and managed execution), not a fast path.
    internal unsafe struct fProxyW
    {
        /// <summary>Lanes per vector: 8 (float) / 4 (double).</summary>
        /// <remarks>The template placeholder must be 8: the template assembly's own tests RUN
        /// this code with the 4-byte float-backed fProxy stub, so the template-compiled variant
        /// must be the float (8-lane) configuration or every wide load overlaps.</remarks>
        public const int Width = /*+choose[8|4]*/8/*-choose*/;

        public v256 v;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW Load([NoAlias] fProxy* p, int i)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_loadu_ps(p + (long)i * Width) };
            //-skipFor
            return new fProxyW { v = *(v256*)(p + (long)i * Width) };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Store([NoAlias] fProxy* p, int i, fProxyW w)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
            {
                X86.Avx.mm256_storeu_ps(p + (long)i * Width, w.v);
                return;
            }
            //-skipFor
            *(v256*)(p + (long)i * Width) = w.v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW Splat(fProxy s)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_set1_ps(s) };
            return new fProxyW { v = new v256(s, s, s, s, s, s, s, s) };
            //-skipFor
            //+emitFor[double]
            //!fProxy4 r = new fProxy4(s, s, s, s);
            //!return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            //-emitFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW operator -(fProxyW a, fProxyW b)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_sub_ps(a.v, b.v) };
            return new fProxyW { v = new v256(
                a.v.Float0 - b.v.Float0, a.v.Float1 - b.v.Float1,
                a.v.Float2 - b.v.Float2, a.v.Float3 - b.v.Float3,
                a.v.Float4 - b.v.Float4, a.v.Float5 - b.v.Float5,
                a.v.Float6 - b.v.Float6, a.v.Float7 - b.v.Float7) };
            //-skipFor
            //+emitFor[double]
            //!fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
            //!fProxy4 bv = UnsafeUtility.As<v256, fProxy4>(ref b.v);
            //!fProxy4 r = av - bv;
            //!return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            //-emitFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW operator +(fProxyW a, fProxyW b)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_add_ps(a.v, b.v) };
            return new fProxyW { v = new v256(
                a.v.Float0 + b.v.Float0, a.v.Float1 + b.v.Float1,
                a.v.Float2 + b.v.Float2, a.v.Float3 + b.v.Float3,
                a.v.Float4 + b.v.Float4, a.v.Float5 + b.v.Float5,
                a.v.Float6 + b.v.Float6, a.v.Float7 + b.v.Float7) };
            //-skipFor
            //+skipFor[float]
            {
                fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
                fProxy4 bv = UnsafeUtility.As<v256, fProxy4>(ref b.v);
                fProxy4 r = av + bv;
                return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            }
            //-skipFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW operator *(fProxyW a, fProxyW b)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_mul_ps(a.v, b.v) };
            return new fProxyW { v = new v256(
                a.v.Float0 * b.v.Float0, a.v.Float1 * b.v.Float1,
                a.v.Float2 * b.v.Float2, a.v.Float3 * b.v.Float3,
                a.v.Float4 * b.v.Float4, a.v.Float5 * b.v.Float5,
                a.v.Float6 * b.v.Float6, a.v.Float7 * b.v.Float7) };
            //-skipFor
            //+skipFor[float]
            {
                fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
                fProxy4 bv = UnsafeUtility.As<v256, fProxy4>(ref b.v);
                fProxy4 r = av * bv;
                return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            }
            //-skipFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW operator /(fProxyW a, fProxyW b)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_div_ps(a.v, b.v) };
            return new fProxyW { v = new v256(
                a.v.Float0 / b.v.Float0, a.v.Float1 / b.v.Float1,
                a.v.Float2 / b.v.Float2, a.v.Float3 / b.v.Float3,
                a.v.Float4 / b.v.Float4, a.v.Float5 / b.v.Float5,
                a.v.Float6 / b.v.Float6, a.v.Float7 / b.v.Float7) };
            //-skipFor
            //+skipFor[float]
            {
                fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
                fProxy4 bv = UnsafeUtility.As<v256, fProxy4>(ref b.v);
                fProxy4 r = av / bv;
                return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            }
            //-skipFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW Abs(fProxyW a)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_and_ps(a.v, X86.Avx.mm256_set1_ps(math.asfloat(0x7FFFFFFF))) };
            return new fProxyW { v = new v256(
                math.abs(a.v.Float0), math.abs(a.v.Float1), math.abs(a.v.Float2), math.abs(a.v.Float3),
                math.abs(a.v.Float4), math.abs(a.v.Float5), math.abs(a.v.Float6), math.abs(a.v.Float7)) };
            //-skipFor
            //+emitFor[double]
            //!fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
            //!fProxy4 r = math.abs(av);
            //!return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            //-emitFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW Max(fProxyW a, fProxyW b)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_max_ps(a.v, b.v) };
            return new fProxyW { v = new v256(
                math.max(a.v.Float0, b.v.Float0), math.max(a.v.Float1, b.v.Float1),
                math.max(a.v.Float2, b.v.Float2), math.max(a.v.Float3, b.v.Float3),
                math.max(a.v.Float4, b.v.Float4), math.max(a.v.Float5, b.v.Float5),
                math.max(a.v.Float6, b.v.Float6), math.max(a.v.Float7, b.v.Float7)) };
            //-skipFor
            //+emitFor[double]
            //!fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
            //!fProxy4 bv = UnsafeUtility.As<v256, fProxy4>(ref b.v);
            //!fProxy4 r = math.max(av, bv);
            //!return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            //-emitFor
        }

        // Fixed max-fold companion to HSum (max is exact, so the order only matters for a
        // consistent NaN story — kept fixed anyway).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy HMax(fProxyW a)
        {
            //+skipFor[double]
            {
                fProxy h0 = math.max(a.v.Float0, a.v.Float4);
                fProxy h1 = math.max(a.v.Float1, a.v.Float5);
                fProxy h2 = math.max(a.v.Float2, a.v.Float6);
                fProxy h3 = math.max(a.v.Float3, a.v.Float7);
                return math.max(math.max(h0, h1), math.max(h2, h3));
            }
            //-skipFor
            //+emitFor[double]
            //!fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
            //!return math.max(math.max(av.x, av.y), math.max(av.z, av.w));
            //-emitFor
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyW Min(fProxyW a, fProxyW b)
        {
            //+skipFor[double]
            if (X86.Avx.IsAvxSupported)
                return new fProxyW { v = X86.Avx.mm256_min_ps(a.v, b.v) };
            return new fProxyW { v = new v256(
                math.min(a.v.Float0, b.v.Float0), math.min(a.v.Float1, b.v.Float1),
                math.min(a.v.Float2, b.v.Float2), math.min(a.v.Float3, b.v.Float3),
                math.min(a.v.Float4, b.v.Float4), math.min(a.v.Float5, b.v.Float5),
                math.min(a.v.Float6, b.v.Float6), math.min(a.v.Float7, b.v.Float7)) };
            //-skipFor
            //+emitFor[double]
            //!fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
            //!fProxy4 bv = UnsafeUtility.As<v256, fProxy4>(ref b.v);
            //!fProxy4 r = math.min(av, bv);
            //!return new fProxyW { v = UnsafeUtility.As<fProxy4, v256>(ref r) };
            //-emitFor
        }

        // Fixed min-fold companion to HMax (min is exact; order only matters for a consistent NaN
        // story — kept fixed anyway).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy HMin(fProxyW a)
        {
            //+skipFor[double]
            {
                fProxy h0 = math.min(a.v.Float0, a.v.Float4);
                fProxy h1 = math.min(a.v.Float1, a.v.Float5);
                fProxy h2 = math.min(a.v.Float2, a.v.Float6);
                fProxy h3 = math.min(a.v.Float3, a.v.Float7);
                return math.min(math.min(h0, h1), math.min(h2, h3));
            }
            //-skipFor
            //+emitFor[double]
            //!fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
            //!return math.min(math.min(av.x, av.y), math.min(av.z, av.w));
            //-emitFor
        }

        // Fixed fold — part of every consuming kernel's frozen numeric contract: opposite
        // halves pair first (lane l + lane l+W/2), then the balanced width-4 tree.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy HSum(fProxyW a)
        {
            //+skipFor[double]
            {
                fProxy h0 = a.v.Float0 + a.v.Float4;
                fProxy h1 = a.v.Float1 + a.v.Float5;
                fProxy h2 = a.v.Float2 + a.v.Float6;
                fProxy h3 = a.v.Float3 + a.v.Float7;
                return (h0 + h1) + (h2 + h3);
            }
            //-skipFor
            //+skipFor[float]
            {
                fProxy4 av = UnsafeUtility.As<v256, fProxy4>(ref a.v);
                return (av.x + av.y) + (av.z + av.w);
            }
            //-skipFor
        }
    }
}
