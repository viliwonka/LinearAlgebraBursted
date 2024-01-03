#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    public static partial class floatNormsOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float L2<T>(in T a) where T : unmanaged, IUnsafefloatArray {

            unsafe
            {
                return math.sqrt(UnsafeOP.vecDot(a.Data.Ptr, a.Data.Ptr, a.Data.Length));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float L1<T>(in T a) where T : unmanaged, IUnsafefloatArray {

            unsafe {
                return UnsafeOP.sumAbs(a.Data.Ptr, a.Data.Length) / a.Data.Length;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float L2Range(in floatN a, int start, int end)
        {
            if (start >= end)
                throw new System.Exception("NormsOP.L2: start must be less than end");

            if (start < 0 || end > a.Data.Length)
                throw new System.Exception("NormsOP.L2: start and end must be within bounds of vector");

            unsafe
            {
                return math.sqrt(UnsafeOP.vecDotRange(a.Data.Ptr, a.Data.Ptr, start, end));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void NormalizeL2<T>(in T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                UnsafeOP.normalizeL2Inpl(x.Data.Ptr, x.Data.Length);
            }
        }

        // returns length before normalization
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeL2<T>(in T x, int start, int end) where T : unmanaged, IUnsafefloatArray
        {
            if (start >= end)
                throw new System.Exception("NormalizeL2: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new System.Exception("NormalizeL2: start and end must be within bounds of vector");

            unsafe
            {
                return UnsafeOP.normalizeL2Inpl(x.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeL1<T>(in T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                return UnsafeOP.normalizeL1(x.Data.Ptr, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeL1<T>(in T x, int start, int end) where T : unmanaged, IUnsafefloatArray
        {
            if (start >= end)
                throw new System.Exception("NormalizeL1: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new System.Exception("NormalizeL1: start and end must be within bounds of vector");

            unsafe
            {
                return UnsafeOP.normalizeL1(x.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeLMax<T>(in T x) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                return UnsafeOP.normalizeLMax(x.Data.Ptr, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeLMax<T>(in T x, int start, int end) where T : unmanaged, IUnsafefloatArray
        {
            if (start >= end)
                throw new System.Exception("NormalizeLMax: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new System.Exception("NormalizeLMax: start and end must be within bounds of vector");

            unsafe
            {
                return UnsafeOP.normalizeLMax(x.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeLP<T>(in T x, float p) where T : unmanaged, IUnsafefloatArray
        {
            unsafe
            {
                return UnsafeOP.normalizeLP(x.Data.Ptr, x.Data.Length, p);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalizeLP<T>(in T x, int start, int end, float p) where T : unmanaged, IUnsafefloatArray
        {
            if (start >= end)
                throw new System.Exception("NormalizeLP: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new System.Exception("NormalizeLP: start and end must be within bounds of vector");

            unsafe
            {
                return UnsafeOP.normalizeLP(x.Data.Ptr, start, end, p);
            }
        }
    }
}
