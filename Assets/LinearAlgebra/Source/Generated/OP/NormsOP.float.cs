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

        // Standard L1 norm: the sum of absolute values, Σ|xᵢ| (NOT averaged by length).
        // Naïve accumulation (no Kahan/pairwise compensation): accurate at moderate sizes; very
        // long float vectors may lose precision. The same caveat applies to matrixL1 / matrixLInf.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float L1<T>(in T a) where T : unmanaged, IUnsafefloatArray {

            unsafe {
                return UnsafeOP.sumAbs(a.Data.Ptr, a.Data.Length);
            }
        }

        // L-infinity (max-abs) norm: the largest absolute element, max_i |xᵢ|.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LInf<T>(in T a) where T : unmanaged, IUnsafefloatArray {

            unsafe {
                return UnsafeOP.maxAbs(a.Data.Ptr, a.Data.Length);
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

        // ---- Induced (operator) matrix norms ----

        // Induced 1-norm ‖A‖₁: the maximum absolute column sum, max_j Σ_i |A[i,j]|. Allocation-free.
        public static float matrixL1(in floatMxN A)
        {
            float best = (float)0;
            for (int j = 0; j < A.N_Cols; j++)
            {
                float colSum = (float)0;
                for (int i = 0; i < A.M_Rows; i++)
                    colSum += math.abs(A[i, j]);
                if (colSum > best)
                    best = colSum;
            }
            return best;
        }

        // Induced ∞-norm ‖A‖∞: the maximum absolute row sum, max_i Σ_j |A[i,j]|. Allocation-free.
        public static float matrixLInf(in floatMxN A)
        {
            float best = (float)0;
            for (int i = 0; i < A.M_Rows; i++)
            {
                float rowSum = (float)0;
                for (int j = 0; j < A.N_Cols; j++)
                    rowSum += math.abs(A[i, j]);
                if (rowSum > best)
                    best = rowSum;
            }
            return best;
        }

        // Induced 2-norm (spectral norm) ‖A‖₂ = σ_max(A), the largest singular value. Runs a
        // one-sided Jacobi SVD on a copy (A is not modified); allocates SVD scratch from A's arena.
        public static float matrixL2(in floatMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (float)0;

            floatN S = A.tempfloatVec(k);
            SVD.singularValues(in A, ref S);
            return S[0];   // singular values are sorted descending -> σ_max
        }
    }
}
