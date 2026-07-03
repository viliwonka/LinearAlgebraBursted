#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class fProxyNorms_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L2<T>(in T a) where T : unmanaged, IUnsafefProxyArray {

            unsafe
            {
                return math.sqrt(Unsafe_OP.vecDot(a.Data.Ptr, a.Data.Ptr, a.Data.Length));
            }
        }

        // Standard L1 norm: the sum of absolute values, Σ|xᵢ| (NOT averaged by length).
        // Naïve accumulation (no Kahan/pairwise compensation): accurate at moderate sizes; very
        // long float vectors may lose precision. The same caveat applies to matrixL1 / matrixLInf.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L1<T>(in T a) where T : unmanaged, IUnsafefProxyArray {

            unsafe {
                return Unsafe_OP.sumAbs(a.Data.Ptr, a.Data.Length);
            }
        }

        // L-infinity (max-abs) norm: the largest absolute element, max_i |xᵢ|.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy LInf<T>(in T a) where T : unmanaged, IUnsafefProxyArray {

            unsafe {
                return Unsafe_OP.maxAbs(a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L2Range(in fProxyN a, int start, int end)
        {
            if (start >= end)
                throw new ArgumentException("NormsOP.L2: start must be less than end");

            if (start < 0 || end > a.Data.Length)
                throw new ArgumentOutOfRangeException("NormsOP.L2: start and end must be within bounds of vector");

            unsafe
            {
                return math.sqrt(Unsafe_OP.vecDotRange(a.Data.Ptr, a.Data.Ptr, start, end));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void NormalizeL2<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe
            {
                Unsafe_OP.normalizeL2Inpl(x.Data.Ptr, x.Data.Length);
            }
        }

        // returns length before normalization
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeL2<T>(in T x, int start, int end) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeL2: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeL2: start and end must be within bounds of vector");

            unsafe
            {
                return Unsafe_OP.normalizeL2Inpl(x.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeL1<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe
            {
                return Unsafe_OP.normalizeL1(x.Data.Ptr, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeL1<T>(in T x, int start, int end) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeL1: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeL1: start and end must be within bounds of vector");

            unsafe
            {
                return Unsafe_OP.normalizeL1(x.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLMax<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe
            {
                return Unsafe_OP.normalizeLMax(x.Data.Ptr, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLMax<T>(in T x, int start, int end) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeLMax: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeLMax: start and end must be within bounds of vector");

            unsafe
            {
                return Unsafe_OP.normalizeLMax(x.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLP<T>(in T x, fProxy p) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe
            {
                return Unsafe_OP.normalizeLP(x.Data.Ptr, x.Data.Length, p);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLP<T>(in T x, int start, int end, fProxy p) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeLP: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeLP: start and end must be within bounds of vector");

            unsafe
            {
                return Unsafe_OP.normalizeLP(x.Data.Ptr, start, end, p);
            }
        }

        // ---- Enum-dispatch normalize ----

        /// <summary>Normalize x to unit norm in-place, using the specified <paramref name="n"/> (L1/L2/Linf).
        /// Delegates to the corresponding <c>NormalizeL1</c>/<c>NormalizeL2</c>/<c>NormalizeLMax</c> kernel.</summary>
        /// <remarks>Flat form — treats the input as one 1-D array. For a matrix this is the
        /// <b>whole-matrix</b> scope (all elements as a single distribution); use
        /// <see cref="NormalizeRows"/> or <see cref="NormalizeColumns"/> for per-axis normalization.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Normalize<T>(in T x, Norm n) where T : unmanaged, IUnsafefProxyArray
        {
            switch (n)
            {
                case Norm.L1:   NormalizeL1(in x);   break;
                case Norm.L2:   NormalizeL2(in x);   break;
                default:        NormalizeLMax(in x);  break; // Linf
            }
        }

        // Zero-norm row → left at 0 (NaN-safe !(norm > 0) guard). No allocation.
        public static void NormalizeRows(ref fProxyMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy rowNorm;
                switch (n)
                {
                    case Norm.L1:
                    {
                        fProxy s = 0f;
                        for (int c = 0; c < A.N_Cols; c++) s += math.abs(A[r, c]);
                        rowNorm = s;
                        break;
                    }
                    case Norm.L2:
                    {
                        fProxy s = 0f;
                        for (int c = 0; c < A.N_Cols; c++) s += A[r, c] * A[r, c];
                        rowNorm = math.sqrt(s);
                        break;
                    }
                    default: // Linf
                    {
                        fProxy s = 0f;
                        for (int c = 0; c < A.N_Cols; c++) s = math.max(s, math.abs(A[r, c]));
                        rowNorm = s;
                        break;
                    }
                }

                if (!(rowNorm > 0f)) continue; // zero-norm row → leave unchanged

                fProxy inv = (fProxy)1f / rowNorm;
                for (int c = 0; c < A.N_Cols; c++) A[r, c] *= inv;
            }
        }

        // Zero-norm column → left at 0 (NaN-safe !(norm > 0) guard). No allocation.
        public static void NormalizeColumns(ref fProxyMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy colNorm;
                switch (n)
                {
                    case Norm.L1:
                    {
                        fProxy s = 0f;
                        for (int r = 0; r < A.M_Rows; r++) s += math.abs(A[r, c]);
                        colNorm = s;
                        break;
                    }
                    case Norm.L2:
                    {
                        fProxy s = 0f;
                        for (int r = 0; r < A.M_Rows; r++) s += A[r, c] * A[r, c];
                        colNorm = math.sqrt(s);
                        break;
                    }
                    default: // Linf
                    {
                        fProxy s = 0f;
                        for (int r = 0; r < A.M_Rows; r++) s = math.max(s, math.abs(A[r, c]));
                        colNorm = s;
                        break;
                    }
                }

                if (!(colNorm > 0f)) continue; // zero-norm column → leave unchanged

                fProxy inv = (fProxy)1f / colNorm;
                for (int r = 0; r < A.M_Rows; r++) A[r, c] *= inv;
            }
        }

        // ---- Induced (operator) matrix norms ----

        // Induced 1-norm ‖A‖₁: the maximum absolute column sum, max_j Σ_i |A[i,j]|. Allocation-free.
        public static fProxy matrixL1(in fProxyMxN A)
        {
            fProxy best = (fProxy)0;
            for (int j = 0; j < A.N_Cols; j++)
            {
                fProxy colSum = (fProxy)0;
                for (int i = 0; i < A.M_Rows; i++)
                    colSum += math.abs(A[i, j]);
                if (colSum > best)
                    best = colSum;
            }
            return best;
        }

        // Induced ∞-norm ‖A‖∞: the maximum absolute row sum, max_i Σ_j |A[i,j]|. Allocation-free.
        public static fProxy matrixLInf(in fProxyMxN A)
        {
            fProxy best = (fProxy)0;
            for (int i = 0; i < A.M_Rows; i++)
            {
                fProxy rowSum = (fProxy)0;
                for (int j = 0; j < A.N_Cols; j++)
                    rowSum += math.abs(A[i, j]);
                if (rowSum > best)
                    best = rowSum;
            }
            return best;
        }

        // Induced 2-norm (spectral norm) ‖A‖₂ = σ_max(A), the largest singular value. Runs a
        // one-sided Jacobi SVD on a copy (A is not modified); allocates SVD scratch from A's arena.
        public static fProxy matrixL2(in fProxyMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (fProxy)0;

            fProxyN S = A.tempfProxyVec(k);
            SVD.singularValues(in A, ref S);
            return S[0];   // singular values are sorted descending -> σ_max
        }
    }
}
