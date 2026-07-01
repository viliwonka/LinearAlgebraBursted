#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class doubleNorms_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double L2<T>(in T a) where T : unmanaged, IUnsafedoubleArray {

            unsafe
            {
                return math.sqrt(Unsafe_OP.vecDot(a.Data.Ptr, a.Data.Ptr, a.Data.Length));
            }
        }

        // Standard L1 norm: the sum of absolute values, Σ|xᵢ| (NOT averaged by length).
        // Naïve accumulation (no Kahan/pairwise compensation): accurate at moderate sizes; very
        // long float vectors may lose precision. The same caveat applies to matrixL1 / matrixLInf.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double L1<T>(in T a) where T : unmanaged, IUnsafedoubleArray {

            unsafe {
                return Unsafe_OP.sumAbs(a.Data.Ptr, a.Data.Length);
            }
        }

        // L-infinity (max-abs) norm: the largest absolute element, max_i |xᵢ|.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double LInf<T>(in T a) where T : unmanaged, IUnsafedoubleArray {

            unsafe {
                return Unsafe_OP.maxAbs(a.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double L2Range(in doubleN a, int start, int end)
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
        public static void NormalizeL2<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                Unsafe_OP.normalizeL2Inpl(x.Data.Ptr, x.Data.Length);
            }
        }

        // returns length before normalization
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double NormalizeL2<T>(in T x, int start, int end) where T : unmanaged, IUnsafedoubleArray
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
        public static double NormalizeL1<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                return Unsafe_OP.normalizeL1(x.Data.Ptr, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double NormalizeL1<T>(in T x, int start, int end) where T : unmanaged, IUnsafedoubleArray
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
        public static double NormalizeLMax<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                return Unsafe_OP.normalizeLMax(x.Data.Ptr, x.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double NormalizeLMax<T>(in T x, int start, int end) where T : unmanaged, IUnsafedoubleArray
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
        public static double NormalizeLP<T>(in T x, double p) where T : unmanaged, IUnsafedoubleArray
        {
            unsafe
            {
                return Unsafe_OP.normalizeLP(x.Data.Ptr, x.Data.Length, p);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double NormalizeLP<T>(in T x, int start, int end, double p) where T : unmanaged, IUnsafedoubleArray
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
        public static void Normalize<T>(in T x, Norm n) where T : unmanaged, IUnsafedoubleArray
        {
            switch (n)
            {
                case Norm.L1:   NormalizeL1(in x);   break;
                case Norm.L2:   NormalizeL2(in x);   break;
                default:        NormalizeLMax(in x);  break; // Linf
            }
        }

        // NormalizeRows: normalize each row of A to unit norm (by the chosen norm).
        // Zero-norm row → left at 0 (NaN-safe !(norm > 0) guard). No allocation.
        public static void NormalizeRows(ref doubleMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;

            for (int r = 0; r < A.M_Rows; r++)
            {
                double rowNorm;
                switch (n)
                {
                    case Norm.L1:
                    {
                        double s = 0f;
                        for (int c = 0; c < A.N_Cols; c++) s += math.abs(A[r, c]);
                        rowNorm = s;
                        break;
                    }
                    case Norm.L2:
                    {
                        double s = 0f;
                        for (int c = 0; c < A.N_Cols; c++) s += A[r, c] * A[r, c];
                        rowNorm = math.sqrt(s);
                        break;
                    }
                    default: // Linf
                    {
                        double s = 0f;
                        for (int c = 0; c < A.N_Cols; c++) s = math.max(s, math.abs(A[r, c]));
                        rowNorm = s;
                        break;
                    }
                }

                if (!(rowNorm > 0f)) continue; // zero-norm row → leave unchanged

                double inv = (double)1f / rowNorm;
                for (int c = 0; c < A.N_Cols; c++) A[r, c] *= inv;
            }
        }

        // NormalizeColumns: normalize each column of A to unit norm (by the chosen norm).
        // Zero-norm column → left at 0 (NaN-safe !(norm > 0) guard). No allocation.
        public static void NormalizeColumns(ref doubleMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;

            for (int c = 0; c < A.N_Cols; c++)
            {
                double colNorm;
                switch (n)
                {
                    case Norm.L1:
                    {
                        double s = 0f;
                        for (int r = 0; r < A.M_Rows; r++) s += math.abs(A[r, c]);
                        colNorm = s;
                        break;
                    }
                    case Norm.L2:
                    {
                        double s = 0f;
                        for (int r = 0; r < A.M_Rows; r++) s += A[r, c] * A[r, c];
                        colNorm = math.sqrt(s);
                        break;
                    }
                    default: // Linf
                    {
                        double s = 0f;
                        for (int r = 0; r < A.M_Rows; r++) s = math.max(s, math.abs(A[r, c]));
                        colNorm = s;
                        break;
                    }
                }

                if (!(colNorm > 0f)) continue; // zero-norm column → leave unchanged

                double inv = (double)1f / colNorm;
                for (int r = 0; r < A.M_Rows; r++) A[r, c] *= inv;
            }
        }

        // ---- Induced (operator) matrix norms ----

        // Induced 1-norm ‖A‖₁: the maximum absolute column sum, max_j Σ_i |A[i,j]|. Allocation-free.
        public static double matrixL1(in doubleMxN A)
        {
            double best = (double)0;
            for (int j = 0; j < A.N_Cols; j++)
            {
                double colSum = (double)0;
                for (int i = 0; i < A.M_Rows; i++)
                    colSum += math.abs(A[i, j]);
                if (colSum > best)
                    best = colSum;
            }
            return best;
        }

        // Induced ∞-norm ‖A‖∞: the maximum absolute row sum, max_i Σ_j |A[i,j]|. Allocation-free.
        public static double matrixLInf(in doubleMxN A)
        {
            double best = (double)0;
            for (int i = 0; i < A.M_Rows; i++)
            {
                double rowSum = (double)0;
                for (int j = 0; j < A.N_Cols; j++)
                    rowSum += math.abs(A[i, j]);
                if (rowSum > best)
                    best = rowSum;
            }
            return best;
        }

        // Induced 2-norm (spectral norm) ‖A‖₂ = σ_max(A), the largest singular value. Runs a
        // one-sided Jacobi SVD on a copy (A is not modified); allocates SVD scratch from A's arena.
        public static double matrixL2(in doubleMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (double)0;

            doubleN S = A.tempdoubleVec(k);
            SVD.singularValues(in A, ref S);
            return S[0];   // singular values are sorted descending -> σ_max
        }
    }
}
