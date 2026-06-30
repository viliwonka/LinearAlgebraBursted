#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class float_OP {

        /// <summary>
        /// Trace: the sum of the diagonal entries, Σ A[i,i]. A must be square.
        /// </summary>
        public static float trace(in floatMxN A)
        {
            if (!A.IsSquare)
                throw new ArgumentException("trace: A must be square");

            float sum = (float)0;
            for (int i = 0; i < A.M_Rows; i++)
                sum += A[i, i];
            return sum;
        }

        /// <summary>
        /// 2-norm condition number κ₂(A) = σ_max / σ_min (any shape, via SVD). Returns positive
        /// infinity when A is singular / rank-deficient (σ_min == 0). Allocates SVD scratch;
        /// A is not modified. κ₂ ≈ 1 means well-conditioned; large κ₂ means ill-conditioned.
        /// </summary>
        public static float cond(in floatMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (float)0;

            floatN S = A.tempfloatVec(k);
            SVD.singularValues(in A, ref S);

            float sMin = S[k - 1];          // singular values are descending
            if (!(sMin > (float)0))         // NaN-safe: singular -> infinite condition number
                return float.PositiveInfinity;

            return S[0] / sMin;
        }

        /// <summary>
        /// Numerical rank: the number of singular values greater than relTol * σ_max (any shape,
        /// via SVD). relTol &lt; 0 selects the automatic tolerance max(m, n) * Consts.floatZeroThreshold
        /// (matching pinvSolve). Allocates SVD scratch; A is not modified.
        /// </summary>
        public static int rank(in floatMxN A, float relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            if (k == 0)
                return 0;

            floatN S = A.tempfloatVec(k);
            SVD.singularValues(in A, ref S);

            if (S[0] == (float)0)
                return 0;

            if (relTol < (float)0)
                relTol = (float)math.max(m, n) * Consts.floatZeroThreshold;

            float tol = relTol * S[0];
            int r = 0;
            for (int i = 0; i < k; i++)
                if (S[i] > tol)
                    r++;
            return r;
        }

        /// <summary>Numerical rank with the automatic tolerance (relTol &lt; 0).</summary>
        public static int rank(in floatMxN A) => rank(in A, (float)(-1));
    }
}
