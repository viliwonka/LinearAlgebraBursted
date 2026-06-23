#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class doubleOP {

        /// <summary>
        /// Trace: the sum of the diagonal entries, Σ A[i,i]. A must be square.
        /// </summary>
        public static double trace(in doubleMxN A)
        {
            if (!A.IsSquare)
                throw new ArgumentException("trace: A must be square");

            double sum = (double)0;
            for (int i = 0; i < A.M_Rows; i++)
                sum += A[i, i];
            return sum;
        }

        /// <summary>
        /// 2-norm condition number κ₂(A) = σ_max / σ_min (any shape, via SVD). Returns positive
        /// infinity when A is singular / rank-deficient (σ_min == 0). Allocates SVD scratch;
        /// A is not modified. κ₂ ≈ 1 means well-conditioned; large κ₂ means ill-conditioned.
        /// </summary>
        public static double cond(in doubleMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (double)0;

            doubleN S = A.tempdoubleVec(k);
            SVD.singularValues(in A, ref S);

            double sMin = S[k - 1];          // singular values are descending
            if (!(sMin > (double)0))         // NaN-safe: singular -> infinite condition number
                return double.PositiveInfinity;

            return S[0] / sMin;
        }

        /// <summary>
        /// Numerical rank: the number of singular values greater than relTol * σ_max (any shape,
        /// via SVD). relTol &lt; 0 selects the automatic tolerance max(m, n) * Consts.doubleZeroTreshold
        /// (matching pinvSolve). Allocates SVD scratch; A is not modified.
        /// </summary>
        public static int rank(in doubleMxN A, double relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            if (k == 0)
                return 0;

            doubleN S = A.tempdoubleVec(k);
            SVD.singularValues(in A, ref S);

            if (S[0] == (double)0)
                return 0;

            if (relTol < (double)0)
                relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;

            double tol = relTol * S[0];
            int r = 0;
            for (int i = 0; i < k; i++)
                if (S[i] > tol)
                    r++;
            return r;
        }

        /// <summary>Numerical rank with the automatic tolerance (relTol &lt; 0).</summary>
        public static int rank(in doubleMxN A) => rank(in A, (double)(-1));
    }
}
