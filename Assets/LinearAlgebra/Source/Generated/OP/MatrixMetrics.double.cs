#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class Linear_OP {

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
        /// Squared L2 norm of each column: d2[j] = Σ_i A[i,j]² = diag(AᵀA)[j]. Written into the
        /// caller's <paramref name="d2"/> (length A.N_Cols), no allocation. This is the diagonal of
        /// the normal matrix computed directly from storage in one O(mn) pass -- the ingredient for
        /// an AᵀA-Jacobi (column-equilibration) least-squares preconditioner (see
        /// <see cref="doubleColScaledOperator{TInner}"/> / <c>buildJacobiScale</c>) -- without ever
        /// forming AᵀA or running n transpose-matvecs.
        /// </summary>
        public static void columnNormsSquared(in doubleMxN A, ref doubleN d2)
        {
            if (d2.N != A.N_Cols)
                throw new ArgumentException("columnNormsSquared: d2.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = (double)0;
                for (int r = 0; r < A.M_Rows; r++)
                    s += A[r, c] * A[r, c];
                d2[c] = s;
            }
        }

        /// <summary>
        /// Build the Jacobi (column-equilibration) scale vector from column norms²:
        /// d[j] = 1/‖A_:,j‖₂ = 1/sqrt(colNorm2[j]), with the NaN-safe convention d[j] = 1 for a
        /// zero / degenerate column (colNorm2[j] &lt;= 0 or NaN) -- that column is left UNSCALED
        /// rather than dividing by zero. Written into the caller's d (length colNorm2.N), no
        /// allocation. Pairs with <see cref="columnNormsSquared(in doubleMxN, ref doubleN)"/> (or
        /// its BSM sibling) to right-precondition a least-squares solve via
        /// <see cref="doubleColScaledOperator{TInner}"/>: solve (A·diag(d)) y = b, then x = diag(d)·y,
        /// which equilibrates the normal-equation diagonal so ill-conditioned LS converges faster.
        /// </summary>
        public static void buildJacobiScale(in doubleN colNorm2, ref doubleN d)
        {
            if (d.N != colNorm2.N)
                throw new ArgumentException("buildJacobiScale: d.N must equal colNorm2.N");

            for (int j = 0; j < colNorm2.N; j++)
            {
                double c = colNorm2[j];
                d[j] = (c > (double)0) ? (double)1 / math.sqrt(c) : (double)1;   // NaN-safe: !(c>0) -> 1
            }
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
        /// via SVD). relTol &lt; 0 selects the automatic tolerance max(m, n) * Consts.doubleZeroThreshold
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
                relTol = (double)math.max(m, n) * Consts.doubleZeroThreshold;

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
