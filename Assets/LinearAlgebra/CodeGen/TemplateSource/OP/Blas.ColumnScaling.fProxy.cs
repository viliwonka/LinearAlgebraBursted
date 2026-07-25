using System;

using Unity.Mathematics;

namespace BULA
{
    // Blas column-norm / column-scaling kernels: the ingredients for an AᵀA-Jacobi
    // (column-equilibration) least-squares preconditioner. These are computational primitives
    // (not scalar characterizations -- trace/cond/rank live on Analysis).
    public static partial class Blas {

        /// <summary>
        /// Squared L2 norm of each column: d2[j] = Σ_i A[i,j]² = diag(AᵀA)[j]. Written into the
        /// caller's <paramref name="d2"/> (length A.N_Cols), no allocation. This is the diagonal of
        /// the normal matrix computed directly from storage in one O(mn) pass -- the ingredient for
        /// an AᵀA-Jacobi (column-equilibration) least-squares preconditioner (see
        /// <see cref="fProxyColScaledOperator{TInner}"/> / <c>buildJacobiScale</c>) -- without ever
        /// forming AᵀA or running n transpose-matvecs.
        /// </summary>
        public static void columnNormsSquared(in fProxyMxN A, ref fProxyN d2)
        {
            if (d2.N != A.N_Cols)
                throw new ArgumentException("columnNormsSquared: d2.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = (fProxy)0;
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
        /// allocation. Pairs with <see cref="columnNormsSquared(in fProxyMxN, ref fProxyN)"/> (or
        /// its BSR sibling) to right-precondition a least-squares solve via
        /// <see cref="fProxyColScaledOperator{TInner}"/>: solve (A·diag(d)) y = b, then x = diag(d)·y,
        /// which equilibrates the normal-equation diagonal so ill-conditioned LS converges faster.
        /// </summary>
        public static void buildJacobiScale(in fProxyN colNorm2, ref fProxyN d)
        {
            if (d.N != colNorm2.N)
                throw new ArgumentException("buildJacobiScale: d.N must equal colNorm2.N");

            for (int j = 0; j < colNorm2.N; j++)
            {
                fProxy c = colNorm2[j];
                d[j] = math.select((fProxy)1, (fProxy)1 / math.sqrt(c), c > (fProxy)0);   // NaN-safe: !(c>0) -> 1
            }
        }
    }
}
