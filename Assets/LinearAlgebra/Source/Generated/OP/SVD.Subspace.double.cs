#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD {

        // Fundamental-subspace bases from the SVD A = U diag(S) Vᵀ. With a numerical rank r =
        // #{ S[j] > tol } (tol = relTol * S[0]; relTol < 0 -> auto = max(m,n)*eps), the trailing
        // right-singular vectors span the NULLSPACE and the leading left-singular vectors span the
        // RANGE (column space). A is m x n with m >= n (the same precondition as svdGolubKahan); the
        // wide m < n case needs the orthogonal complement of a thin factor and is left for later.

        /// <summary>
        /// Orthonormal basis for the NULLSPACE (kernel) of A (m x n, m >= n): the set of x with Ax = 0.
        /// From A = U diag(S) Vᵀ, the nullspace is spanned by the right-singular vectors (columns of V)
        /// whose singular value is negligible. Columns 0..dim-1 of <paramref name="basis"/> (n x n,
        /// caller-allocated) receive those vectors (orthonormal); the RETURN VALUE is dim = n - rank.
        /// Remaining columns of basis are left untouched.
        ///
        /// relTol &lt; 0 selects the auto tolerance max(m, n) * Consts.doubleZeroTreshold; a singular
        /// value S[j] &lt;= relTol * S[0] counts as zero. A is NOT modified. <paramref name="converged"/>
        /// is the SVD's convergence flag — when false the result is 0 and basis is untouched. Allocates
        /// SVD scratch (m x n + n x n + n) from A's arena.
        /// </summary>
        public static int nullspaceBasis(in doubleMxN A, ref doubleMxN basis, out bool converged,
                                         double relTol, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("nullspaceBasis: A must have m >= n (more rows than columns)");
            if (basis.M_Rows != n || basis.N_Cols != n)
                throw new ArgumentException("nullspaceBasis: basis must be n x n");
            if (maxIter < 1)
                throw new ArgumentException("nullspaceBasis: maxIter must be >= 1");

            converged = true;
            if (n == 0)
                return 0;

            var U = A.tempdoubleMat(m, n);
            var S = A.tempdoubleVec(n);
            var V = A.tempdoubleMat(n, n);

            converged = svdGolubKahan(in A, ref U, ref S, ref V, maxIter);
            if (!converged)
                return 0;

            if (relTol < (double)0)
                relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;
            double tol = relTol * S[0];

            // S is descending, so the negligible singular values are the trailing ones; compact their
            // V-columns to the front of basis.
            int dim = 0;
            for (int j = 0; j < n; j++)
            {
                if (S[j] <= tol)
                {
                    for (int i = 0; i < n; i++)
                        basis[i, dim] = V[i, j];
                    dim++;
                }
            }
            return dim;
        }

        /// <summary>nullspaceBasis with default maxIter (75).</summary>
        public static int nullspaceBasis(in doubleMxN A, ref doubleMxN basis, out bool converged, double relTol)
            => nullspaceBasis(in A, ref basis, out converged, relTol, 75);

        /// <summary>nullspaceBasis with auto tolerance (relTol = -1) and default maxIter (75).</summary>
        public static int nullspaceBasis(in doubleMxN A, ref doubleMxN basis, out bool converged)
            => nullspaceBasis(in A, ref basis, out converged, (double)(-1), 75);

        /// <summary>
        /// Orthonormal basis for the RANGE (column space) of A (m x n, m >= n): span of A's columns.
        /// From A = U diag(S) Vᵀ, the range is spanned by the left-singular vectors (columns of U) whose
        /// singular value exceeds the tolerance. Columns 0..rank-1 of <paramref name="basis"/> (m x n,
        /// caller-allocated) receive those vectors (orthonormal); the RETURN VALUE is rank. Remaining
        /// columns of basis are left untouched.
        ///
        /// Same tolerance / convergence / allocation semantics as <see cref="nullspaceBasis"/>.
        /// </summary>
        public static int rangeBasis(in doubleMxN A, ref doubleMxN basis, out bool converged,
                                     double relTol, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("rangeBasis: A must have m >= n (more rows than columns)");
            if (basis.M_Rows != m || basis.N_Cols != n)
                throw new ArgumentException("rangeBasis: basis must be m x n");
            if (maxIter < 1)
                throw new ArgumentException("rangeBasis: maxIter must be >= 1");

            converged = true;
            if (n == 0)
                return 0;

            var U = A.tempdoubleMat(m, n);
            var S = A.tempdoubleVec(n);
            var V = A.tempdoubleMat(n, n);

            converged = svdGolubKahan(in A, ref U, ref S, ref V, maxIter);
            if (!converged)
                return 0;

            if (relTol < (double)0)
                relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;
            double tol = relTol * S[0];

            // S is descending, so the significant singular values are the leading ones (columns
            // 0..rank-1 of U already in place).
            int rank = 0;
            for (int j = 0; j < n; j++)
            {
                if (S[j] > tol)
                {
                    for (int i = 0; i < m; i++)
                        basis[i, rank] = U[i, j];
                    rank++;
                }
            }
            return rank;
        }

        /// <summary>rangeBasis with default maxIter (75).</summary>
        public static int rangeBasis(in doubleMxN A, ref doubleMxN basis, out bool converged, double relTol)
            => rangeBasis(in A, ref basis, out converged, relTol, 75);

        /// <summary>rangeBasis with auto tolerance (relTol = -1) and default maxIter (75).</summary>
        public static int rangeBasis(in doubleMxN A, ref doubleMxN basis, out bool converged)
            => rangeBasis(in A, ref basis, out converged, (double)(-1), 75);
    }
}
