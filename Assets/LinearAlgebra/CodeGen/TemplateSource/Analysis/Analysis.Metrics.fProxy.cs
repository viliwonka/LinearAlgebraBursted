using System;
using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Scalar matrix characterizations (trace / condition number / numerical rank / determinant).
    // These summarize a matrix as a single number, so they live on Analysis alongside the structural
    // predicates (isSymmetric / isOrthogonal / ...) -- NOT on Blas. The matrix-in overloads factor
    // internally into Temp (determinant/logDeterminant via LU, cond/rank via SVD) and leave A intact.
    public static partial class Analysis
    {
        /// <summary>
        /// Trace: the sum of the diagonal entries, Σ A[i,i]. A must be square.
        /// </summary>
        public static fProxy trace(in fProxyMxN A)
        {
            if (!A.IsSquare)
                throw new ArgumentException("trace: A must be square");

            fProxy sum = (fProxy)0;
            for (int i = 0; i < A.M_Rows; i++)
                sum += A[i, i];
            return sum;
        }

        /// <summary>
        /// 2-norm condition number κ₂(A) = σ_max / σ_min (any shape, via SVD). Returns positive
        /// infinity when A is singular / rank-deficient (σ_min == 0). Allocates SVD scratch;
        /// A is not modified. κ₂ ≈ 1 means well-conditioned; large κ₂ means ill-conditioned.
        /// </summary>
        public static fProxy cond(in fProxyMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (fProxy)0;

            fProxyN S = A.fProxyTempVec(k);
            SVD.singularValues(in A, ref S);

            fProxy sMin = S[k - 1];          // singular values are descending
            if (!(sMin > (fProxy)0))         // NaN-safe: singular -> infinite condition number
                return fProxy.PositiveInfinity;

            return S[0] / sMin;
        }

        /// <summary>
        /// Numerical rank: the number of singular values greater than relTol * σ_max (any shape,
        /// via SVD). relTol &lt; 0 selects the automatic tolerance max(m, n) * Consts.fProxyZeroThreshold
        /// (matching pinvSolve). Allocates SVD scratch; A is not modified.
        /// </summary>
        public static int rank(in fProxyMxN A, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            if (k == 0)
                return 0;

            fProxyN S = A.fProxyTempVec(k);
            SVD.singularValues(in A, ref S);

            if (S[0] == (fProxy)0)
                return 0;

            if (relTol < (fProxy)0)
                relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

            fProxy tol = relTol * S[0];
            int r = 0;
            for (int i = 0; i < k; i++)
                if (S[i] > tol)
                    r++;
            return r;
        }

        /// <summary>Numerical rank with the automatic tolerance (relTol &lt; 0).</summary>
        public static int rank(in fProxyMxN A) => rank(in A, (fProxy)(-1));

        /// <summary>
        /// Determinant det(A) via partial-pivoting LU (the standard O(n³) method): sign · Π U[i,i].
        /// A must be square. Allocates LU scratch (an n×n copy + a pivot) in Temp; A is not modified.
        /// A singular A returns exactly 0. NOTE: this is a product of n numbers and therefore
        /// over/underflows the float/double range for even moderate n (e.g. det ≈ 2¹⁰²⁴ → Inf) --
        /// for anything but small matrices prefer <see cref="logDeterminant"/>.
        /// </summary>
        public static fProxy determinant(in fProxyMxN A)
        {
            if (!A.IsSquare)
                throw new ArgumentException("determinant: A must be square");

            int n = A.M_Rows;
            if (n == 0)
                return (fProxy)1;               // determinant of the empty matrix is 1 (empty product)

            fProxyMxN lu = A.TempCopy();         // LU is destructive; factor a copy, leave A intact
            var P = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref lu, ref P);
            return determinant(in lu, in P);
        }

        /// <summary>
        /// Determinant read directly off a compact in-place LU factor and its pivot (zero-alloc) --
        /// sign · Π LU[P[i], i]. Free after any LU factorization/solve you already ran; use the
        /// matrix-in overload if you only have A. Throws if LU is not square or P.N != LU.M_Rows.
        /// </summary>
        public static fProxy determinant(in fProxyMxN LU, in Pivot P)
        {
            if (!LU.IsSquare)
                throw new ArgumentException("determinant: LU must be square");

            if (P.N != LU.M_Rows)
                throw new ArgumentException("determinant: P.N must equal LU.M_Rows");

            int m = LU.M_Rows;
            fProxy det = P.Sign;

            for (int i = 0; i < m; i++)
                det *= LU[P[i], i];

            return det;
        }

        /// <summary>
        /// Returns log|det(A)| with the sign of det(A) in <paramref name="sign"/> -- use this over
        /// <see cref="determinant(in fProxyMxN)"/> for anything but small matrices, since the raw
        /// product over/underflows where the log-sum stays finite. A must be square; allocates LU
        /// scratch in Temp, A is not modified. A singular A returns (sign 0, negative infinity).
        /// </summary>
        /// <param name="sign">On exit: +1, −1, or 0 (singular) -- the sign of det(A).</param>
        public static fProxy logDeterminant(in fProxyMxN A, out fProxy sign)
        {
            if (!A.IsSquare)
                throw new ArgumentException("logDeterminant: A must be square");

            int n = A.M_Rows;
            if (n == 0) { sign = (fProxy)1; return (fProxy)0; }   // empty matrix: det 1, log 0

            fProxyMxN lu = A.TempCopy();
            var P = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref lu, ref P);

            fProxy s = P.Sign;
            fProxy logAbs = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy d = lu[P[i], i];
                if (d < (fProxy)0) s = -s;
                else if (d == (fProxy)0) s = (fProxy)0;          // singular -> det 0
                logAbs += math.log(math.abs(d));                 // |d| == 0 -> log = -infinity
            }

            sign = s;
            return logAbs;
        }
    }
}
