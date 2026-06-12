#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Inpl = inplace
    /// </summary>
    public static partial class SVD {

        /// <summary>
        /// Minimum-norm least-squares solve: x = argmin ||A x - b||2 (minimum ||x|| among minimizers).
        /// Works for any shape (m >= n and m &lt; n) and any rank, including rank 0.
        /// A is DESTROYED (used as SVD workspace). b is not modified. x (length N_Cols) is overwritten.
        /// relTol &lt; 0 selects auto tolerance: relTol = max(m, n) * Consts.doubleZeroTreshold.
        /// Singular values S[j] &lt;= relTol * S[0] are treated as zero.
        /// Allocates temporaries from A's arena via tempdoubleVec/tempdoubleMat (not an Inpl op).
        /// Returns the numerical rank used; converged is svdDecomposition's return value.
        /// </summary>
        public static int pinvSolve(ref doubleMxN A, in doubleN b, ref doubleN x, out bool converged,
                                    double relTol, int maxSweeps)
        {
            if (b.N != A.M_Rows)
                throw new ArgumentException("pinvSolve: b.N must equal A.M_Rows");

            if (x.N != A.N_Cols)
                throw new ArgumentException("pinvSolve: x.N must equal A.N_Cols");

            if (maxSweeps < 1)
                throw new ArgumentException("pinvSolve: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m >= n) {
                // Tall or square case: A = U * diag(S) * V^T, U stored in A after decomposition
                doubleN S = A.tempdoubleVec(n);
                doubleMxN V = A.tempdoubleMat(n, n);

                converged = svdDecomposition(ref A, ref S, ref V, maxSweeps);

                // Auto tolerance
                if (relTol < (double)0)
                    relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;

                // Zero x
                for (int k = 0; k < n; k++)
                    x[k] = (double)0;

                if (n == 0 || S[0] == (double)0)
                    return 0;

                double tol = relTol * S[0];
                int rank = 0;

                // x = V * diag(1/S_j) * U^T * b  (only for S[j] > tol)
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    // coeff = (U[:,j]^T * b) / S[j], U[:,j] is column j of A (which now holds U)
                    double dot = (double)0;
                    for (int i = 0; i < m; i++)
                        dot += A[i, j] * b[i];

                    double coeff = dot / S[j];

                    for (int k = 0; k < n; k++)
                        x[k] += coeff * V[k, j];

                    rank++;
                }

                return rank;
            }
            else {
                // Wide case: decompose A^T (n x m, tall). Right singular vectors of A
                // are columns of A^T after decomposition; left singular vectors of A are columns of W.
                doubleMxN At = doubleOP.trans(A);
                doubleN S = A.tempdoubleVec(m);
                doubleMxN W = A.tempdoubleMat(m, m);

                converged = svdDecomposition(ref At, ref S, ref W, maxSweeps);

                // Auto tolerance
                if (relTol < (double)0)
                    relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;

                // Zero x
                for (int k = 0; k < n; k++)
                    x[k] = (double)0;

                if (m == 0 || S[0] == (double)0)
                    return 0;

                double tol = relTol * S[0];
                int rank = 0;

                // x = At * diag(1/S_j) * W^T * b  (only for S[j] > tol)
                // At columns (length n) are right singular vectors of A
                // W columns (length m) are left singular vectors of A
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    // coeff = (W[:,j]^T * b) / S[j]
                    double dot = (double)0;
                    for (int i = 0; i < m; i++)
                        dot += W[i, j] * b[i];

                    double coeff = dot / S[j];

                    for (int k = 0; k < n; k++)
                        x[k] += coeff * At[k, j];

                    rank++;
                }

                return rank;
            }
        }

        /// <summary>pinvSolve with default maxSweeps (30).</summary>
        public static int pinvSolve(ref doubleMxN A, in doubleN b, ref doubleN x, out bool converged,
                                    double relTol)
            => pinvSolve(ref A, in b, ref x, out converged, relTol, 30);

        /// <summary>pinvSolve with default relTol (-1, auto tolerance) and maxSweeps (30).</summary>
        public static int pinvSolve(ref doubleMxN A, in doubleN b, ref doubleN x, out bool converged)
            => pinvSolve(ref A, in b, ref x, out converged, (double)(-1), 30);

        /// <summary>
        /// Moore-Penrose pseudo-inverse: Aplus (N_Cols x M_Rows, caller-allocated) = V diag(1/S_i, S_i > tol) U^T.
        /// A is DESTROYED. Same tolerance/rank/return semantics as pinvSolve. Any shape.
        /// </summary>
        public static int pseudoInverse(ref doubleMxN A, ref doubleMxN Aplus, out bool converged,
                                        double relTol, int maxSweeps)
        {
            if (Aplus.M_Rows != A.N_Cols)
                throw new ArgumentException("pseudoInverse: Aplus.M_Rows must equal A.N_Cols");

            if (Aplus.N_Cols != A.M_Rows)
                throw new ArgumentException("pseudoInverse: Aplus.N_Cols must equal A.M_Rows");

            if (maxSweeps < 1)
                throw new ArgumentException("pseudoInverse: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;

            // Zero-initialize Aplus
            for (int r = 0; r < Aplus.M_Rows; r++)
                for (int c = 0; c < Aplus.N_Cols; c++)
                    Aplus[r, c] = (double)0;

            if (m >= n) {
                // A = U * diag(S) * V^T, A now holds U after decomposition
                doubleN S = A.tempdoubleVec(n);
                doubleMxN V = A.tempdoubleMat(n, n);

                converged = svdDecomposition(ref A, ref S, ref V, maxSweeps);

                if (relTol < (double)0)
                    relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;

                if (n == 0 || S[0] == (double)0)
                    return 0;

                double tol = relTol * S[0];
                int rank = 0;

                // Aplus[r, c] = sum_{j: S[j]>tol} V[r,j] * (1/S[j]) * U[c,j]
                // r in 0..n-1, c in 0..m-1
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    double invS = (double)1 / S[j];

                    for (int r = 0; r < n; r++) {
                        double vr = V[r, j] * invS;
                        for (int c = 0; c < m; c++)
                            Aplus[r, c] += vr * A[c, j];
                    }

                    rank++;
                }

                return rank;
            }
            else {
                // Wide case: decompose A^T (n x m)
                doubleMxN At = doubleOP.trans(A);
                doubleN S = A.tempdoubleVec(m);
                doubleMxN W = A.tempdoubleMat(m, m);

                converged = svdDecomposition(ref At, ref S, ref W, maxSweeps);

                if (relTol < (double)0)
                    relTol = (double)math.max(m, n) * Consts.doubleZeroTreshold;

                if (m == 0 || S[0] == (double)0)
                    return 0;

                double tol = relTol * S[0];
                int rank = 0;

                // Aplus[r, c] = sum_{j: S[j]>tol} At[r,j] * (1/S[j]) * W[c,j]
                // r in 0..n-1, c in 0..m-1
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    double invS = (double)1 / S[j];

                    for (int r = 0; r < n; r++) {
                        double atr = At[r, j] * invS;
                        for (int c = 0; c < m; c++)
                            Aplus[r, c] += atr * W[c, j];
                    }

                    rank++;
                }

                return rank;
            }
        }

        /// <summary>pseudoInverse with default maxSweeps (30).</summary>
        public static int pseudoInverse(ref doubleMxN A, ref doubleMxN Aplus, out bool converged,
                                        double relTol)
            => pseudoInverse(ref A, ref Aplus, out converged, relTol, 30);

        /// <summary>pseudoInverse with default relTol (-1, auto tolerance) and maxSweeps (30).</summary>
        public static int pseudoInverse(ref doubleMxN A, ref doubleMxN Aplus, out bool converged)
            => pseudoInverse(ref A, ref Aplus, out converged, (double)(-1), 30);
    }
}
