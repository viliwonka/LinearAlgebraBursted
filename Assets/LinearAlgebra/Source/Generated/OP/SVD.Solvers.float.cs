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
        /// A is NOT modified (the Golub-Kahan path takes it as input). b is not modified. x (length
        /// N_Cols) is overwritten.
        /// relTol &lt; 0 selects auto tolerance: relTol = max(m, n) * Consts.floatZeroTreshold.
        /// Singular values S[j] &lt;= relTol * S[0] are treated as zero.
        /// Allocates temporaries from A's arena via tempfloatVec/tempfloatMat (not an Inpl op).
        /// Returns the numerical rank used; converged is svdGolubKahan's return value.
        /// </summary>
        // Caller-provided scratch overload (zero-alloc). Let k = min(A.M_Rows, A.N_Cols):
        //   S  - singular values, length k
        //   M  - singular-vector matrix, k x k (plays the role of V for tall A, W for wide A)
        //   U  - left-factor scratch, max(m,n) x k (receives the Golub-Kahan U of the decomposed
        //        orientation: A for tall, A^T for wide)
        //   At - A^T scratch, A.N_Cols x A.M_Rows; USED ONLY when A is wide (m < n). For m >= n
        //        pass default(floatMxN) (it is never read). Filled in-place via the ref-dest trans.
        // Hoist these out of a hot loop solving many same-shape systems to avoid per-call allocs.
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged,
                                    float relTol, int maxSweeps,
                                    ref floatN S, ref floatMxN M, ref floatMxN U, ref floatMxN At)
        {
            if (b.N != A.M_Rows)
                throw new ArgumentException("pinvSolve: b.N must equal A.M_Rows");

            if (x.N != A.N_Cols)
                throw new ArgumentException("pinvSolve: x.N must equal A.N_Cols");

            if (maxSweeps < 1)
                throw new ArgumentException("pinvSolve: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            if (S.N != k)
                throw new ArgumentException("pinvSolve: S scratch length must equal min(A.M_Rows, A.N_Cols)");

            if (M.M_Rows != k || M.N_Cols != k)
                throw new ArgumentException("pinvSolve: M scratch must be k x k, k = min(A.M_Rows, A.N_Cols)");

            if (U.M_Rows != big || U.N_Cols != k)
                throw new ArgumentException("pinvSolve: U scratch must be max(m,n) x min(m,n)");

            if (m >= n) {
                // Tall or square case: A = U * diag(S) * V^T; U receives the left factor, M = V.
                converged = svdGolubKahan(in A, ref U, ref S, ref M, maxSweeps);

                // Auto tolerance
                if (relTol < (float)0)
                    relTol = (float)math.max(m, n) * Consts.floatZeroTreshold;

                // Zero x
                for (int kk = 0; kk < n; kk++)
                    x[kk] = (float)0;

                if (n == 0 || S[0] == (float)0)
                    return 0;

                float tol = relTol * S[0];
                int rank = 0;

                // x = V * diag(1/S_j) * U^T * b  (only for S[j] > tol)
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    // coeff = (U[:,j]^T * b) / S[j], U[:,j] is column j of the left factor
                    float dot = (float)0;
                    for (int i = 0; i < m; i++)
                        dot += U[i, j] * b[i];

                    float coeff = dot / S[j];

                    for (int kk = 0; kk < n; kk++)
                        x[kk] += coeff * M[kk, j];

                    rank++;
                }

                return rank;
            }
            else {
                // Wide case: decompose A^T (n x m, tall). Right singular vectors of A are columns of
                // U (the left factor of A^T); left singular vectors of A are columns of M (= W).
                if (At.M_Rows != n || At.N_Cols != m)
                    throw new ArgumentException("pinvSolve: At scratch must be A.N_Cols x A.M_Rows for the wide (m < n) case");

                floatOP.trans(in A, ref At);   // At = A^T (zero-alloc, ref-dest trans)

                converged = svdGolubKahan(in At, ref U, ref S, ref M, maxSweeps);

                // Auto tolerance
                if (relTol < (float)0)
                    relTol = (float)math.max(m, n) * Consts.floatZeroTreshold;

                // Zero x
                for (int kk = 0; kk < n; kk++)
                    x[kk] = (float)0;

                if (m == 0 || S[0] == (float)0)
                    return 0;

                float tol = relTol * S[0];
                int rank = 0;

                // x = U * diag(1/S_j) * W^T * b  (only for S[j] > tol)
                // U columns (length n) are right singular vectors of A
                // M (= W) columns (length m) are left singular vectors of A
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    // coeff = (W[:,j]^T * b) / S[j]
                    float dot = (float)0;
                    for (int i = 0; i < m; i++)
                        dot += M[i, j] * b[i];

                    float coeff = dot / S[j];

                    for (int kk = 0; kk < n; kk++)
                        x[kk] += coeff * U[kk, j];

                    rank++;
                }

                return rank;
            }
        }

        /// <summary>
        /// pinvSolve allocating wrapper: allocates the SVD scratch (S, k x k singular-vector matrix,
        /// and A^T for the wide case) from A's arena and delegates to the zero-alloc primitive.
        /// </summary>
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged,
                                    float relTol, int maxSweeps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            floatN S = A.tempfloatVec(k);
            floatMxN M = A.tempfloatMat(k, k);
            floatMxN U = A.tempfloatMat(big, k);
            floatMxN At = default;
            if (m < n)
                At = A.tempfloatMat(n, m);

            return pinvSolve(ref A, in b, ref x, out converged, relTol, maxSweeps, ref S, ref M, ref U, ref At);
        }

        /// <summary>
        /// pinvSolve using a reusable workspace (Arena.floatSvdWorkspace(m, n)) — zero-alloc.
        /// The workspace must be sized for A's shape (k = min(A.M_Rows, A.N_Cols)); the guards in
        /// the underlying scratch primitive enforce this.
        /// </summary>
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged,
                                    ref floatSvdWorkspace ws, float relTol, int maxSweeps)
            => pinvSolve(ref A, in b, ref x, out converged, relTol, maxSweeps, ref ws.S, ref ws.M, ref ws.U, ref ws.At);

        /// <summary>pinvSolve (workspace) with default maxSweeps (30).</summary>
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged,
                                    ref floatSvdWorkspace ws, float relTol)
            => pinvSolve(ref A, in b, ref x, out converged, ref ws, relTol, 30);

        /// <summary>pinvSolve (workspace) with default relTol (-1, auto) and maxSweeps (30).</summary>
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged,
                                    ref floatSvdWorkspace ws)
            => pinvSolve(ref A, in b, ref x, out converged, ref ws, (float)(-1), 30);

        /// <summary>pinvSolve with default maxSweeps (30).</summary>
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged,
                                    float relTol)
            => pinvSolve(ref A, in b, ref x, out converged, relTol, 30);

        /// <summary>pinvSolve with default relTol (-1, auto tolerance) and maxSweeps (30).</summary>
        public static int pinvSolve(ref floatMxN A, in floatN b, ref floatN x, out bool converged)
            => pinvSolve(ref A, in b, ref x, out converged, (float)(-1), 30);

        /// <summary>
        /// Moore-Penrose pseudo-inverse: Aplus (N_Cols x M_Rows, caller-allocated) = V diag(1/S_i, S_i > tol) U^T.
        /// A is NOT modified (the Golub-Kahan path takes it as input). Same tolerance/rank/return
        /// semantics as pinvSolve. Any shape.
        /// </summary>
        // Caller-provided scratch overload (zero-alloc); same scratch contract as pinvSolve:
        // k = min(A.M_Rows, A.N_Cols); S length k; M is k x k (V for tall A, W for wide A);
        // U is max(m,n) x k (the Golub-Kahan left factor of the decomposed orientation);
        // At (A.N_Cols x A.M_Rows) used only when A is wide (m < n), else pass default(floatMxN).
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged,
                                        float relTol, int maxSweeps,
                                        ref floatN S, ref floatMxN M, ref floatMxN U, ref floatMxN At)
        {
            if (Aplus.M_Rows != A.N_Cols)
                throw new ArgumentException("pseudoInverse: Aplus.M_Rows must equal A.N_Cols");

            if (Aplus.N_Cols != A.M_Rows)
                throw new ArgumentException("pseudoInverse: Aplus.N_Cols must equal A.M_Rows");

            if (maxSweeps < 1)
                throw new ArgumentException("pseudoInverse: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            if (S.N != k)
                throw new ArgumentException("pseudoInverse: S scratch length must equal min(A.M_Rows, A.N_Cols)");

            if (M.M_Rows != k || M.N_Cols != k)
                throw new ArgumentException("pseudoInverse: M scratch must be k x k, k = min(A.M_Rows, A.N_Cols)");

            if (U.M_Rows != big || U.N_Cols != k)
                throw new ArgumentException("pseudoInverse: U scratch must be max(m,n) x min(m,n)");

            // Zero-initialize Aplus
            for (int r = 0; r < Aplus.M_Rows; r++)
                for (int c = 0; c < Aplus.N_Cols; c++)
                    Aplus[r, c] = (float)0;

            if (m >= n) {
                // A = U * diag(S) * V^T; U receives the left factor, M = V
                converged = svdGolubKahan(in A, ref U, ref S, ref M, maxSweeps);

                if (relTol < (float)0)
                    relTol = (float)math.max(m, n) * Consts.floatZeroTreshold;

                if (n == 0 || S[0] == (float)0)
                    return 0;

                float tol = relTol * S[0];
                int rank = 0;

                // Aplus[r, c] = sum_{j: S[j]>tol} V[r,j] * (1/S[j]) * U[c,j]
                // r in 0..n-1, c in 0..m-1
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    float invS = (float)1 / S[j];

                    for (int r = 0; r < n; r++) {
                        float vr = M[r, j] * invS;
                        for (int c = 0; c < m; c++)
                            Aplus[r, c] += vr * U[c, j];
                    }

                    rank++;
                }

                return rank;
            }
            else {
                // Wide case: decompose A^T (n x m); U receives its left factor, M = W
                if (At.M_Rows != n || At.N_Cols != m)
                    throw new ArgumentException("pseudoInverse: At scratch must be A.N_Cols x A.M_Rows for the wide (m < n) case");

                floatOP.trans(in A, ref At);   // At = A^T (zero-alloc, ref-dest trans)

                converged = svdGolubKahan(in At, ref U, ref S, ref M, maxSweeps);

                if (relTol < (float)0)
                    relTol = (float)math.max(m, n) * Consts.floatZeroTreshold;

                if (m == 0 || S[0] == (float)0)
                    return 0;

                float tol = relTol * S[0];
                int rank = 0;

                // Aplus[r, c] = sum_{j: S[j]>tol} U[r,j] * (1/S[j]) * W[c,j]
                // r in 0..n-1, c in 0..m-1  (U holds the right singular vectors of A)
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    float invS = (float)1 / S[j];

                    for (int r = 0; r < n; r++) {
                        float atr = U[r, j] * invS;
                        for (int c = 0; c < m; c++)
                            Aplus[r, c] += atr * M[c, j];
                    }

                    rank++;
                }

                return rank;
            }
        }

        /// <summary>
        /// pseudoInverse allocating wrapper: allocates the SVD scratch (S, k x k singular-vector
        /// matrix, and A^T for the wide case) from A's arena and delegates to the zero-alloc primitive.
        /// </summary>
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged,
                                        float relTol, int maxSweeps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            floatN S = A.tempfloatVec(k);
            floatMxN M = A.tempfloatMat(k, k);
            floatMxN U = A.tempfloatMat(big, k);
            floatMxN At = default;
            if (m < n)
                At = A.tempfloatMat(n, m);

            return pseudoInverse(ref A, ref Aplus, out converged, relTol, maxSweeps, ref S, ref M, ref U, ref At);
        }

        /// <summary>
        /// pseudoInverse using a reusable workspace (Arena.floatSvdWorkspace(m, n)) — zero-alloc.
        /// The workspace must be sized for A's shape (k = min(A.M_Rows, A.N_Cols)).
        /// </summary>
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged,
                                        ref floatSvdWorkspace ws, float relTol, int maxSweeps)
            => pseudoInverse(ref A, ref Aplus, out converged, relTol, maxSweeps, ref ws.S, ref ws.M, ref ws.U, ref ws.At);

        /// <summary>pseudoInverse (workspace) with default maxSweeps (30).</summary>
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged,
                                        ref floatSvdWorkspace ws, float relTol)
            => pseudoInverse(ref A, ref Aplus, out converged, ref ws, relTol, 30);

        /// <summary>pseudoInverse (workspace) with default relTol (-1, auto) and maxSweeps (30).</summary>
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged,
                                        ref floatSvdWorkspace ws)
            => pseudoInverse(ref A, ref Aplus, out converged, ref ws, (float)(-1), 30);

        /// <summary>pseudoInverse with default maxSweeps (30).</summary>
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged,
                                        float relTol)
            => pseudoInverse(ref A, ref Aplus, out converged, relTol, 30);

        /// <summary>pseudoInverse with default relTol (-1, auto tolerance) and maxSweeps (30).</summary>
        public static int pseudoInverse(ref floatMxN A, ref floatMxN Aplus, out bool converged)
            => pseudoInverse(ref A, ref Aplus, out converged, (float)(-1), 30);
    }
}
