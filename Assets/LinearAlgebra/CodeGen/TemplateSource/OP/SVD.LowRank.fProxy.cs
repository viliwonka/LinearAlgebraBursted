#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD {

        // Low-rank SVD: keep only the leading k singular triplets. svdTruncated returns the exact
        // top-k factors (best rank-k approximation, Eckart-Young); lowRankApprox returns the rank-k
        // approximation matrix directly. Both compute the full Golub-Kahan SVD and slice — exact but
        // not cheaper than the full SVD; the *approximate* fast path (random projection) is randomized
        // SVD. A is m x n with m >= n (the svdGolubKahan precondition).

        /// <summary>
        /// Truncated (thin) SVD: the k largest singular triplets of A (m x n, m >= n), so
        /// A ≈ Uk diag(Sk) Vkᵀ is the best rank-k approximation of A (Eckart-Young).
        /// Uk (m x k) and Vk (n x k) receive the leading k left/right singular vectors as columns
        /// (orthonormal); Sk (length k) the leading k singular values (descending, non-negative).
        /// All three are caller-allocated. 0 &lt;= k &lt;= n.
        ///
        /// Computes the full Golub-Kahan SVD and slices, so it is EXACT but no cheaper than a full SVD
        /// (use randomized SVD when k ≪ n and an approximation suffices). A is NOT modified.
        /// <paramref name="converged"/> is the SVD's flag (when false the outputs are undefined).
        /// Allocates full-SVD scratch (m x n + n x n + n) from A's arena.
        /// </summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, out bool converged, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdTruncated: A must have m >= n (more rows than columns)");
            if (k < 0 || k > n)
                throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            if (Uk.M_Rows != m || Uk.N_Cols != k)
                throw new ArgumentException("svdTruncated: Uk must be m x k");
            if (Sk.N != k)
                throw new ArgumentException("svdTruncated: Sk must have length k");
            if (Vk.M_Rows != n || Vk.N_Cols != k)
                throw new ArgumentException("svdTruncated: Vk must be n x k");
            if (maxIter < 1)
                throw new ArgumentException("svdTruncated: maxIter must be >= 1");

            converged = true;
            if (n == 0 || k == 0)
                return;

            var U = A.tempfProxyMat(m, n);
            var S = A.tempfProxyVec(n);
            var V = A.tempfProxyMat(n, n);

            converged = svdGolubKahan(in A, ref U, ref S, ref V, maxIter);
            if (!converged)
                return;

            for (int t = 0; t < k; t++)
            {
                Sk[t] = S[t];
                for (int i = 0; i < m; i++) Uk[i, t] = U[i, t];
                for (int i = 0; i < n; i++) Vk[i, t] = V[i, t];
            }
        }

        /// <summary>svdTruncated with default maxIter (75).</summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, out converged, 75);

        /// <summary>
        /// Best rank-k approximation of A (m x n, m >= n) written into Ak (m x n, caller-allocated):
        /// Ak = Σ_{t&lt;k} σ_t u_t v_tᵀ = Uk diag(Sk) Vkᵀ. This is the matrix that minimizes
        /// ||A - Ak|| over all rank-k matrices (Eckart-Young); the Frobenius error is sqrt(Σ_{i&gt;=k} σ_i²).
        /// Useful for compression / denoising. 0 &lt;= k &lt;= n. A is NOT modified.
        /// <paramref name="converged"/> is the SVD's flag (when false Ak is undefined). Allocates
        /// full-SVD scratch from A's arena.
        /// </summary>
        public static void lowRankApprox(in fProxyMxN A, ref fProxyMxN Ak, int k, out bool converged, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("lowRankApprox: A must have m >= n (more rows than columns)");
            if (k < 0 || k > n)
                throw new ArgumentException("lowRankApprox: k must be in [0, A.N_Cols]");
            if (Ak.M_Rows != m || Ak.N_Cols != n)
                throw new ArgumentException("lowRankApprox: Ak must be m x n");
            if (maxIter < 1)
                throw new ArgumentException("lowRankApprox: maxIter must be >= 1");

            converged = true;

            // Ak starts at zero (rank 0); also the correct result for k == 0 / empty A.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Ak[i, j] = (fProxy)0;

            if (n == 0 || k == 0)
                return;

            var U = A.tempfProxyMat(m, n);
            var S = A.tempfProxyVec(n);
            var V = A.tempfProxyMat(n, n);

            converged = svdGolubKahan(in A, ref U, ref S, ref V, maxIter);
            if (!converged)
                return;

            // Ak += σ_t · u_t v_tᵀ for t < k (rank-1 accumulation; inner row update is unit-stride
            // in the Ak row, V column gathered per t).
            for (int t = 0; t < k; t++)
            {
                fProxy s = S[t];
                for (int i = 0; i < m; i++)
                {
                    fProxy us = U[i, t] * s;
                    for (int j = 0; j < n; j++)
                        Ak[i, j] += us * V[j, t];
                }
            }
        }

        /// <summary>lowRankApprox with default maxIter (75).</summary>
        public static void lowRankApprox(in fProxyMxN A, ref fProxyMxN Ak, int k, out bool converged)
            => lowRankApprox(in A, ref Ak, k, out converged, 75);
    }
}
