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
        /// One-sided Jacobi SVD: A = U * diag(S) * V^T (thin/economy).
        /// On input U holds A (m x n, m >= n); on output U's columns are the left singular
        /// vectors (m x n), S the singular values (length n, non-negative, descending),
        /// V the right singular vectors (n x n, NOT transposed). V is overwritten
        /// (initialized to identity internally); S is overwritten.
        /// Columns of U matching zero singular values are left as zero vectors.
        /// Returns true if converged within maxSweeps; false otherwise (outputs are
        /// still normalized and sorted, and contain no NaN/Inf for finite input of
        /// moderate magnitude (column 2-norms within the type's representable range);
        /// extreme magnitudes that overflow when squared are not rescaled in this version).
        /// For m &lt; n: decompose trans(A) and swap the roles of U and V.
        /// Does not allocate.
        /// </summary>
        public static bool svdDecomposition(ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                            int maxSweeps, double eps)
        {
            if (U.M_Rows < U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: U must have m >= n (more rows than columns)");

            if (S.N != U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: S.N must equal U.N_Cols");

            if (!V.IsSquare || V.M_Rows != U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: V must be square with side equal to U.N_Cols");

            if (maxSweeps < 1)
                throw new System.ArgumentException("svdDecomposition: maxSweeps must be >= 1");

            if (eps <= (double)0)
                throw new System.ArgumentException("svdDecomposition: eps must be > 0");

            int m = U.M_Rows;
            int n = U.N_Cols;

            if (n == 0)
                return true;

            // Step 1: Set V to identity
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = (i == j) ? (double)1 : (double)0;

            // Step 2: One-sided Jacobi sweeps
            bool converged = false;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {

                int rotations = 0;

                for (int p = 0; p < n - 1; p++) {
                    for (int q = p + 1; q < n; q++) {

                        // Fused loop: compute alpha, beta, gamma
                        double alpha = (double)0;
                        double beta  = (double)0;
                        double gamma = (double)0;

                        for (int i = 0; i < m; i++) {
                            double bip = U[i, p];
                            double biq = U[i, q];
                            alpha += bip * bip;
                            beta  += biq * biq;
                            gamma += bip * biq;
                        }

                        // Skip if either column is zero or pair is already orthogonal
                        if (alpha == (double)0 || beta == (double)0)
                            continue;

                        if (math.abs(gamma) <= eps * math.sqrt(alpha) * math.sqrt(beta))
                            continue;

                        // Rutishauser rotation
                        double zeta = (beta - alpha) / ((double)2 * gamma);
                        // sign(zeta) with 0 -> +1
                        double signZeta = zeta >= (double)0 ? (double)1 : (double)(-1);
                        double absZeta = math.abs(zeta);
                        double t;
                        if (absZeta > (double)1) {
                            // Factor out |zeta| to avoid zeta*zeta overflow; inv*inv <= 1
                            double inv = (double)1 / zeta;
                            t = signZeta / (absZeta * ((double)1 + math.sqrt((double)1 + inv * inv)));
                        } else {
                            // |zeta| <= 1 -> zeta*zeta <= 1, safe; zeta==0 -> t = signZeta
                            t = signZeta / (absZeta + math.sqrt((double)1 + zeta * zeta));
                        }
                        double c = (double)1 / math.sqrt((double)1 + t * t);
                        double s = c * t;

                        // Rotate columns p and q of U (B)
                        for (int i = 0; i < m; i++) {
                            double bip = U[i, p];
                            double biq = U[i, q];
                            U[i, p] = c * bip - s * biq;
                            U[i, q] = s * bip + c * biq;
                        }

                        // Rotate columns p and q of V
                        for (int i = 0; i < n; i++) {
                            double vip = V[i, p];
                            double viq = V[i, q];
                            V[i, p] = c * vip - s * viq;
                            V[i, q] = s * vip + c * viq;
                        }

                        rotations++;
                    }
                }

                if (rotations == 0) {
                    converged = true;
                    break;
                }
            }

            // Step 3: Extract singular values and normalize columns of U
            for (int j = 0; j < n; j++) {
                double colNormSq = (double)0;
                for (int i = 0; i < m; i++) {
                    double bij = U[i, j];
                    colNormSq += bij * bij;
                }

                double sigma = math.sqrt(colNormSq);
                S[j] = sigma;

                if (sigma > (double)0) {
                    double invSigma = (double)1 / sigma;
                    for (int i = 0; i < m; i++)
                        U[i, j] = U[i, j] * invSigma;
                }
                else {
                    // Explicit zero for rank-deficient columns
                    for (int i = 0; i < m; i++)
                        U[i, j] = (double)0;
                    S[j] = (double)0;
                }
            }

            // Step 4: Selection sort descending by singular value
            for (int j = 0; j < n; j++) {
                int maxIdx = j;
                double maxVal = S[j];

                for (int k = j + 1; k < n; k++) {
                    if (S[k] > maxVal) {
                        maxIdx = k;
                        maxVal = S[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap singular values
                    double tmp = S[j];
                    S[j] = S[maxIdx];
                    S[maxIdx] = tmp;

                    // Swap columns of U and V
                    SwapOP.Columns(ref U, j, maxIdx);
                    SwapOP.Columns(ref V, j, maxIdx);
                }
            }

            return converged;
        }

        /// <summary>svdDecomposition with default eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdDecomposition(ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                            int maxSweeps)
            => svdDecomposition(ref U, ref S, ref V, maxSweeps, Consts.doubleZeroTreshold);

        /// <summary>svdDecomposition with default maxSweeps (30) and eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdDecomposition(ref doubleMxN U, ref doubleN S, ref doubleMxN V)
            => svdDecomposition(ref U, ref S, ref V, 30, Consts.doubleZeroTreshold);
    }
}
