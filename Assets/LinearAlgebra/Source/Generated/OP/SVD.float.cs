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
        /// Allocates an n x m and an n x n Temp workspace (transposed layout so Burst can
        /// SIMD-vectorize the inner plane-rotation loops; same algorithm, unit-stride rows
        /// instead of strided columns).
        /// </summary>
        public static bool svdDecomposition(ref floatMxN U, ref floatN S, ref floatMxN V,
                                            int maxSweeps, float eps)
        {
            if (U.M_Rows < U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: U must have m >= n (more rows than columns)");

            if (S.N != U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: S.N must equal U.N_Cols");

            if (!V.IsSquare || V.M_Rows != U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: V must be square with side equal to U.N_Cols");

            if (maxSweeps < 1)
                throw new System.ArgumentException("svdDecomposition: maxSweeps must be >= 1");

            if (eps <= (float)0)
                throw new System.ArgumentException("svdDecomposition: eps must be > 0");

            int m = U.M_Rows;
            int n = U.N_Cols;

            if (n == 0)
                return true;

            // Transpose U into Ut (n x m) so that column j of U becomes ROW j of Ut.
            // Working on rows of Ut gives unit-stride contiguous access that Burst vectorizes.
            // Vt (n x n) accumulates V^T: row operations on Vt correspond to column operations
            // on V, so at the end we transpose Vt back into V.
            var Ut = new floatMxN(n, m, Allocator.Temp, true);  // will fill every element
            var Vt = new floatMxN(n, n, Allocator.Temp, false); // zeroed; diagonal set below

            // Fill Ut: row j of Ut = column j of U
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Ut[j, i] = U[i, j];

            // Initialize Vt to identity (Vt accumulates right singular vectors as rows)
            for (int i = 0; i < n; i++)
                Vt[i, i] = (float)1;

            // Step 2: One-sided Jacobi sweeps — all inner loops now hit contiguous rows
            bool converged = false;

            unsafe
            {
                float* utp = Ut.Data.Ptr;
                float* vtp = Vt.Data.Ptr;

                for (int sweep = 0; sweep < maxSweeps; sweep++) {

                    int rotations = 0;

                    for (int p = 0; p < n - 1; p++) {
                        for (int q = p + 1; q < n; q++) {

                            float* rowPU = utp + (long)p * m;
                            float* rowQU = utp + (long)q * m;

                            // Fused dot: alpha = ||row_p||^2, beta = ||row_q||^2, gamma = row_p·row_q
                            // All three are now unit-stride reads of length m — Burst vectorizes this.
                            float alpha = (float)0;
                            float beta  = (float)0;
                            float gamma = (float)0;

                            for (int i = 0; i < m; i++) {
                                float bip = rowPU[i];
                                float biq = rowQU[i];
                                alpha += bip * bip;
                                beta  += biq * biq;
                                gamma += bip * biq;
                            }

                            // Skip if either column is zero or pair is already orthogonal
                            if (alpha == (float)0 || beta == (float)0)
                                continue;

                            if (math.abs(gamma) <= eps * math.sqrt(alpha) * math.sqrt(beta))
                                continue;

                            // Rutishauser rotation (byte-for-byte unchanged from the original)
                            float zeta = (beta - alpha) / ((float)2 * gamma);
                            // sign(zeta) with 0 -> +1
                            float signZeta = zeta >= (float)0 ? (float)1 : (float)(-1);
                            float absZeta = math.abs(zeta);
                            float t;
                            if (absZeta > (float)1) {
                                // Factor out |zeta| to avoid zeta*zeta overflow; inv*inv <= 1
                                float inv = (float)1 / zeta;
                                t = signZeta / (absZeta * ((float)1 + math.sqrt((float)1 + inv * inv)));
                            } else {
                                // |zeta| <= 1 -> zeta*zeta <= 1, safe; zeta==0 -> t = signZeta
                                t = signZeta / (absZeta + math.sqrt((float)1 + zeta * zeta));
                            }
                            float c = (float)1 / math.sqrt((float)1 + t * t);
                            float s = c * t;

                            // Rotate rows p and q of Ut (length m) and of Vt (length n) — contiguous
                            // + [NoAlias] (p != q ⇒ distinct rows) so Burst vectorizes the butterfly.
                            // Row rotation on Vt = G^T * Vt is equivalent to V -> V * G (column
                            // rotation), so Vt^T = V throughout.
                            UnsafeOP.jacobiRotate(rowPU, rowQU, c, s, m);
                            UnsafeOP.jacobiRotate(vtp + (long)p * n, vtp + (long)q * n, c, s, n);

                            rotations++;
                        }
                    }

                    if (rotations == 0) {
                        converged = true;
                        break;
                    }
                }

                // Step 3: Extract singular values from row norms of Ut and normalize
                for (int j = 0; j < n; j++) {
                    float* rowJ = utp + (long)j * m;
                    float colNormSq = (float)0;
                    for (int i = 0; i < m; i++) {
                        float bij = rowJ[i];
                        colNormSq += bij * bij;
                    }

                    float sigma = math.sqrt(colNormSq);
                    S[j] = sigma;

                    if (sigma > (float)0) {
                        float invSigma = (float)1 / sigma;
                        for (int i = 0; i < m; i++)
                            rowJ[i] *= invSigma;
                    }
                    else {
                        // Explicit zero for rank-deficient rows
                        for (int i = 0; i < m; i++)
                            rowJ[i] = (float)0;
                        S[j] = (float)0;
                    }
                }
            }

            // Step 4: Selection sort descending by singular value
            for (int j = 0; j < n; j++) {
                int maxIdx = j;
                float maxVal = S[j];

                for (int k = j + 1; k < n; k++) {
                    if (S[k] > maxVal) {
                        maxIdx = k;
                        maxVal = S[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap singular values
                    float tmp = S[j];
                    S[j] = S[maxIdx];
                    S[maxIdx] = tmp;

                    // Swap rows of Ut and Vt (unit-stride — vectorizes; equivalent to
                    // swapping columns of U and V in the original layout)
                    SwapOP.Rows(ref Ut, j, maxIdx);
                    SwapOP.Rows(ref Vt, j, maxIdx);
                }
            }

            // Final: transpose Ut back into U (U[i,j] = Ut[j,i]) and Vt into V (V[i,j] = Vt[j,i])
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    U[i, j] = Ut[j, i];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = Vt[j, i];

            Ut.Dispose();
            Vt.Dispose();

            return converged;
        }

        /// <summary>svdDecomposition with default eps (Consts.floatZeroTreshold).</summary>
        public static bool svdDecomposition(ref floatMxN U, ref floatN S, ref floatMxN V,
                                            int maxSweeps)
            => svdDecomposition(ref U, ref S, ref V, maxSweeps, Consts.floatZeroTreshold);

        /// <summary>svdDecomposition with default maxSweeps (30) and eps (Consts.floatZeroTreshold).</summary>
        public static bool svdDecomposition(ref floatMxN U, ref floatN S, ref floatMxN V)
            => svdDecomposition(ref U, ref S, ref V, 30, Consts.floatZeroTreshold);

        /// <summary>
        /// Singular VALUES only of A (m x n, m >= n), via the symmetric eigenvalues of the augmented
        /// matrix H = [[0, A], [Aᵀ, 0]] (the Jordan–Wielandt form). H's eigenvalues are ±σ_i plus
        /// (m-n) zeros, so the n largest are exactly the singular values, descending. This routes the
        /// O(n^3) work through the fast Householder Eigen.eigenvaluesSymmetric, and unlike forming AᵀA
        /// it keeps the condition number κ(A) (not κ(A)²), so small singular values are not lost.
        ///
        /// A is NOT modified (it is copied into the augmented matrix). S (length n) receives the
        /// singular values, descending, clamped to be non-negative. Returns the convergence flag of
        /// the underlying QL iteration. Allocates an (m+n)² + (m+n) Temp workspace.
        /// </summary>
        public static bool svdValues(in floatMxN A, ref floatN S, int maxIter, float eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdValues: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("svdValues: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("svdValues: maxIter must be >= 1");
            if (eps <= (float)0)
                throw new ArgumentException("svdValues: eps must be > 0");

            if (n == 0)
                return true;

            int d = m + n;
            var H = new floatMxN(d, d, Allocator.Temp, false);
            var eig = new floatN(d, Allocator.Temp, false);

            // H = [[0, A], [Aᵀ, 0]]  (symmetric). Zero everything, then fill the two off-diagonal
            // blocks: H[i, m+j] = H[m+j, i] = A[i,j].
            unsafe
            {
                UnsafeUtility.MemClear(H.Data.Ptr, (long)d * d * UnsafeUtility.SizeOf<float>());
            }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float a = A[i, j];
                    H[i, m + j] = a;
                    H[m + j, i] = a;
                }

            bool ok = Eigen.eigenvaluesSymmetric(ref H, ref eig, maxIter, eps);

            // The n largest eigenvalues (descending) are σ_1 ≥ ... ≥ σ_n ≥ 0. Clamp tiny negatives
            // (rounding around a zero singular value) up to 0.
            if (ok)
                for (int i = 0; i < n; i++)
                    S[i] = math.max(eig[i], (float)0);

            eig.Dispose();
            H.Dispose();
            return ok;
        }

        /// <summary>svdValues with default maxIter (30) and eps (Consts.floatZeroTreshold).</summary>
        public static bool svdValues(in floatMxN A, ref floatN S)
            => svdValues(in A, ref S, 30, Consts.floatZeroTreshold);
    }
}
