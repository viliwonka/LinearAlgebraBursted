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
        public static bool svdDecomposition(ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V,
                                            int maxSweeps, fProxy eps)
        {
            if (U.M_Rows < U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: U must have m >= n (more rows than columns)");

            if (S.N != U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: S.N must equal U.N_Cols");

            if (!V.IsSquare || V.M_Rows != U.N_Cols)
                throw new System.ArgumentException("svdDecomposition: V must be square with side equal to U.N_Cols");

            if (maxSweeps < 1)
                throw new System.ArgumentException("svdDecomposition: maxSweeps must be >= 1");

            if (eps <= (fProxy)0)
                throw new System.ArgumentException("svdDecomposition: eps must be > 0");

            int m = U.M_Rows;
            int n = U.N_Cols;

            if (n == 0)
                return true;

            // Transpose U into Ut (n x m) so that column j of U becomes ROW j of Ut.
            // Working on rows of Ut gives unit-stride contiguous access that Burst vectorizes.
            // Vt (n x n) accumulates V^T: row operations on Vt correspond to column operations
            // on V, so at the end we transpose Vt back into V.
            var Ut = new fProxyMxN(n, m, Allocator.Temp, true);  // will fill every element
            var Vt = new fProxyMxN(n, n, Allocator.Temp, false); // zeroed; diagonal set below

            // Fill Ut: row j of Ut = column j of U
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Ut[j, i] = U[i, j];

            // Initialize Vt to identity (Vt accumulates right singular vectors as rows)
            for (int i = 0; i < n; i++)
                Vt[i, i] = (fProxy)1;

            // Step 2: One-sided Jacobi sweeps — all inner loops now hit contiguous rows
            bool converged = false;

            unsafe
            {
                fProxy* utp = Ut.Data.Ptr;
                fProxy* vtp = Vt.Data.Ptr;

                for (int sweep = 0; sweep < maxSweeps; sweep++) {

                    int rotations = 0;

                    for (int p = 0; p < n - 1; p++) {
                        for (int q = p + 1; q < n; q++) {

                            fProxy* rowPU = utp + (long)p * m;
                            fProxy* rowQU = utp + (long)q * m;

                            // Fused dot: alpha = ||row_p||^2, beta = ||row_q||^2, gamma = row_p·row_q
                            // All three are now unit-stride reads of length m — Burst vectorizes this.
                            fProxy alpha = (fProxy)0;
                            fProxy beta  = (fProxy)0;
                            fProxy gamma = (fProxy)0;

                            for (int i = 0; i < m; i++) {
                                fProxy bip = rowPU[i];
                                fProxy biq = rowQU[i];
                                alpha += bip * bip;
                                beta  += biq * biq;
                                gamma += bip * biq;
                            }

                            // Skip if either column is zero or pair is already orthogonal
                            if (alpha == (fProxy)0 || beta == (fProxy)0)
                                continue;

                            if (math.abs(gamma) <= eps * math.sqrt(alpha) * math.sqrt(beta))
                                continue;

                            // Rutishauser rotation (byte-for-byte unchanged from the original)
                            fProxy zeta = (beta - alpha) / ((fProxy)2 * gamma);
                            // sign(zeta) with 0 -> +1
                            fProxy signZeta = zeta >= (fProxy)0 ? (fProxy)1 : (fProxy)(-1);
                            fProxy absZeta = math.abs(zeta);
                            fProxy t;
                            if (absZeta > (fProxy)1) {
                                // Factor out |zeta| to avoid zeta*zeta overflow; inv*inv <= 1
                                fProxy inv = (fProxy)1 / zeta;
                                t = signZeta / (absZeta * ((fProxy)1 + math.sqrt((fProxy)1 + inv * inv)));
                            } else {
                                // |zeta| <= 1 -> zeta*zeta <= 1, safe; zeta==0 -> t = signZeta
                                t = signZeta / (absZeta + math.sqrt((fProxy)1 + zeta * zeta));
                            }
                            fProxy c = (fProxy)1 / math.sqrt((fProxy)1 + t * t);
                            fProxy s = c * t;

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
                    fProxy* rowJ = utp + (long)j * m;
                    fProxy colNormSq = (fProxy)0;
                    for (int i = 0; i < m; i++) {
                        fProxy bij = rowJ[i];
                        colNormSq += bij * bij;
                    }

                    fProxy sigma = math.sqrt(colNormSq);
                    S[j] = sigma;

                    if (sigma > (fProxy)0) {
                        fProxy invSigma = (fProxy)1 / sigma;
                        for (int i = 0; i < m; i++)
                            rowJ[i] *= invSigma;
                    }
                    else {
                        // Explicit zero for rank-deficient rows
                        for (int i = 0; i < m; i++)
                            rowJ[i] = (fProxy)0;
                        S[j] = (fProxy)0;
                    }
                }
            }

            // Step 4: Selection sort descending by singular value
            for (int j = 0; j < n; j++) {
                int maxIdx = j;
                fProxy maxVal = S[j];

                for (int k = j + 1; k < n; k++) {
                    if (S[k] > maxVal) {
                        maxIdx = k;
                        maxVal = S[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap singular values
                    fProxy tmp = S[j];
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

        /// <summary>svdDecomposition with default eps (Consts.fProxyZeroTreshold).</summary>
        public static bool svdDecomposition(ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V,
                                            int maxSweeps)
            => svdDecomposition(ref U, ref S, ref V, maxSweeps, Consts.fProxyZeroTreshold);

        /// <summary>svdDecomposition with default maxSweeps (30) and eps (Consts.fProxyZeroTreshold).</summary>
        public static bool svdDecomposition(ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V)
            => svdDecomposition(ref U, ref S, ref V, 30, Consts.fProxyZeroTreshold);

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
        public static bool svdValues(in fProxyMxN A, ref fProxyN S, int maxIter, fProxy eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdValues: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("svdValues: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("svdValues: maxIter must be >= 1");
            if (eps <= (fProxy)0)
                throw new ArgumentException("svdValues: eps must be > 0");

            if (n == 0)
                return true;

            int d = m + n;
            var H = new fProxyMxN(d, d, Allocator.Temp, false);
            var eig = new fProxyN(d, Allocator.Temp, false);

            // H = [[0, A], [Aᵀ, 0]]  (symmetric). Zero everything, then fill the two off-diagonal
            // blocks: H[i, m+j] = H[m+j, i] = A[i,j].
            unsafe
            {
                UnsafeUtility.MemClear(H.Data.Ptr, (long)d * d * UnsafeUtility.SizeOf<fProxy>());
            }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy a = A[i, j];
                    H[i, m + j] = a;
                    H[m + j, i] = a;
                }

            bool ok = Eigen.eigenvaluesSymmetric(ref H, ref eig, maxIter, eps);

            // The n largest eigenvalues (descending) are σ_1 ≥ ... ≥ σ_n ≥ 0. Clamp tiny negatives
            // (rounding around a zero singular value) up to 0.
            if (ok)
                for (int i = 0; i < n; i++)
                    S[i] = math.max(eig[i], (fProxy)0);

            eig.Dispose();
            H.Dispose();
            return ok;
        }

        /// <summary>svdValues with default maxIter (30) and eps (Consts.fProxyZeroTreshold).</summary>
        public static bool svdValues(in fProxyMxN A, ref fProxyN S)
            => svdValues(in A, ref S, 30, Consts.fProxyZeroTreshold);
    }
}
