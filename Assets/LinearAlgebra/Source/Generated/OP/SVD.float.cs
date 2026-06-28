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

                            // Fused 2x2 Gram dot: alpha = ||row_p||^2, beta = ||row_q||^2,
                            // gamma = row_p·row_q. Unit-stride reads of length m, 4-way unrolled with
                            // independent partial accumulators (breaks the loop-carried reduction chain).
                            UnsafeOP.gram2x2(rowPU, rowQU, out float alpha, out float beta, out float gamma, m);

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

        // pythag(a,b) = sqrt(a^2 + b^2) without destructive under/overflow.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float svdPythag(float a, float b)
        {
            float aa = math.abs(a), ab = math.abs(b);
            if (aa > ab) { float r = ab / aa; return aa * math.sqrt((float)1 + r * r); }
            if (ab == (float)0) return (float)0;
            { float r = aa / ab; return ab * math.sqrt((float)1 + r * r); }
        }

        // magnitude of a with the sign of b (NR SIGN; b >= 0 -> +|a|).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float svdSign(float a, float b) => b >= (float)0 ? math.abs(a) : -math.abs(a);

        /// <summary>
        /// Full SVD A = U * diag(S) * Vᵀ via Golub-Kahan: Householder bidiagonalization
        /// (Bidiag.bidiagonalize) followed by the implicit-shift bidiagonal QR (Golub-Reinsch).
        /// A (m x n, m >= n) is NOT modified. On output U (m x n) has orthonormal columns (left
        /// singular vectors), S (length n) the singular values (non-negative, DESCENDING), and V
        /// (n x n, NOT transposed) the right singular vectors. Returns true on convergence; false if
        /// the bidiagonal QR hit maxIter (outputs then undefined). Allocates an n x n + 2*n Temp
        /// workspace (plus whatever Bidiag.bidiagonalize uses). For m &lt; n, transpose A and swap U/V.
        /// </summary>
        public static bool svdGolubKahan(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                         int maxIter, float eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdGolubKahan: A must have m >= n (more rows than columns)");
            if (U.M_Rows != m || U.N_Cols != n)
                throw new ArgumentException("svdGolubKahan: U must be m x n");
            if (S.N != n)
                throw new ArgumentException("svdGolubKahan: S.N must equal A.N_Cols");
            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("svdGolubKahan: V must be square with side equal to A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("svdGolubKahan: maxIter must be >= 1");
            if (eps <= (float)0)
                throw new ArgumentException("svdGolubKahan: eps must be > 0");

            if (n == 0)
                return true;

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal.
            var B = new floatMxN(n, n, Allocator.Temp, false);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);

            // Extract the bidiagonal in NR convention: d = diagonal, e the superdiagonal with
            // e[0] = 0 and e[i] = B[i-1, i] for i = 1..n-1.
            var dVec = new floatN(n, Allocator.Temp, false);
            var eVec = new floatN(n, Allocator.Temp, false);
            for (int i = 0; i < n; i++) dVec[i] = B[i, i];
            eVec[0] = (float)0;
            for (int i = 1; i < n; i++) eVec[i] = B[i - 1, i];
            B.Dispose();

            bool ok = bidiagonalQR(ref U, ref dVec, ref eVec, ref V, m, n, maxIter);

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // sort descending, carrying the matching U and V columns
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    float maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        float tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                        SwapOP.Columns(ref U, j, maxIdx);
                        SwapOP.Columns(ref V, j, maxIdx);
                    }
                }
            }

            dVec.Dispose();
            eVec.Dispose();
            return ok;
        }

        /// <summary>svdGolubKahan with default eps (Consts.floatZeroTreshold).</summary>
        public static bool svdGolubKahan(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                         int maxIter)
            => svdGolubKahan(in A, ref U, ref S, ref V, maxIter, Consts.floatZeroTreshold);

        /// <summary>svdGolubKahan with default maxIter (75) and eps (Consts.floatZeroTreshold).</summary>
        public static bool svdGolubKahan(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V)
            => svdGolubKahan(in A, ref U, ref S, ref V, 75, Consts.floatZeroTreshold);

        // Implicit-shift QR diagonalization of an upper-bidiagonal matrix (diagonal d, superdiagonal e
        // with e[0]=0), accumulating left rotations into U (m x n) columns and right rotations into V
        // (n x n) columns. Golub-Reinsch (Numerical Recipes svdcmp diagonalization). The deflation
        // threshold is machine-eps relative to the GLOBAL scale anorm (not a local |d|+|e|), which is
        // what lets FLOAT converge on clustered / zero singular values (same lesson as the symmetric
        // eigen QL). Returns false if any singular value fails to converge within maxIter sweeps.
        static bool bidiagonalQR(ref floatMxN U, ref floatN d, ref floatN e, ref floatMxN V,
                                 int m, int n, int maxIter)
        {
            float anorm = (float)0;
            for (int i = 0; i < n; i++)
            {
                float t = math.abs(d[i]) + math.abs(e[i]);
                if (t > anorm) anorm = t;
            }
            float thresh = Consts.floatEpsilon * anorm;

            for (int k = n - 1; k >= 0; k--)
            {
                for (int its = 0; its < maxIter; its++)
                {
                    bool flag = true;
                    int l;
                    int nm = 0;
                    for (l = k; l >= 0; l--)
                    {
                        nm = l - 1;
                        if (l == 0 || math.abs(e[l]) <= thresh) { flag = false; break; }
                        if (math.abs(d[nm]) <= thresh) break;
                    }

                    if (flag)
                    {
                        // Cancel e[l] (l > 0): Givens rotations zero the superdiagonal, applied to
                        // U columns nm = l-1 and i.
                        float c = (float)0, s = (float)1;
                        for (int i = l; i <= k; i++)
                        {
                            float f = s * e[i];
                            e[i] = c * e[i];
                            if (math.abs(f) <= thresh) break;
                            float g = d[i];
                            float h = svdPythag(f, g);
                            d[i] = h;
                            h = (float)1 / h;
                            c = g * h;
                            s = -f * h;
                            for (int j = 0; j < m; j++)
                            {
                                float y = U[j, nm];
                                float z = U[j, i];
                                U[j, nm] = y * c + z * s;
                                U[j, i]  = z * c - y * s;
                            }
                        }
                    }

                    float zz = d[k];
                    if (l == k)
                    {
                        // Converged: make the singular value non-negative.
                        if (zz < (float)0)
                        {
                            d[k] = -zz;
                            for (int j = 0; j < n; j++) V[j, k] = -V[j, k];
                        }
                        break;
                    }

                    if (its == maxIter - 1)
                        return false;

                    // Wilkinson shift from the trailing 2x2 of BᵀB.
                    float x = d[l];
                    nm = k - 1;
                    float yy = d[nm];
                    float g2 = e[nm];
                    float h2 = e[k];
                    float f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((float)2 * h2 * yy);
                    g2 = svdPythag(f2, (float)1);
                    f2 = ((x - zz) * (x + zz) + h2 * ((yy / (f2 + svdSign(g2, f2))) - h2)) / x;

                    // Implicit QR sweep: chase the bulge l..k-1, rotating V (right) and U (left).
                    float c2 = (float)1, s2 = (float)1;
                    for (int j = l; j <= nm; j++)
                    {
                        int i = j + 1;
                        g2 = e[i];
                        yy = d[i];
                        h2 = s2 * g2;
                        g2 = c2 * g2;
                        float zr = svdPythag(f2, h2);
                        e[j] = zr;
                        c2 = f2 / zr;
                        s2 = h2 / zr;
                        f2 = x * c2 + g2 * s2;
                        g2 = g2 * c2 - x * s2;
                        h2 = yy * s2;
                        yy *= c2;
                        for (int jj = 0; jj < n; jj++)
                        {
                            float xv = V[jj, j];
                            float zv = V[jj, i];
                            V[jj, j] = xv * c2 + zv * s2;
                            V[jj, i] = zv * c2 - xv * s2;
                        }
                        zr = svdPythag(f2, h2);
                        d[j] = zr;
                        if (zr != (float)0)
                        {
                            zr = (float)1 / zr;
                            c2 = f2 * zr;
                            s2 = h2 * zr;
                        }
                        f2 = c2 * g2 + s2 * yy;
                        x = c2 * yy - s2 * g2;
                        for (int jj = 0; jj < m; jj++)
                        {
                            float yu = U[jj, j];
                            float zu = U[jj, i];
                            U[jj, j] = yu * c2 + zu * s2;
                            U[jj, i] = zu * c2 - yu * s2;
                        }
                    }
                    e[l] = (float)0;
                    e[k] = f2;
                    d[k] = x;
                }
            }
            return true;
        }
    }
}
