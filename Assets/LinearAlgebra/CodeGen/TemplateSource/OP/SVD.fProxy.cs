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
        /// <remarks>DEPRECATED: prefer <see cref="svdThin(in fProxyMxN, ref fProxyMxN, ref fProxyN, ref fProxyMxN)"/>
        /// (Golub-Kahan bidiagonal SVD, ~3x faster and does not modify its input) for the full SVD, or
        /// <see cref="svdValues(in fProxyMxN, ref fProxyN)"/> for singular values only. Retained for reference.</remarks>
        [System.Obsolete("Prefer SVD.svdThin (Golub-Kahan, ~3x faster) for the full SVD, or SVD.svdValues for singular values only. This one-sided Jacobi SVD is retained for reference.", false)]
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

                            // Fused 2x2 Gram dot: alpha = ||row_p||^2, beta = ||row_q||^2,
                            // gamma = row_p·row_q. Unit-stride reads of length m, 4-way unrolled with
                            // independent partial accumulators (breaks the loop-carried reduction chain).
                            UnsafeOP.gram2x2(rowPU, rowQU, out fProxy alpha, out fProxy beta, out fProxy gamma, m);

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

        // The default-argument overloads forward to the deprecated primitive; suppress the
        // self-referential obsolete warning (618) on the forwarding calls.
#pragma warning disable 618
        /// <summary>svdDecomposition with default eps (Consts.fProxyZeroThreshold).</summary>
        [System.Obsolete("Prefer SVD.svdThin (Golub-Kahan, ~3x faster) for the full SVD, or SVD.svdValues for singular values only. This one-sided Jacobi SVD is retained for reference.", false)]
        public static bool svdDecomposition(ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V,
                                            int maxSweeps)
            => svdDecomposition(ref U, ref S, ref V, maxSweeps, Consts.fProxyZeroThreshold);

        /// <summary>svdDecomposition with default maxSweeps (30) and eps (Consts.fProxyZeroThreshold).</summary>
        [System.Obsolete("Prefer SVD.svdThin (Golub-Kahan, ~3x faster) for the full SVD, or SVD.svdValues for singular values only. This one-sided Jacobi SVD is retained for reference.", false)]
        public static bool svdDecomposition(ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V)
            => svdDecomposition(ref U, ref S, ref V, 30, Consts.fProxyZeroThreshold);
#pragma warning restore 618

        /// <summary>
        /// Singular VALUES only of A (m x n, m >= n), via the Golub-Kahan bidiagonal path: reduce A to
        /// upper bidiagonal form with Householder reflectors (NOT forming U/V) and diagonalize the
        /// bidiagonal with the rotation-free implicit-shift QR. Like the full svdThin it operates
        /// on A directly, so it keeps the condition number κ(A) (not κ(A)²) — small singular values are
        /// not lost — but it skips ALL the orthogonal-factor work, making it the fast values-only path.
        ///
        /// A is NOT modified (worked on a Temp copy). S (length n) receives the singular values,
        /// descending and non-negative. Returns the convergence flag of the bidiagonal QR. Allocates an
        /// O(mn) Temp workspace.
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

            var dVec = new fProxyN(n, Allocator.Temp, false);
            var eVec = new fProxyN(n, Allocator.Temp, false);

            Bidiag.bidiagonalizeValues(in A, ref dVec, ref eVec);
            bool ok = bidiagonalQRValues(ref dVec, ref eVec, n, maxIter);

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // Sort descending (selection sort; no factors to carry).
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    fProxy maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        S[maxIdx] = S[j];
                        S[j] = maxVal;
                    }
                }
            }

            eVec.Dispose();
            dVec.Dispose();
            return ok;
        }

        /// <summary>svdValues with default maxIter (75) and eps (Consts.fProxyZeroThreshold).</summary>
        public static bool svdValues(in fProxyMxN A, ref fProxyN S)
            => svdValues(in A, ref S, 75, Consts.fProxyZeroThreshold);

        // pythag(a,b) = sqrt(a^2 + b^2) without destructive under/overflow.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy svdPythag(fProxy a, fProxy b)
        {
            fProxy aa = math.abs(a), ab = math.abs(b);
            if (aa > ab) { fProxy r = ab / aa; return aa * math.sqrt((fProxy)1 + r * r); }
            if (ab == (fProxy)0) return (fProxy)0;
            { fProxy r = aa / ab; return ab * math.sqrt((fProxy)1 + r * r); }
        }

        // magnitude of a with the sign of b (NR SIGN; b >= 0 -> +|a|).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy svdSign(fProxy a, fProxy b) => b >= (fProxy)0 ? math.abs(a) : -math.abs(a);

        /// <summary>
        /// Full SVD A = U * diag(S) * Vᵀ via Golub-Kahan: Householder bidiagonalization
        /// (Bidiag.bidiagonalize) followed by the implicit-shift bidiagonal QR (Golub-Reinsch).
        /// A (m x n, m >= n) is NOT modified. On output U (m x n) has orthonormal columns (left
        /// singular vectors), S (length n) the singular values (non-negative, DESCENDING), and V
        /// (n x n, NOT transposed) the right singular vectors. Returns true on convergence; false if
        /// the bidiagonal QR hit maxIter (outputs then undefined). Allocates an n x n + 2*n Temp
        /// workspace (plus whatever Bidiag.bidiagonalize uses). For m &lt; n, transpose A and swap U/V.
        /// </summary>
        public static bool svdThin(in fProxyMxN A, ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V,
                                   int maxIter, fProxy eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdThin: A must have m >= n (more rows than columns)");
            if (U.M_Rows != m || U.N_Cols != n)
                throw new ArgumentException("svdThin: U must be m x n");
            if (S.N != n)
                throw new ArgumentException("svdThin: S.N must equal A.N_Cols");
            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("svdThin: V must be square with side equal to A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("svdThin: maxIter must be >= 1");
            if (eps <= (fProxy)0)
                throw new ArgumentException("svdThin: eps must be > 0");

            if (n == 0)
                return true;

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal.
            var B = new fProxyMxN(n, n, Allocator.Temp, false);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);

            // Extract the bidiagonal in NR convention: d = diagonal, e the superdiagonal with
            // e[0] = 0 and e[i] = B[i-1, i] for i = 1..n-1.
            var dVec = new fProxyN(n, Allocator.Temp, false);
            var eVec = new fProxyN(n, Allocator.Temp, false);
            for (int i = 0; i < n; i++) dVec[i] = B[i, i];
            eVec[0] = (fProxy)0;
            for (int i = 1; i < n; i++) eVec[i] = B[i - 1, i];
            B.Dispose();

            // Transpose U (m x n) -> Ut (n x m) and V (n x n) -> Vt (n x n) so the bidiagonal QR's
            // plane rotations hit CONTIGUOUS rows (unit-stride, SIMD via UnsafeOP.jacobiRotate)
            // instead of strided columns — same trick that vectorized eigenSymmetric / svdDecomposition.
            bool ok;
            {
                var Ut = new fProxyMxN(n, m, Allocator.Temp, false);
                var Vt = new fProxyMxN(n, n, Allocator.Temp, false);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        Ut[j, i] = U[i, j];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Vt[j, i] = V[i, j];

                ok = bidiagonalQR(ref Ut, ref dVec, ref eVec, ref Vt, m, n, maxIter);

                if (ok)
                {
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            U[i, j] = Ut[j, i];
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            V[i, j] = Vt[j, i];
                }
                Ut.Dispose();
                Vt.Dispose();
            }

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // sort descending, carrying the matching U and V columns
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    fProxy maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        fProxy tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                        SwapOP.Columns(ref U, j, maxIdx);
                        SwapOP.Columns(ref V, j, maxIdx);
                    }
                }
            }

            dVec.Dispose();
            eVec.Dispose();
            return ok;
        }

        /// <summary>svdThin with default eps (Consts.fProxyZeroThreshold).</summary>
        public static bool svdThin(in fProxyMxN A, ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V,
                                   int maxIter)
            => svdThin(in A, ref U, ref S, ref V, maxIter, Consts.fProxyZeroThreshold);

        /// <summary>svdThin with default maxIter (75) and eps (Consts.fProxyZeroThreshold).</summary>
        public static bool svdThin(in fProxyMxN A, ref fProxyMxN U, ref fProxyN S, ref fProxyMxN V)
            => svdThin(in A, ref U, ref S, ref V, 75, Consts.fProxyZeroThreshold);

        // Implicit-shift QR diagonalization of an upper-bidiagonal matrix (diagonal d, superdiagonal e
        // with e[0]=0), accumulating left rotations into Ut (n x m) ROWS and right rotations into Vt
        // (n x n) ROWS — Ut/Vt are the TRANSPOSES of the SVD's U/V, so each plane rotation touches two
        // contiguous rows (unit-stride, SIMD via UnsafeOP.jacobiRotate). NR's Givens convention
        // a'=c*a+s*b, b'=c*b-s*a equals jacobiRotate(a,b,c,-s). Golub-Reinsch (Numerical Recipes svdcmp
        // diagonalization). The deflation threshold is machine-eps relative to the GLOBAL scale anorm
        // (not a local |d|+|e|), which is what lets FLOAT converge on clustered / zero singular values
        // (same lesson as the symmetric eigen QL). Returns false if any singular value fails to
        // converge within maxIter sweeps.
        static unsafe bool bidiagonalQR(ref fProxyMxN Ut, ref fProxyN d, ref fProxyN e, ref fProxyMxN Vt,
                                        int m, int n, int maxIter)
        {
            fProxy* utp = Ut.Data.Ptr;
            fProxy* vtp = Vt.Data.Ptr;

            fProxy anorm = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy t = math.abs(d[i]) + math.abs(e[i]);
                if (t > anorm) anorm = t;
            }
            fProxy thresh = Consts.fProxyEpsilon * anorm;

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
                        // U columns nm = l-1 and i (= Ut rows nm and i).
                        fProxy c = (fProxy)0, s = (fProxy)1;
                        for (int i = l; i <= k; i++)
                        {
                            fProxy f = s * e[i];
                            e[i] = c * e[i];
                            if (math.abs(f) <= thresh) break;
                            fProxy g = d[i];
                            fProxy h = svdPythag(f, g);
                            d[i] = h;
                            h = (fProxy)1 / h;
                            c = g * h;
                            s = -f * h;
                            UnsafeOP.jacobiRotate(utp + (long)nm * m, utp + (long)i * m, c, -s, m);
                        }
                    }

                    fProxy zz = d[k];
                    if (l == k)
                    {
                        // Converged: make the singular value non-negative.
                        if (zz < (fProxy)0)
                        {
                            d[k] = -zz;
                            fProxy* vrow = vtp + (long)k * n; // column k of V = row k of Vt
                            for (int j = 0; j < n; j++) vrow[j] = -vrow[j];
                        }
                        break;
                    }

                    if (its == maxIter - 1)
                        return false;

                    // Wilkinson shift from the trailing 2x2 of BᵀB.
                    fProxy x = d[l];
                    nm = k - 1;
                    fProxy yy = d[nm];
                    fProxy g2 = e[nm];
                    fProxy h2 = e[k];
                    fProxy f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((fProxy)2 * h2 * yy);
                    g2 = svdPythag(f2, (fProxy)1);
                    f2 = ((x - zz) * (x + zz) + h2 * ((yy / (f2 + svdSign(g2, f2))) - h2)) / x;

                    // Implicit QR sweep: chase the bulge l..k-1, rotating V (right) and U (left).
                    fProxy c2 = (fProxy)1, s2 = (fProxy)1;
                    for (int j = l; j <= nm; j++)
                    {
                        int i = j + 1;
                        g2 = e[i];
                        yy = d[i];
                        h2 = s2 * g2;
                        g2 = c2 * g2;
                        fProxy zr = svdPythag(f2, h2);
                        e[j] = zr;
                        c2 = f2 / zr;
                        s2 = h2 / zr;
                        f2 = x * c2 + g2 * s2;
                        g2 = g2 * c2 - x * s2;
                        h2 = yy * s2;
                        yy *= c2;
                        // V columns j,i = Vt rows j,i
                        UnsafeOP.jacobiRotate(vtp + (long)j * n, vtp + (long)i * n, c2, -s2, n);
                        zr = svdPythag(f2, h2);
                        d[j] = zr;
                        if (zr != (fProxy)0)
                        {
                            zr = (fProxy)1 / zr;
                            c2 = f2 * zr;
                            s2 = h2 * zr;
                        }
                        f2 = c2 * g2 + s2 * yy;
                        x = c2 * yy - s2 * g2;
                        // U columns j,i = Ut rows j,i
                        UnsafeOP.jacobiRotate(utp + (long)j * m, utp + (long)i * m, c2, -s2, m);
                    }
                    e[l] = (fProxy)0;
                    e[k] = f2;
                    d[k] = x;
                }
            }
            return true;
        }

        // SVD of a p×p UPPER-BIDIAGONAL matrix given directly by diagonal d[0..p-1] and superdiagonal
        // e (e[0]=0, e[i]=B[i-1,i]). Skips the Householder bidiagonalization svdThin would redo on an
        // already-bidiagonal matrix. Writes P (p×p, left singular vectors as COLUMNS), S (singular values,
        // DESCENDING, non-negative), Q (p×p, right singular vectors as COLUMNS). Ut/Vt are p×p caller-owned
        // scratch (the transposed accumulators bidiagonalQR fills). d and e are DESTROYED. No allocation.
        // Mirrors svdThin's post-bidiagonalize tail exactly (bidiagonalQR on transposed accumulators, then
        // transpose back, then descending selection sort carrying columns). Returns bidiagonalQR's flag.
        static bool bidiagonalSvdFromDE(ref fProxyN d, ref fProxyN e, ref fProxyMxN Ut, ref fProxyMxN Vt,
                                        ref fProxyMxN P, ref fProxyN S, ref fProxyMxN Q, int p, int maxIter)
        {
            if (p == 0) return true;

            // Clear Ut and Vt (persistent workspace — may hold stale data from a previous call),
            // then set diagonal to 1. This mirrors the identity-init that bidiagonalize's V starts
            // from; here there is no Householder phase so both accumulators start at identity.
            unsafe
            {
                UnsafeUtility.MemClear(Ut.Data.Ptr, (long)Ut.Data.Length * UnsafeUtility.SizeOf<fProxy>());
                UnsafeUtility.MemClear(Vt.Data.Ptr, (long)Vt.Data.Length * UnsafeUtility.SizeOf<fProxy>());
            }
            for (int i = 0; i < p; i++)
            {
                Ut[i, i] = (fProxy)1;
                Vt[i, i] = (fProxy)1;
            }

            bool ok = bidiagonalQR(ref Ut, ref d, ref e, ref Vt, p, p, maxIter);
            if (!ok) return false;

            // Transpose Ut→P and Vt→Q (svdThin's transpose-back-to-column-form step).
            for (int i = 0; i < p; i++)
                for (int j = 0; j < p; j++)
                {
                    P[i, j] = Ut[j, i];
                    Q[i, j] = Vt[j, i];
                }

            // Copy d→S.
            for (int i = 0; i < p; i++) S[i] = d[i];

            // Descending selection sort carrying columns of P and Q (identical to svdThin's sort).
            for (int j = 0; j < p; j++)
            {
                int maxIdx = j;
                fProxy maxVal = S[j];
                for (int kk = j + 1; kk < p; kk++)
                    if (S[kk] > maxVal) { maxIdx = kk; maxVal = S[kk]; }
                if (maxIdx != j)
                {
                    fProxy tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                    SwapOP.Columns(ref P, j, maxIdx);
                    SwapOP.Columns(ref Q, j, maxIdx);
                }
            }

            return true;
        }

        // VALUES-ONLY implicit-shift QR diagonalization of an upper-bidiagonal matrix (diagonal d,
        // superdiagonal e with e[0]=0). Identical scalar recurrence to bidiagonalQR, but it does NOT
        // accumulate any plane rotations (no U/V), so the inner sweeps are pure O(n) work on d/e —
        // the cheap path when only singular values are wanted. On convergence d[k] is made
        // non-negative (no V column to flip). Returns false on non-convergence within maxIter sweeps.
        static bool bidiagonalQRValues(ref fProxyN d, ref fProxyN e, int n, int maxIter)
        {
            fProxy anorm = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy t = math.abs(d[i]) + math.abs(e[i]);
                if (t > anorm) anorm = t;
            }
            fProxy thresh = Consts.fProxyEpsilon * anorm;

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
                        // Cancel e[l] (l > 0): the rotations that would hit U are dropped.
                        fProxy c = (fProxy)0, s = (fProxy)1;
                        for (int i = l; i <= k; i++)
                        {
                            fProxy f = s * e[i];
                            e[i] = c * e[i];
                            if (math.abs(f) <= thresh) break;
                            fProxy g = d[i];
                            fProxy h = svdPythag(f, g);
                            d[i] = h;
                            h = (fProxy)1 / h;
                            c = g * h;
                            s = -f * h;
                        }
                    }

                    fProxy zz = d[k];
                    if (l == k)
                    {
                        if (zz < (fProxy)0)
                            d[k] = -zz;
                        break;
                    }

                    if (its == maxIter - 1)
                        return false;

                    fProxy x = d[l];
                    nm = k - 1;
                    fProxy yy = d[nm];
                    fProxy g2 = e[nm];
                    fProxy h2 = e[k];
                    fProxy f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((fProxy)2 * h2 * yy);
                    g2 = svdPythag(f2, (fProxy)1);
                    f2 = ((x - zz) * (x + zz) + h2 * ((yy / (f2 + svdSign(g2, f2))) - h2)) / x;

                    fProxy c2 = (fProxy)1, s2 = (fProxy)1;
                    for (int j = l; j <= nm; j++)
                    {
                        int i = j + 1;
                        g2 = e[i];
                        yy = d[i];
                        h2 = s2 * g2;
                        g2 = c2 * g2;
                        fProxy zr = svdPythag(f2, h2);
                        e[j] = zr;
                        c2 = f2 / zr;
                        s2 = h2 / zr;
                        f2 = x * c2 + g2 * s2;
                        g2 = g2 * c2 - x * s2;
                        h2 = yy * s2;
                        yy *= c2;
                        zr = svdPythag(f2, h2);
                        d[j] = zr;
                        if (zr != (fProxy)0)
                        {
                            zr = (fProxy)1 / zr;
                            c2 = f2 * zr;
                            s2 = h2 * zr;
                        }
                        f2 = c2 * g2 + s2 * yy;
                        x = c2 * yy - s2 * g2;
                    }
                    e[l] = (fProxy)0;
                    e[k] = f2;
                    d[k] = x;
                }
            }
            return true;
        }
    }
}
