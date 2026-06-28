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

            // Transpose U into Ut (n x m) so that column j of U becomes ROW j of Ut.
            // Working on rows of Ut gives unit-stride contiguous access that Burst vectorizes.
            // Vt (n x n) accumulates V^T: row operations on Vt correspond to column operations
            // on V, so at the end we transpose Vt back into V.
            var Ut = new doubleMxN(n, m, Allocator.Temp, true);  // will fill every element
            var Vt = new doubleMxN(n, n, Allocator.Temp, false); // zeroed; diagonal set below

            // Fill Ut: row j of Ut = column j of U
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Ut[j, i] = U[i, j];

            // Initialize Vt to identity (Vt accumulates right singular vectors as rows)
            for (int i = 0; i < n; i++)
                Vt[i, i] = (double)1;

            // Step 2: One-sided Jacobi sweeps — all inner loops now hit contiguous rows
            bool converged = false;

            unsafe
            {
                double* utp = Ut.Data.Ptr;
                double* vtp = Vt.Data.Ptr;

                for (int sweep = 0; sweep < maxSweeps; sweep++) {

                    int rotations = 0;

                    for (int p = 0; p < n - 1; p++) {
                        for (int q = p + 1; q < n; q++) {

                            double* rowPU = utp + (long)p * m;
                            double* rowQU = utp + (long)q * m;

                            // Fused 2x2 Gram dot: alpha = ||row_p||^2, beta = ||row_q||^2,
                            // gamma = row_p·row_q. Unit-stride reads of length m, 4-way unrolled with
                            // independent partial accumulators (breaks the loop-carried reduction chain).
                            UnsafeOP.gram2x2(rowPU, rowQU, out double alpha, out double beta, out double gamma, m);

                            // Skip if either column is zero or pair is already orthogonal
                            if (alpha == (double)0 || beta == (double)0)
                                continue;

                            if (math.abs(gamma) <= eps * math.sqrt(alpha) * math.sqrt(beta))
                                continue;

                            // Rutishauser rotation (byte-for-byte unchanged from the original)
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
                    double* rowJ = utp + (long)j * m;
                    double colNormSq = (double)0;
                    for (int i = 0; i < m; i++) {
                        double bij = rowJ[i];
                        colNormSq += bij * bij;
                    }

                    double sigma = math.sqrt(colNormSq);
                    S[j] = sigma;

                    if (sigma > (double)0) {
                        double invSigma = (double)1 / sigma;
                        for (int i = 0; i < m; i++)
                            rowJ[i] *= invSigma;
                    }
                    else {
                        // Explicit zero for rank-deficient rows
                        for (int i = 0; i < m; i++)
                            rowJ[i] = (double)0;
                        S[j] = (double)0;
                    }
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

        /// <summary>svdDecomposition with default eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdDecomposition(ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                            int maxSweeps)
            => svdDecomposition(ref U, ref S, ref V, maxSweeps, Consts.doubleZeroTreshold);

        /// <summary>svdDecomposition with default maxSweeps (30) and eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdDecomposition(ref doubleMxN U, ref doubleN S, ref doubleMxN V)
            => svdDecomposition(ref U, ref S, ref V, 30, Consts.doubleZeroTreshold);

        /// <summary>
        /// Singular VALUES only of A (m x n, m >= n), via the Golub-Kahan bidiagonal path: reduce A to
        /// upper bidiagonal form with Householder reflectors (NOT forming U/V) and diagonalize the
        /// bidiagonal with the rotation-free implicit-shift QR. Like the full svdGolubKahan it operates
        /// on A directly, so it keeps the condition number κ(A) (not κ(A)²) — small singular values are
        /// not lost — but it skips ALL the orthogonal-factor work, making it the fast values-only path.
        ///
        /// A is NOT modified (worked on a Temp copy). S (length n) receives the singular values,
        /// descending and non-negative. Returns the convergence flag of the bidiagonal QR. Allocates an
        /// O(mn) Temp workspace.
        /// </summary>
        public static bool svdValues(in doubleMxN A, ref doubleN S, int maxIter, double eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdValues: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("svdValues: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("svdValues: maxIter must be >= 1");
            if (eps <= (double)0)
                throw new ArgumentException("svdValues: eps must be > 0");

            if (n == 0)
                return true;

            var dVec = new doubleN(n, Allocator.Temp, false);
            var eVec = new doubleN(n, Allocator.Temp, false);

            Bidiag.bidiagonalizeValues(in A, ref dVec, ref eVec);
            bool ok = bidiagonalQRValues(ref dVec, ref eVec, n, maxIter);

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // Sort descending (selection sort; no factors to carry).
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    double maxVal = S[j];
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

        /// <summary>svdValues with default maxIter (75) and eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdValues(in doubleMxN A, ref doubleN S)
            => svdValues(in A, ref S, 75, Consts.doubleZeroTreshold);

        // pythag(a,b) = sqrt(a^2 + b^2) without destructive under/overflow.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double svdPythag(double a, double b)
        {
            double aa = math.abs(a), ab = math.abs(b);
            if (aa > ab) { double r = ab / aa; return aa * math.sqrt((double)1 + r * r); }
            if (ab == (double)0) return (double)0;
            { double r = aa / ab; return ab * math.sqrt((double)1 + r * r); }
        }

        // magnitude of a with the sign of b (NR SIGN; b >= 0 -> +|a|).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double svdSign(double a, double b) => b >= (double)0 ? math.abs(a) : -math.abs(a);

        /// <summary>
        /// Full SVD A = U * diag(S) * Vᵀ via Golub-Kahan: Householder bidiagonalization
        /// (Bidiag.bidiagonalize) followed by the implicit-shift bidiagonal QR (Golub-Reinsch).
        /// A (m x n, m >= n) is NOT modified. On output U (m x n) has orthonormal columns (left
        /// singular vectors), S (length n) the singular values (non-negative, DESCENDING), and V
        /// (n x n, NOT transposed) the right singular vectors. Returns true on convergence; false if
        /// the bidiagonal QR hit maxIter (outputs then undefined). Allocates an n x n + 2*n Temp
        /// workspace (plus whatever Bidiag.bidiagonalize uses). For m &lt; n, transpose A and swap U/V.
        /// </summary>
        public static bool svdGolubKahan(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                         int maxIter, double eps)
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
            if (eps <= (double)0)
                throw new ArgumentException("svdGolubKahan: eps must be > 0");

            if (n == 0)
                return true;

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal.
            var B = new doubleMxN(n, n, Allocator.Temp, false);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);

            // Extract the bidiagonal in NR convention: d = diagonal, e the superdiagonal with
            // e[0] = 0 and e[i] = B[i-1, i] for i = 1..n-1.
            var dVec = new doubleN(n, Allocator.Temp, false);
            var eVec = new doubleN(n, Allocator.Temp, false);
            for (int i = 0; i < n; i++) dVec[i] = B[i, i];
            eVec[0] = (double)0;
            for (int i = 1; i < n; i++) eVec[i] = B[i - 1, i];
            B.Dispose();

            // Transpose U (m x n) -> Ut (n x m) and V (n x n) -> Vt (n x n) so the bidiagonal QR's
            // plane rotations hit CONTIGUOUS rows (unit-stride, SIMD via UnsafeOP.jacobiRotate)
            // instead of strided columns — same trick that vectorized eigenSymmetric / svdDecomposition.
            bool ok;
            {
                var Ut = new doubleMxN(n, m, Allocator.Temp, false);
                var Vt = new doubleMxN(n, n, Allocator.Temp, false);
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
                    double maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        double tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                        SwapOP.Columns(ref U, j, maxIdx);
                        SwapOP.Columns(ref V, j, maxIdx);
                    }
                }
            }

            dVec.Dispose();
            eVec.Dispose();
            return ok;
        }

        /// <summary>svdGolubKahan with default eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdGolubKahan(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                         int maxIter)
            => svdGolubKahan(in A, ref U, ref S, ref V, maxIter, Consts.doubleZeroTreshold);

        /// <summary>svdGolubKahan with default maxIter (75) and eps (Consts.doubleZeroTreshold).</summary>
        public static bool svdGolubKahan(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V)
            => svdGolubKahan(in A, ref U, ref S, ref V, 75, Consts.doubleZeroTreshold);

        // Implicit-shift QR diagonalization of an upper-bidiagonal matrix (diagonal d, superdiagonal e
        // with e[0]=0), accumulating left rotations into Ut (n x m) ROWS and right rotations into Vt
        // (n x n) ROWS — Ut/Vt are the TRANSPOSES of the SVD's U/V, so each plane rotation touches two
        // contiguous rows (unit-stride, SIMD via UnsafeOP.jacobiRotate). NR's Givens convention
        // a'=c*a+s*b, b'=c*b-s*a equals jacobiRotate(a,b,c,-s). Golub-Reinsch (Numerical Recipes svdcmp
        // diagonalization). The deflation threshold is machine-eps relative to the GLOBAL scale anorm
        // (not a local |d|+|e|), which is what lets FLOAT converge on clustered / zero singular values
        // (same lesson as the symmetric eigen QL). Returns false if any singular value fails to
        // converge within maxIter sweeps.
        static unsafe bool bidiagonalQR(ref doubleMxN Ut, ref doubleN d, ref doubleN e, ref doubleMxN Vt,
                                        int m, int n, int maxIter)
        {
            double* utp = Ut.Data.Ptr;
            double* vtp = Vt.Data.Ptr;

            double anorm = (double)0;
            for (int i = 0; i < n; i++)
            {
                double t = math.abs(d[i]) + math.abs(e[i]);
                if (t > anorm) anorm = t;
            }
            double thresh = Consts.doubleEpsilon * anorm;

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
                        double c = (double)0, s = (double)1;
                        for (int i = l; i <= k; i++)
                        {
                            double f = s * e[i];
                            e[i] = c * e[i];
                            if (math.abs(f) <= thresh) break;
                            double g = d[i];
                            double h = svdPythag(f, g);
                            d[i] = h;
                            h = (double)1 / h;
                            c = g * h;
                            s = -f * h;
                            UnsafeOP.jacobiRotate(utp + (long)nm * m, utp + (long)i * m, c, -s, m);
                        }
                    }

                    double zz = d[k];
                    if (l == k)
                    {
                        // Converged: make the singular value non-negative.
                        if (zz < (double)0)
                        {
                            d[k] = -zz;
                            double* vrow = vtp + (long)k * n; // column k of V = row k of Vt
                            for (int j = 0; j < n; j++) vrow[j] = -vrow[j];
                        }
                        break;
                    }

                    if (its == maxIter - 1)
                        return false;

                    // Wilkinson shift from the trailing 2x2 of BᵀB.
                    double x = d[l];
                    nm = k - 1;
                    double yy = d[nm];
                    double g2 = e[nm];
                    double h2 = e[k];
                    double f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((double)2 * h2 * yy);
                    g2 = svdPythag(f2, (double)1);
                    f2 = ((x - zz) * (x + zz) + h2 * ((yy / (f2 + svdSign(g2, f2))) - h2)) / x;

                    // Implicit QR sweep: chase the bulge l..k-1, rotating V (right) and U (left).
                    double c2 = (double)1, s2 = (double)1;
                    for (int j = l; j <= nm; j++)
                    {
                        int i = j + 1;
                        g2 = e[i];
                        yy = d[i];
                        h2 = s2 * g2;
                        g2 = c2 * g2;
                        double zr = svdPythag(f2, h2);
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
                        if (zr != (double)0)
                        {
                            zr = (double)1 / zr;
                            c2 = f2 * zr;
                            s2 = h2 * zr;
                        }
                        f2 = c2 * g2 + s2 * yy;
                        x = c2 * yy - s2 * g2;
                        // U columns j,i = Ut rows j,i
                        UnsafeOP.jacobiRotate(utp + (long)j * m, utp + (long)i * m, c2, -s2, m);
                    }
                    e[l] = (double)0;
                    e[k] = f2;
                    d[k] = x;
                }
            }
            return true;
        }

        // VALUES-ONLY implicit-shift QR diagonalization of an upper-bidiagonal matrix (diagonal d,
        // superdiagonal e with e[0]=0). Identical scalar recurrence to bidiagonalQR, but it does NOT
        // accumulate any plane rotations (no U/V), so the inner sweeps are pure O(n) work on d/e —
        // the cheap path when only singular values are wanted. On convergence d[k] is made
        // non-negative (no V column to flip). Returns false on non-convergence within maxIter sweeps.
        static bool bidiagonalQRValues(ref doubleN d, ref doubleN e, int n, int maxIter)
        {
            double anorm = (double)0;
            for (int i = 0; i < n; i++)
            {
                double t = math.abs(d[i]) + math.abs(e[i]);
                if (t > anorm) anorm = t;
            }
            double thresh = Consts.doubleEpsilon * anorm;

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
                        double c = (double)0, s = (double)1;
                        for (int i = l; i <= k; i++)
                        {
                            double f = s * e[i];
                            e[i] = c * e[i];
                            if (math.abs(f) <= thresh) break;
                            double g = d[i];
                            double h = svdPythag(f, g);
                            d[i] = h;
                            h = (double)1 / h;
                            c = g * h;
                            s = -f * h;
                        }
                    }

                    double zz = d[k];
                    if (l == k)
                    {
                        if (zz < (double)0)
                            d[k] = -zz;
                        break;
                    }

                    if (its == maxIter - 1)
                        return false;

                    double x = d[l];
                    nm = k - 1;
                    double yy = d[nm];
                    double g2 = e[nm];
                    double h2 = e[k];
                    double f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((double)2 * h2 * yy);
                    g2 = svdPythag(f2, (double)1);
                    f2 = ((x - zz) * (x + zz) + h2 * ((yy / (f2 + svdSign(g2, f2))) - h2)) / x;

                    double c2 = (double)1, s2 = (double)1;
                    for (int j = l; j <= nm; j++)
                    {
                        int i = j + 1;
                        g2 = e[i];
                        yy = d[i];
                        h2 = s2 * g2;
                        g2 = c2 * g2;
                        double zr = svdPythag(f2, h2);
                        e[j] = zr;
                        c2 = f2 / zr;
                        s2 = h2 / zr;
                        f2 = x * c2 + g2 * s2;
                        g2 = g2 * c2 - x * s2;
                        h2 = yy * s2;
                        yy *= c2;
                        zr = svdPythag(f2, h2);
                        d[j] = zr;
                        if (zr != (double)0)
                        {
                            zr = (double)1 / zr;
                            c2 = f2 * zr;
                            s2 = h2 * zr;
                        }
                        f2 = c2 * g2 + s2 * yy;
                        x = c2 * yy - s2 * g2;
                    }
                    e[l] = (double)0;
                    e[k] = f2;
                    d[k] = x;
                }
            }
            return true;
        }
    }
}
