#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class SVD {

        /// <summary>
        /// Singular VALUES only of A (m x n, m >= n), via the Golub-Kahan bidiagonal path: reduce A to
        /// upper bidiagonal form with Householder reflectors (NOT forming U/V) and diagonalize the
        /// bidiagonal with the rotation-free implicit-shift QR. Like the full thin it operates
        /// on A directly, so it keeps the condition number κ(A) (not κ(A)²) — small singular values are
        /// not lost — but it skips ALL the orthogonal-factor work, making it the fast values-only path.
        ///
        /// A is NOT modified (worked on a Temp copy). S (length n) receives the singular values,
        /// descending and non-negative. Returns the convergence flag of the bidiagonal QR. Allocates an
        /// O(mn) Temp workspace.
        /// </summary>
        public static bool values(in floatMxN A, ref floatN S, int maxIter, float eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("values: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("values: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("values: maxIter must be >= 1");
            if (eps <= (float)0)
                throw new ArgumentException("values: eps must be > 0");

            if (n == 0)
                return true;

            var dVec = new floatN(n, Allocator.Temp, false);
            var eVec = new floatN(n, Allocator.Temp, false);

            Bidiag.values(in A, ref dVec, ref eVec);
            bool ok = bidiagonalQRValues(ref dVec, ref eVec, n, maxIter);

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // Sort descending (selection sort; no factors to carry).
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    float maxVal = S[j];
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

        /// <summary>values with default maxIter (75) and eps (Consts.floatZeroThreshold).</summary>
        public static bool values(in floatMxN A, ref floatN S)
            => values(in A, ref S, 75, Consts.floatZeroThreshold);

        /// <summary>
        /// values using a reusable workspace (Arena.floatSVDValuesCache(m, n)) — zero-alloc.
        /// Semantics identical to the allocating overload; see that one for full documentation.
        /// </summary>
        public static bool values(in floatMxN A, ref floatN S, ref floatSVDValuesCache ws,
                                     int maxIter, float eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("values: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("values: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("values: maxIter must be >= 1");
            if (eps <= (float)0)
                throw new ArgumentException("values: eps must be > 0");
            RequireSvdValuesWorkspace(in ws, n);

            if (n == 0)
                return true;

            var dVec = ws.dVec;
            var eVec = ws.eVec;

            Bidiag.values(in A, ref dVec, ref eVec, ref ws.BidiagWs);
            bool ok = bidiagonalQRValues(ref dVec, ref eVec, n, maxIter);

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // Sort descending (selection sort; no factors to carry).
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    float maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        S[maxIdx] = S[j];
                        S[j] = maxVal;
                    }
                }
            }

            return ok;
        }

        /// <summary>values (workspace) with default maxIter (75) and eps (Consts.floatZeroThreshold).</summary>
        public static bool values(in floatMxN A, ref floatN S, ref floatSVDValuesCache ws)
            => values(in A, ref S, ref ws, 75, Consts.floatZeroThreshold);

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

        // sqrt(a^2 + b^2 + a*b) (a,b >= 0), an upper bound on the spectral norm of a 2x2 upper-
        // bidiagonal block, scaled to avoid overflow (mirrors lanbpro's FUDGE*sqrt(a^2+b^2+a*b)
        // ||A||_2 estimate). Used only to size the round-off floor in the partial-reorth
        // omega-recurrence — an order-of-magnitude estimate, so the cross term must not be dropped.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float svdAnormBlock(float a, float b)
        {
            float aa = math.abs(a), ab = math.abs(b);
            float mx = math.max(aa, ab);
            if (mx == (float)0) return (float)0;
            float ra = aa / mx, rb = ab / mx;
            return mx * math.sqrt(ra * ra + rb * rb + ra * rb);
        }

        /// <summary>
        /// Full SVD A = U * diag(S) * Vᵀ via Golub-Kahan: Householder bidiagonalization
        /// (Bidiag.decomp) followed by the implicit-shift bidiagonal QR (Golub-Reinsch).
        /// A (m x n, m >= n) is NOT modified. On output U (m x n) has orthonormal columns (left
        /// singular vectors), S (length n) the singular values (non-negative, DESCENDING), and V
        /// (n x n, NOT transposed) the right singular vectors. Returns true on convergence; false if
        /// the bidiagonal QR hit maxIter (outputs then undefined). Allocates an n x n + 2*n Temp
        /// workspace (plus whatever Bidiag.decomp uses). For m &lt; n, transpose A and swap U/V.
        /// </summary>
        public static bool thin(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                   int maxIter, float eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("thin: A must have m >= n (more rows than columns)");
            if (U.M_Rows != m || U.N_Cols != n)
                throw new ArgumentException("thin: U must be m x n");
            if (S.N != n)
                throw new ArgumentException("thin: S.N must equal A.N_Cols");
            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("thin: V must be square with side equal to A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("thin: maxIter must be >= 1");
            if (eps <= (float)0)
                throw new ArgumentException("thin: eps must be > 0");

            if (n == 0)
                return true;

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal.
            var B = new floatMxN(n, n, Allocator.Temp, false);
            Bidiag.decomp(in A, ref U, ref B, ref V);

            // Extract the bidiagonal in NR convention: d = diagonal, e the superdiagonal with
            // e[0] = 0 and e[i] = B[i-1, i] for i = 1..n-1.
            var dVec = new floatN(n, Allocator.Temp, false);
            var eVec = new floatN(n, Allocator.Temp, false);
            for (int i = 0; i < n; i++) dVec[i] = B[i, i];
            eVec[0] = (float)0;
            for (int i = 1; i < n; i++) eVec[i] = B[i - 1, i];
            B.Dispose();

            // Transpose U (m x n) -> Ut (n x m) and V (n x n) -> Vt (n x n) so the bidiagonal QR's
            // plane rotations hit CONTIGUOUS rows (unit-stride, SIMD via UnsafeOP.jacobiRotate)
            // instead of strided columns — same trick that vectorized Eigen.symmetric (and the
            // deleted one-sided Jacobi SVD; see git history).
            bool ok;
            {
                var Ut = new floatMxN(n, m, Allocator.Temp, false);
                var Vt = new floatMxN(n, n, Allocator.Temp, false);
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
                    float maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        float tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                        Swap.Columns(ref U, j, maxIdx);
                        Swap.Columns(ref V, j, maxIdx);
                    }
                }
            }

            dVec.Dispose();
            eVec.Dispose();
            return ok;
        }

        /// <summary>thin with default eps (Consts.floatZeroThreshold).</summary>
        public static bool thin(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                   int maxIter)
            => thin(in A, ref U, ref S, ref V, maxIter, Consts.floatZeroThreshold);

        /// <summary>thin with default maxIter (75) and eps (Consts.floatZeroThreshold).</summary>
        public static bool thin(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V)
            => thin(in A, ref U, ref S, ref V, 75, Consts.floatZeroThreshold);

        /// <summary>
        /// thin using a reusable workspace (Arena.floatSVDThinCache(m, n)) — zero-alloc (including
        /// the inner Bidiag.decomp call, via the workspace's nested BidiagWs). Semantics
        /// identical to the allocating overload; see that one for full documentation.
        /// </summary>
        public static bool thin(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                   ref floatSVDThinCache ws, int maxIter, float eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("thin: A must have m >= n (more rows than columns)");
            if (U.M_Rows != m || U.N_Cols != n)
                throw new ArgumentException("thin: U must be m x n");
            if (S.N != n)
                throw new ArgumentException("thin: S.N must equal A.N_Cols");
            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("thin: V must be square with side equal to A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("thin: maxIter must be >= 1");
            if (eps <= (float)0)
                throw new ArgumentException("thin: eps must be > 0");
            RequireSvdThinWorkspace(in ws, m, n);

            if (n == 0)
                return true;

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal (caller-workspace scratch, zero-alloc).
            var B = ws.B;
            Bidiag.decomp(in A, ref U, ref B, ref V, ref ws.BidiagWs);

            // Extract bidiagonal in NR convention — see thin's allocating overload above.
            var dVec = ws.dVec;
            var eVec = ws.eVec;
            for (int i = 0; i < n; i++) dVec[i] = B[i, i];
            eVec[0] = (float)0;
            for (int i = 1; i < n; i++) eVec[i] = B[i - 1, i];

            // Transpose to contiguous rows for the bidiagonal QR — see the allocating overload above.
            bool ok;
            {
                var Ut = ws.Ut;
                var Vt = ws.Vt;
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
            }

            if (ok)
            {
                for (int i = 0; i < n; i++) S[i] = dVec[i];

                // sort descending (see allocating overload above)
                for (int j = 0; j < n; j++)
                {
                    int maxIdx = j;
                    float maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        float tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                        Swap.Columns(ref U, j, maxIdx);
                        Swap.Columns(ref V, j, maxIdx);
                    }
                }
            }

            return ok;
        }

        /// <summary>thin (workspace) with default eps (Consts.floatZeroThreshold).</summary>
        public static bool thin(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                   ref floatSVDThinCache ws, int maxIter)
            => thin(in A, ref U, ref S, ref V, ref ws, maxIter, Consts.floatZeroThreshold);

        /// <summary>thin (workspace) with default maxIter (75) and eps (Consts.floatZeroThreshold).</summary>
        public static bool thin(in floatMxN A, ref floatMxN U, ref floatN S, ref floatMxN V,
                                   ref floatSVDThinCache ws)
            => thin(in A, ref U, ref S, ref V, ref ws, 75, Consts.floatZeroThreshold);

        // Implicit-shift QR diagonalization of an upper-bidiagonal matrix (d diagonal, e superdiagonal,
        // e[0]=0); accumulates left rotations into Ut (n x m) ROWS and right into Vt (n x n) ROWS — the
        // TRANSPOSES of U/V, so each rotation touches contiguous rows (SIMD via jacobiRotate). NR's
        // Givens convention a'=c*a+s*b, b'=c*b-s*a equals jacobiRotate(a,b,c,-s) (Golub-Reinsch /
        // Numerical Recipes svdcmp). Deflation threshold is machine-eps relative to the GLOBAL scale
        // anorm (not local |d|+|e|) — needed for FLOAT to converge on clustered/zero singular values
        // (same lesson as the symmetric eigen QL). Returns false if a value fails to converge within maxIter.
        static unsafe bool bidiagonalQR(ref floatMxN Ut, ref floatN d, ref floatN e, ref floatMxN Vt,
                                        int m, int n, int maxIter)
        {
            float* utp = Ut.Data.Ptr;
            float* vtp = Vt.Data.Ptr;

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
                        // U columns nm = l-1 and i (= Ut rows nm and i).
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
                            UnsafeOP.jacobiRotate(utp + (long)nm * m, utp + (long)i * m, c, -s, m);
                        }
                    }

                    float zz = d[k];
                    if (l == k)
                    {
                        // Converged: make the singular value non-negative.
                        if (zz < (float)0)
                        {
                            d[k] = -zz;
                            float* vrow = vtp + (long)k * n; // column k of V = row k of Vt
                            for (int j = 0; j < n; j++) vrow[j] = -vrow[j];
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
                        // V columns j,i = Vt rows j,i
                        UnsafeOP.jacobiRotate(vtp + (long)j * n, vtp + (long)i * n, c2, -s2, n);
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
                        // U columns j,i = Ut rows j,i
                        UnsafeOP.jacobiRotate(utp + (long)j * m, utp + (long)i * m, c2, -s2, m);
                    }
                    e[l] = (float)0;
                    e[k] = f2;
                    d[k] = x;
                }
            }
            return true;
        }

        // SVD of a p×p UPPER-BIDIAGONAL matrix given directly by diagonal d[0..p-1] and superdiagonal
        // e (e[0]=0, e[i]=B[i-1,i]). Skips the Householder bidiagonalization thin would redo on an
        // already-bidiagonal matrix. Writes P (p×p, left singular vectors as COLUMNS), S (singular values,
        // DESCENDING, non-negative), Q (p×p, right singular vectors as COLUMNS). Ut/Vt are p×p caller-owned
        // scratch (the transposed accumulators bidiagonalQR fills). d and e are DESTROYED. No allocation.
        // Mirrors thin's post-bidiagonalize tail exactly (bidiagonalQR on transposed accumulators, then
        // transpose back, then descending selection sort carrying columns). Returns bidiagonalQR's flag.
        static bool bidiagonalSvdFromDE(ref floatN d, ref floatN e, ref floatMxN Ut, ref floatMxN Vt,
                                        ref floatMxN P, ref floatN S, ref floatMxN Q, int p, int maxIter)
        {
            if (p == 0) return true;

            // Clear Ut and Vt (persistent workspace — may hold stale data from a previous call),
            // then set diagonal to 1. This mirrors the identity-init that bidiagonalize's V starts
            // from; here there is no Householder phase so both accumulators start at identity.
            unsafe
            {
                UnsafeUtility.MemClear(Ut.Data.Ptr, (long)Ut.Data.Length * UnsafeUtility.SizeOf<float>());
                UnsafeUtility.MemClear(Vt.Data.Ptr, (long)Vt.Data.Length * UnsafeUtility.SizeOf<float>());
            }
            for (int i = 0; i < p; i++)
            {
                Ut[i, i] = (float)1;
                Vt[i, i] = (float)1;
            }

            bool ok = bidiagonalQR(ref Ut, ref d, ref e, ref Vt, p, p, maxIter);
            if (!ok) return false;

            // Transpose Ut→P and Vt→Q (thin's transpose-back-to-column-form step).
            for (int i = 0; i < p; i++)
                for (int j = 0; j < p; j++)
                {
                    P[i, j] = Ut[j, i];
                    Q[i, j] = Vt[j, i];
                }

            // Copy d→S.
            for (int i = 0; i < p; i++) S[i] = d[i];

            // Descending selection sort carrying columns of P and Q (identical to thin's sort).
            for (int j = 0; j < p; j++)
            {
                int maxIdx = j;
                float maxVal = S[j];
                for (int kk = j + 1; kk < p; kk++)
                    if (S[kk] > maxVal) { maxIdx = kk; maxVal = S[kk]; }
                if (maxIdx != j)
                {
                    float tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                    Swap.Columns(ref P, j, maxIdx);
                    Swap.Columns(ref Q, j, maxIdx);
                }
            }

            return true;
        }

        // VALUES-ONLY implicit-shift QR diagonalization of an upper-bidiagonal matrix (diagonal d,
        // superdiagonal e with e[0]=0). Identical scalar recurrence to bidiagonalQR, but it does NOT
        // accumulate any plane rotations (no U/V), so the inner sweeps are pure O(n) work on d/e —
        // the cheap path when only singular values are wanted. On convergence d[k] is made
        // non-negative (no V column to flip). Returns false on non-convergence within maxIter sweeps.
        static bool bidiagonalQRValues(ref floatN d, ref floatN e, int n, int maxIter)
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
                        // Cancel e[l] (l > 0): the rotations that would hit U are dropped.
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
                        }
                    }

                    float zz = d[k];
                    if (l == k)
                    {
                        if (zz < (float)0)
                            d[k] = -zz;
                        break;
                    }

                    if (its == maxIter - 1)
                        return false;

                    float x = d[l];
                    nm = k - 1;
                    float yy = d[nm];
                    float g2 = e[nm];
                    float h2 = e[k];
                    float f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((float)2 * h2 * yy);
                    g2 = svdPythag(f2, (float)1);
                    f2 = ((x - zz) * (x + zz) + h2 * ((yy / (f2 + svdSign(g2, f2))) - h2)) / x;

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
