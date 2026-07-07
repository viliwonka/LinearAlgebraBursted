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
        /// descending and non-negative. Returns an <see cref="SVDInfo"/> (implicit-bool ==
        /// Converged) carrying the bidiagonal QR's convergence status; on MaxIterations S is
        /// unwritten. Allocates an O(mn) Temp workspace.
        /// </summary>
        public static SVDInfo values(in doubleMxN A, ref doubleN S, int maxIter, double eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("values: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("values: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("values: maxIter must be >= 1");
            if (eps <= (double)0)
                throw new ArgumentException("values: eps must be > 0");

            if (n == 0)
                return new SVDInfo { status = IterativeSolveStatus.Converged, sweeps = 0, converged = 0 };

            var dVec = new doubleN(n, Allocator.Temp, false);
            var eVec = new doubleN(n, Allocator.Temp, false);

            Bidiag.values(in A, ref dVec, ref eVec);
            bool ok = bidiagonalQRValues(ref dVec, ref eVec, n, maxIter, out int sweeps, out int convergedCount);

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
            return new SVDInfo
            {
                status = ok ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                sweeps = sweeps,
                converged = convergedCount
            };
        }

        /// <summary>values with default maxIter (Consts.sweepBudget(A.N_Cols)) and eps (Consts.doubleZeroThreshold).</summary>
        public static SVDInfo values(in doubleMxN A, ref doubleN S)
            => values(in A, ref S, Consts.sweepBudget(A.N_Cols), Consts.doubleZeroThreshold);

        /// <summary>
        /// values using a reusable workspace (Arena.doubleSVDValuesCache(m, n)) — zero-alloc.
        /// Semantics identical to the allocating overload; see that one for full documentation.
        /// </summary>
        public static SVDInfo values(in doubleMxN A, ref doubleN S, ref doubleSVDValuesCache ws,
                                     int maxIter, double eps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("values: A must have m >= n (more rows than columns)");
            if (S.N != n)
                throw new ArgumentException("values: S.N must equal A.N_Cols");
            if (maxIter < 1)
                throw new ArgumentException("values: maxIter must be >= 1");
            if (eps <= (double)0)
                throw new ArgumentException("values: eps must be > 0");
            RequireSvdValuesWorkspace(in ws, n);

            if (n == 0)
                return new SVDInfo { status = IterativeSolveStatus.Converged, sweeps = 0, converged = 0 };

            var dVec = ws.dVec;
            var eVec = ws.eVec;

            Bidiag.values(in A, ref dVec, ref eVec, ref ws.BidiagWs);
            bool ok = bidiagonalQRValues(ref dVec, ref eVec, n, maxIter, out int sweeps, out int convergedCount);

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

            return new SVDInfo
            {
                status = ok ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                sweeps = sweeps,
                converged = convergedCount
            };
        }

        /// <summary>values (workspace) with default maxIter (Consts.sweepBudget(A.N_Cols)) and eps (Consts.doubleZeroThreshold).</summary>
        public static SVDInfo values(in doubleMxN A, ref doubleN S, ref doubleSVDValuesCache ws)
            => values(in A, ref S, ref ws, Consts.sweepBudget(A.N_Cols), Consts.doubleZeroThreshold);

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

        // sqrt(a^2 + b^2 + a*b) (a,b >= 0), an upper bound on the spectral norm of a 2x2 upper-
        // bidiagonal block, scaled to avoid overflow (mirrors lanbpro's FUDGE*sqrt(a^2+b^2+a*b)
        // ||A||_2 estimate). Used only to size the round-off floor in the partial-reorth
        // omega-recurrence — an order-of-magnitude estimate, so the cross term must not be dropped.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double svdAnormBlock(double a, double b)
        {
            double aa = math.abs(a), ab = math.abs(b);
            double mx = math.max(aa, ab);
            if (mx == (double)0) return (double)0;
            double ra = aa / mx, rb = ab / mx;
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
        public static SVDInfo thin(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                   int maxIter, double eps)
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
            if (eps <= (double)0)
                throw new ArgumentException("thin: eps must be > 0");

            if (n == 0)
                return new SVDInfo { status = IterativeSolveStatus.Converged, sweeps = 0, converged = 0 };

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal.
            var B = new doubleMxN(n, n, Allocator.Temp, false);
            Bidiag.decomp(in A, ref U, ref B, ref V);

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
            // instead of strided columns — the same trick that vectorized Eigen.symmetric.
            bool ok;
            int sweeps, convergedCount;
            {
                var Ut = new doubleMxN(n, m, Allocator.Temp, false);
                var Vt = new doubleMxN(n, n, Allocator.Temp, false);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        Ut[j, i] = U[i, j];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Vt[j, i] = V[i, j];

                ok = bidiagonalQR(ref Ut, ref dVec, ref eVec, ref Vt, m, n, maxIter, out sweeps, out convergedCount);

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
                        Swap.Columns(ref U, j, maxIdx);
                        Swap.Columns(ref V, j, maxIdx);
                    }
                }
            }

            dVec.Dispose();
            eVec.Dispose();
            return new SVDInfo
            {
                status = ok ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                sweeps = sweeps,
                converged = convergedCount
            };
        }

        /// <summary>thin with default eps (Consts.doubleZeroThreshold).</summary>
        public static SVDInfo thin(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                   int maxIter)
            => thin(in A, ref U, ref S, ref V, maxIter, Consts.doubleZeroThreshold);

        /// <summary>thin with default maxIter (Consts.sweepBudget(A.N_Cols)) and eps (Consts.doubleZeroThreshold).</summary>
        public static SVDInfo thin(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V)
            => thin(in A, ref U, ref S, ref V, Consts.sweepBudget(A.N_Cols), Consts.doubleZeroThreshold);

        /// <summary>
        /// thin using a reusable workspace (Arena.doubleSVDThinCache(m, n)) — zero-alloc (including
        /// the inner Bidiag.decomp call, via the workspace's nested BidiagWs). Semantics
        /// identical to the allocating overload; see that one for full documentation.
        /// </summary>
        public static SVDInfo thin(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                   ref doubleSVDThinCache ws, int maxIter, double eps)
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
            if (eps <= (double)0)
                throw new ArgumentException("thin: eps must be > 0");
            RequireSvdThinWorkspace(in ws, m, n);

            if (n == 0)
                return new SVDInfo { status = IterativeSolveStatus.Converged, sweeps = 0, converged = 0 };

            // Phase 1: A = U * B * Vᵀ, B upper bidiagonal (caller-workspace scratch, zero-alloc).
            var B = ws.B;
            Bidiag.decomp(in A, ref U, ref B, ref V, ref ws.BidiagWs);

            // Extract bidiagonal in NR convention — see thin's allocating overload above.
            var dVec = ws.dVec;
            var eVec = ws.eVec;
            for (int i = 0; i < n; i++) dVec[i] = B[i, i];
            eVec[0] = (double)0;
            for (int i = 1; i < n; i++) eVec[i] = B[i - 1, i];

            // Transpose to contiguous rows for the bidiagonal QR — see the allocating overload above.
            bool ok;
            int sweeps, convergedCount;
            {
                var Ut = ws.Ut;
                var Vt = ws.Vt;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        Ut[j, i] = U[i, j];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Vt[j, i] = V[i, j];

                ok = bidiagonalQR(ref Ut, ref dVec, ref eVec, ref Vt, m, n, maxIter, out sweeps, out convergedCount);

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
                    double maxVal = S[j];
                    for (int k = j + 1; k < n; k++)
                        if (S[k] > maxVal) { maxIdx = k; maxVal = S[k]; }
                    if (maxIdx != j)
                    {
                        double tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
                        Swap.Columns(ref U, j, maxIdx);
                        Swap.Columns(ref V, j, maxIdx);
                    }
                }
            }

            return new SVDInfo
            {
                status = ok ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                sweeps = sweeps,
                converged = convergedCount
            };
        }

        /// <summary>thin (workspace) with default eps (Consts.doubleZeroThreshold).</summary>
        public static SVDInfo thin(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                   ref doubleSVDThinCache ws, int maxIter)
            => thin(in A, ref U, ref S, ref V, ref ws, maxIter, Consts.doubleZeroThreshold);

        /// <summary>thin (workspace) with default maxIter (Consts.sweepBudget(A.N_Cols)) and eps (Consts.doubleZeroThreshold).</summary>
        public static SVDInfo thin(in doubleMxN A, ref doubleMxN U, ref doubleN S, ref doubleMxN V,
                                   ref doubleSVDThinCache ws)
            => thin(in A, ref U, ref S, ref V, ref ws, Consts.sweepBudget(A.N_Cols), Consts.doubleZeroThreshold);

        // Implicit-shift QR diagonalization of an upper-bidiagonal matrix (d diagonal, e superdiagonal,
        // e[0]=0); accumulates left rotations into Ut (n x m) ROWS and right into Vt (n x n) ROWS — the
        // TRANSPOSES of U/V, so each rotation touches contiguous rows (SIMD via jacobiRotate). NR's
        // Givens convention a'=c*a+s*b, b'=c*b-s*a equals jacobiRotate(a,b,c,-s) (Golub-Reinsch /
        // Numerical Recipes svdcmp). Deflation threshold is machine-eps relative to the GLOBAL scale
        // anorm (not local |d|+|e|) — needed for FLOAT to converge on clustered/zero singular values
        // (same lesson as the symmetric eigen QL). Returns false if a value fails to converge within
        // maxIter. `sweeps` (out) is the MAXIMUM number of QR sweeps consumed by any single value
        // (0 if every value deflated immediately; == maxIter on a false return, the exhausted
        // budget); `convergedCount` (out) is how many values had already converged when the loop
        // stopped (== n on a true return) — both feed SVDInfo at the call site.
        static unsafe bool bidiagonalQR(ref doubleMxN Ut, ref doubleN d, ref doubleN e, ref doubleMxN Vt,
                                        int m, int n, int maxIter, out int sweeps, out int convergedCount)
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

            sweeps = 0;
            convergedCount = 0;

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
                        if (its > sweeps) sweeps = its;
                        convergedCount++;
                        break;
                    }

                    if (its == maxIter - 1)
                    {
                        sweeps = maxIter;
                        return false;
                    }

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

        // SVD of a p×p UPPER-BIDIAGONAL matrix given directly by diagonal d[0..p-1] and superdiagonal
        // e (e[0]=0, e[i]=B[i-1,i]). Skips the Householder bidiagonalization thin would redo on an
        // already-bidiagonal matrix. Writes P (p×p, left singular vectors as COLUMNS), S (singular values,
        // DESCENDING, non-negative), Q (p×p, right singular vectors as COLUMNS). Ut/Vt are p×p caller-owned
        // scratch (the transposed accumulators bidiagonalQR fills). d and e are DESTROYED. No allocation.
        // Mirrors thin's post-bidiagonalization tail exactly (bidiagonalQR on transposed accumulators, then
        // transpose back, then descending selection sort carrying columns). Returns bidiagonalQR's flag;
        // `sweeps`/`convergedCount` are threaded straight through from bidiagonalQR (see its doc comment).
        static bool bidiagonalSvdFromDE(ref doubleN d, ref doubleN e, ref doubleMxN Ut, ref doubleMxN Vt,
                                        ref doubleMxN P, ref doubleN S, ref doubleMxN Q, int p, int maxIter,
                                        out int sweeps, out int convergedCount)
        {
            if (p == 0) { sweeps = 0; convergedCount = 0; return true; }

            // Clear Ut and Vt (persistent workspace — may hold stale data from a previous call),
            // then set diagonal to 1. This mirrors the identity-init that Bidiag.decomp's V starts
            // from; here there is no Householder phase so both accumulators start at identity.
            unsafe
            {
                UnsafeUtility.MemClear(Ut.Data.Ptr, (long)Ut.Data.Length * UnsafeUtility.SizeOf<double>());
                UnsafeUtility.MemClear(Vt.Data.Ptr, (long)Vt.Data.Length * UnsafeUtility.SizeOf<double>());
            }
            for (int i = 0; i < p; i++)
            {
                Ut[i, i] = (double)1;
                Vt[i, i] = (double)1;
            }

            bool ok = bidiagonalQR(ref Ut, ref d, ref e, ref Vt, p, p, maxIter, out sweeps, out convergedCount);
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
                double maxVal = S[j];
                for (int kk = j + 1; kk < p; kk++)
                    if (S[kk] > maxVal) { maxIdx = kk; maxVal = S[kk]; }
                if (maxIdx != j)
                {
                    double tmp = S[j]; S[j] = S[maxIdx]; S[maxIdx] = tmp;
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
        // `sweeps`/`convergedCount` follow the same convention as bidiagonalQR's out params.
        static bool bidiagonalQRValues(ref doubleN d, ref doubleN e, int n, int maxIter,
                                       out int sweeps, out int convergedCount)
        {
            double anorm = (double)0;
            for (int i = 0; i < n; i++)
            {
                double t = math.abs(d[i]) + math.abs(e[i]);
                if (t > anorm) anorm = t;
            }
            double thresh = Consts.doubleEpsilon * anorm;

            sweeps = 0;
            convergedCount = 0;

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
                        if (its > sweeps) sweeps = its;
                        convergedCount++;
                        break;
                    }

                    if (its == maxIter - 1)
                    {
                        sweeps = maxIter;
                        return false;
                    }

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
