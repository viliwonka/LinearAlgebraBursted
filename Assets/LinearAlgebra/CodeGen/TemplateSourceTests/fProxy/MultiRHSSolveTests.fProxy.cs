using System;

using BULA;
using BULA.Gallery;   // opt-in: fProxyGallery.fProxyLaplacian1D(n), fProxyGallery.fProxyLauchli(n,eps), ...

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Multi-RHS (TRSM-style) direct solvers — AX = B with B and X whole matrices (each RHS a column).
// Every case is validated two ways: (1) a residual norm ‖A·X − B‖ on a consistent system (backward
// stable ⇒ tiny), and (2) a COLUMN-BY-COLUMN cross-check against the already-trusted single-RHS
// (vector) solver — the multi-RHS result for column c must match running the vector solver on B[:,c].
// The cross-check is the primary validator: it pins the new matrix code directly to the proven vector
// path (they differ only by summation-order rounding), so it stays tight even when the solution
// itself is non-unique (min-norm / pinv / rank-deficient). Tolerances scale with Consts.fProxySqrtEps
// (float ≈ 3.45e-4, double ≈ 1.49e-8), matching the SolverBatteryTests idiom.
public class fProxyMultiRHSSolveTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            BlasTrsmUpperLower,
            LUMultiRHS,
            CHOMultiRHS,
            QRMultiRHSSquare,
            QRMultiRHSTall,
            QRCPMultiRHS,
            QRCPMultiRHSRankDeficient,
            QRCPMultiRHSBlocked,
            LQMultiRHSMinNorm,
            CHOPMultiRHSFullRank,
            CHOPMultiRHSRankDeficient,
            SVDMultiRHSTall,
            SVDMultiRHSWide,
        }

        public TestType Type;

        // [0] flag (1 = failure), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.BlasTrsmUpperLower:        BlasTrsmUpperLower();        break;
                case TestType.LUMultiRHS:                LUMultiRHS();                break;
                case TestType.CHOMultiRHS:               CHOMultiRHS();               break;
                case TestType.QRMultiRHSSquare:          QRMultiRHSSquare();          break;
                case TestType.QRMultiRHSTall:            QRMultiRHSTall();            break;
                case TestType.QRCPMultiRHS:              QRCPMultiRHS();              break;
                case TestType.QRCPMultiRHSRankDeficient: QRCPMultiRHSRankDeficient(); break;
                case TestType.QRCPMultiRHSBlocked:       QRCPMultiRHSBlocked();       break;
                case TestType.LQMultiRHSMinNorm:         LQMultiRHSMinNorm();         break;
                case TestType.CHOPMultiRHSFullRank:      CHOPMultiRHSFullRank();      break;
                case TestType.CHOPMultiRHSRankDeficient: CHOPMultiRHSRankDeficient(); break;
                case TestType.SVDMultiRHSTall:           SVDMultiRHSTall();           break;
                case TestType.SVDMultiRHSWide:           SVDMultiRHSWide();           break;
            }
        }

        // =====================================================================
        // Blas TRSM primitives — triUpper / triLower against a matrix RHS.
        // =====================================================================

        void BlasTrsmUpperLower()
        {

            int n = 6, k = 3;

            // Well-conditioned upper- and lower-triangular factors (unit-ish diagonal + small off-diag).
            var U = new fProxyMxN(n, n, Allocator.Temp);   // zero-init
            var L = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                U[i, i] = (fProxy)(i + 2);
                L[i, i] = (fProxy)(i + 2);
                for (int j = i + 1; j < n; j++) U[i, j] = (fProxy)0.5 / (fProxy)(j - i + 1);
                for (int j = 0; j < i; j++)     L[i, j] = (fProxy)0.5 / (fProxy)(i - j + 1);
            }

            var Xtrue = MakeX(n, k);

            // Upper: B = U·Xtrue, solve U X = B, check.
            var Bu = new fProxyMxN(n, k, Allocator.Temp); Blas.dot(in U, in Xtrue, ref Bu);
            var Xu = new fProxyMxN(in Bu, Allocator.Temp);
            Blas.triUpper(ref U, ref Xu);
            CheckMatClose(in Xu, in Xtrue, Band(in U, (fProxy)200));
            CheckResidual(in U, in Xu, in Bu, (fProxy)200);

            // Lower: B = L·Xtrue, solve L X = B, check.
            var Bl = new fProxyMxN(n, k, Allocator.Temp); Blas.dot(in L, in Xtrue, ref Bl);
            var Xl = new fProxyMxN(in Bl, Allocator.Temp);
            Blas.triLower(ref L, ref Xl);
            CheckMatClose(in Xl, in Xtrue, Band(in L, (fProxy)200));
            CheckResidual(in L, in Xl, in Bl, (fProxy)200);
        }

        // =====================================================================
        // LU — compact decompSolve + solveInPlace, cross-checked vs the vector path.
        // =====================================================================

        void LUMultiRHS()
        {

            int n = 6, k = 3;
            var A = fProxyGallery.fProxyLaplacian1D(n);
            var Xtrue = MakeX(n, k);
            var B = new fProxyMxN(n, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);

            // decompInPlace once, decompSolve on the whole block.
            var LUm = new fProxyMxN(in A, Allocator.Temp);
            var P = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref LUm, ref P);
            var X = new fProxyMxN(in B, Allocator.Temp);
            LU.decompSolve(ref LUm, in P, ref X);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)200));
            CheckResidual(in A, in X, in B, (fProxy)200);

            // Column cross-check vs the vector decompSolve (same factor).
            for (int c = 0; c < k; c++)
            {
                var bc = GetCol(in B, c);
                LU.decompSolve(ref LUm, in P, ref bc);   // bc := column solution
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)200));
            }

            // solveInPlace (factor+solve fused) matches too.
            var A2 = new fProxyMxN(in A, Allocator.Temp);
            var P2 = new Pivot(n, Allocator.Temp);
            var X2 = new fProxyMxN(in B, Allocator.Temp);
            LU.solveInPlace(ref A2, ref P2, ref X2);
            CheckMatClose(in X2, in X, Band(in A, (fProxy)200));

            P2.Dispose();
            P.Dispose();
        }

        // =====================================================================
        // CHO — decompSolve + solveInPlace on an SPD system.
        // =====================================================================

        void CHOMultiRHS()
        {

            int n = 6, k = 3;
            var A = fProxyGallery.fProxyLaplacian1D(n);
            var Xtrue = MakeX(n, k);
            var B = new fProxyMxN(n, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);

            var L = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(CHO.decomp(in A, ref L));
            var X = new fProxyMxN(in B, Allocator.Temp);
            CHO.decompSolve(ref L, ref X);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)200));
            CheckResidual(in A, in X, in B, (fProxy)200);

            for (int c = 0; c < k; c++)
            {
                var bc = GetCol(in B, c);
                CHO.decompSolve(ref L, ref bc);
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)200));
            }

            var A2 = new fProxyMxN(in A, Allocator.Temp);
            var X2 = new fProxyMxN(in B, Allocator.Temp);
            AssertTrue(CHO.solveInPlace(ref A2, ref X2));
            CheckMatClose(in X2, in X, Band(in A, (fProxy)200));
        }

        // =====================================================================
        // QR — decompSolve (reuse Q,R) + fused solveInPlace, square and tall LS.
        // =====================================================================

        void QRMultiRHSSquare()
        {

            int n = 6, k = 3;
            var A = fProxyGallery.fProxyLaplacian1D(n);
            var Xtrue = MakeX(n, k);
            var B = new fProxyMxN(n, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);

            // decompSolve from a precomputed QR (B preserved).
            var Q = new fProxyMxN(in A, Allocator.Temp);
            var R = new fProxyMxN(n, n, Allocator.Temp);
            QR.decompInPlace(ref Q, ref R);
            var Xd = new fProxyMxN(n, k, Allocator.Temp);
            QR.decompSolve(ref Q, ref R, ref B, ref Xd);
            CheckMatClose(in Xd, in Xtrue, Band(in A, (fProxy)300));
            CheckResidual(in A, in Xd, in B, (fProxy)300);

            // fused solveInPlace (destroys A and B) matches, and matches the vector path per column.
            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var Bw = new fProxyMxN(in B, Allocator.Temp);
            var Xs = new fProxyMxN(n, k, Allocator.Temp);
            QR.solveInPlace(ref Aw, ref Bw, ref Xs);
            CheckMatClose(in Xs, in Xd, Band(in A, (fProxy)300));

            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(n, Allocator.Temp);
                QR.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in Xd, c, in xc, Band(in A, (fProxy)300));
            }
        }

        void QRMultiRHSTall()
        {

            int nc = 4, k = 3;
            var A = fProxyGallery.fProxyLauchli(nc, (fProxy)0.5);   // (nc+1) x nc, tall full-column-rank
            int m = A.M_Rows;
            var Xtrue = MakeX(nc, k);
            var B = new fProxyMxN(m, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);   // consistent, in range(A)

            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var Bw = new fProxyMxN(in B, Allocator.Temp);
            var X = new fProxyMxN(nc, k, Allocator.Temp);
            QR.solveInPlace(ref Aw, ref Bw, ref X);

            // consistent LS ⇒ solution is exactly Xtrue, residual ≈ 0.
            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)100));
            CheckResidual(in A, in X, in B, (fProxy)100);

            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(nc, Allocator.Temp);
                QR.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)100));
            }
        }

        // =====================================================================
        // QRCP — rank-safe multi-RHS: full rank + rank-deficient.
        // =====================================================================

        void QRCPMultiRHS()
        {

            int nc = 4, k = 3;
            var A = fProxyGallery.fProxyLauchli(nc, (fProxy)0.5);   // tall full rank
            int m = A.M_Rows;
            var Xtrue = MakeX(nc, k);
            var B = new fProxyMxN(m, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);   // reference, kept

            // Fused solveInPlace DESTROYS A and B (like QR) — use copies.
            var Aq = new fProxyMxN(in A, Allocator.Temp);
            var Bq = new fProxyMxN(in B, Allocator.Temp);
            var X = new fProxyMxN(nc, k, Allocator.Temp);
            RankInfo info = QRCP.solveInPlace(ref Aq, ref Bq, ref X);
            AssertTrue(info.Solved);
            RecordEq(info.rank, nc);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)300));
            CheckResidual(in A, in X, in B, (fProxy)300);   // original A, B

            // Cross-check vs the vector fused solveInPlace per column (each destroys its own copies).
            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(nc, Allocator.Temp);
                QRCP.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)300));
            }

            // decompSolve (preserved-B route, from a precomputed factorization) matches the fused result.
            var Q = new fProxyMxN(in A, Allocator.Temp);
            var R = new fProxyMxN(nc, nc, Allocator.Temp);
            var P = new Pivot(nc, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);
            var Xd = new fProxyMxN(nc, k, Allocator.Temp);
            QRCP.decompSolve(ref Q, ref R, in P, ref B, ref Xd, (fProxy)(-1));   // B preserved
            CheckMatClose(in Xd, in X, Band(in A, (fProxy)300));
            P.Dispose();
        }

        void QRCPMultiRHSRankDeficient()
        {

            int n = 4, k = 2;
            var A = fProxyGallery.fProxyPei(n, (fProxy)0);   // all-ones, rank 1
            var B = MakeX(n, k);           // arbitrary RHS, kept

            // Fused solveInPlace destroys A and B — use copies.
            var Aq = new fProxyMxN(in A, Allocator.Temp);
            var Bq = new fProxyMxN(in B, Allocator.Temp);
            var X = new fProxyMxN(n, k, Allocator.Temp);
            RankInfo info = QRCP.solveInPlace(ref Aq, ref Bq, ref X);
            RecordEq(info.rank, 1);

            // Cross-check vs the vector path per column — the truncated (basic) solution must agree.
            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(n, Allocator.Temp);
                QRCP.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)300));
            }
        }

        // Large enough (n >= 2·QRCP_BLOCK) to drive the BLOCKED fused core's multi-RHS reflector apply.
        void QRCPMultiRHSBlocked()
        {

            int nc = 80, k = 3;
            var A = fProxyGallery.fProxyLauchli(nc, (fProxy)0.5);   // 81 x 80, tall full rank, blocked path
            int m = A.M_Rows;
            var Xtrue = MakeX(nc, k);
            var B = new fProxyMxN(m, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);

            var Aq = new fProxyMxN(in A, Allocator.Temp);
            var Bq = new fProxyMxN(in B, Allocator.Temp);
            var X = new fProxyMxN(nc, k, Allocator.Temp);
            RankInfo info = QRCP.solveInPlace(ref Aq, ref Bq, ref X);
            AssertTrue(info.Solved);
            RecordEq(info.rank, nc);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)2000));
            CheckResidual(in A, in X, in B, (fProxy)2000);

            // Cross-check vs the vector fused solveInPlace (also blocked) per column.
            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(nc, Allocator.Temp);
                QRCP.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)2000));
            }
        }

        // =====================================================================
        // LQ — minimum-norm multi-RHS on a wide, full-row-rank system.
        // =====================================================================

        void LQMultiRHSMinNorm()
        {

            int k = 3;
            var Tall = fProxyGallery.fProxyLauchli(4, (fProxy)0.5);   // 5 x 4
            var A = new fProxyMxN(Tall.N_Cols, Tall.M_Rows, Allocator.Temp); // 4 x 5, wide full row rank
            Blas.trans(in Tall, ref A);
            int m = A.M_Rows;   // 4
            int n = A.N_Cols;   // 5

            var B = MakeX(m, k);   // any RHS (A full row rank ⇒ consistent)

            var X = new fProxyMxN(n, k, Allocator.Temp);
            LQ.minNormSolve(in A, in B, ref X);

            CheckResidual(in A, in X, in B, (fProxy)200);   // A·X ≈ B

            for (int c = 0; c < k; c++)
            {
                var bc = GetCol(in B, c);
                var xc = new fProxyN(n, Allocator.Temp);
                LQ.minNormSolve(in A, in bc, ref xc);   // A not modified
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)200));
            }
        }

        // =====================================================================
        // CHOP — pivoted Cholesky multi-RHS: full rank + rank-deficient (min-norm).
        // =====================================================================

        void CHOPMultiRHSFullRank()
        {

            int n = 6, k = 3;
            var A = fProxyGallery.fProxyLaplacian1D(n);   // SPD full rank
            var Xtrue = MakeX(n, k);
            var B = new fProxyMxN(n, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);

            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var P = new Pivot(n, Allocator.Temp);
            var X = new fProxyMxN(in B, Allocator.Temp);
            RankInfo info = CHOP.solveInPlace(ref Aw, ref P, ref X);
            AssertTrue(info.Solved);
            RecordEq(info.rank, n);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)300));
            CheckResidual(in A, in X, in B, (fProxy)300);

            // Cross-check vs the vector solveInPlace per column.
            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var Pc = new Pivot(n, Allocator.Temp);
                var bc = GetCol(in B, c);
                CHOP.solveInPlace(ref Ac, ref Pc, ref bc);
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)300));
                Pc.Dispose();
            }

            P.Dispose();
        }

        void CHOPMultiRHSRankDeficient()
        {

            int n = 4, k = 2;
            var A = fProxyGallery.fProxyPei(n, (fProxy)0);   // all-ones PSD, rank 1
            var B = MakeX(n, k);

            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var P = new Pivot(n, Allocator.Temp);
            var X = new fProxyMxN(in B, Allocator.Temp);
            RankInfo info = CHOP.solveInPlace(ref Aw, ref P, ref X);
            RecordEq(info.rank, 1);

            // Cross-check vs the vector min-norm path per column.
            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var Pc = new Pivot(n, Allocator.Temp);
                var bc = GetCol(in B, c);
                CHOP.solveInPlace(ref Ac, ref Pc, ref bc);
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)300));
                Pc.Dispose();
            }

            P.Dispose();
        }

        // =====================================================================
        // SVD — minimum-norm least-squares multi-RHS: tall and wide.
        // =====================================================================

        void SVDMultiRHSTall()
        {

            int nc = 4, k = 3;
            var A = fProxyGallery.fProxyLauchli(nc, (fProxy)1E-2);   // (nc+1) x nc tall
            int m = A.M_Rows;
            var Xtrue = MakeX(nc, k);
            var B = new fProxyMxN(m, k, Allocator.Temp); Blas.dot(in A, in Xtrue, ref B);

            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var X = new fProxyMxN(nc, k, Allocator.Temp);
            RankInfo info = SVD.pinvSolve(ref Aw, in B, ref X);
            AssertTrue(info.Solved);

            CheckResidual(in A, in X, in B, (fProxy)500);

            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(nc, Allocator.Temp);
                SVD.pinvSolve(ref Ac, in bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)500));
            }
        }

        void SVDMultiRHSWide()
        {

            int k = 3;
            var Tall = fProxyGallery.fProxyLauchli(4, (fProxy)0.5);   // 5 x 4
            var A = new fProxyMxN(Tall.N_Cols, Tall.M_Rows, Allocator.Temp); // 4 x 5 wide
            Blas.trans(in Tall, ref A);
            int m = A.M_Rows;
            int n = A.N_Cols;

            var B = MakeX(m, k);

            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var X = new fProxyMxN(n, k, Allocator.Temp);
            RankInfo info = SVD.pinvSolve(ref Aw, in B, ref X);
            AssertTrue(info.Solved);

            CheckResidual(in A, in X, in B, (fProxy)500);   // consistent underdetermined

            for (int c = 0; c < k; c++)
            {
                var Ac = new fProxyMxN(in A, Allocator.Temp);
                var bc = GetCol(in B, c);
                var xc = new fProxyN(n, Allocator.Temp);
                SVD.pinvSolve(ref Ac, in bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)500));
            }
        }

        // =====================================================================
        // helpers
        // =====================================================================

        fProxyMxN MakeX(int n, int k)
        {
            var X = new fProxyMxN(n, k, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int c = 0; c < k; c++)
                    X[i, c] = (fProxy)(1 + i - (fProxy)0.5 * c + (fProxy)0.25 * (c + 1) * (i % 3));
            return X;
        }

        fProxyN GetCol(in fProxyMxN M, int c)
        {
            int n = M.M_Rows;
            var v = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) v[i] = M[i, c];
            return v;
        }

        // backward-stable band scaled by matrix magnitude and precision.
        fProxy Band(in fProxyMxN A, fProxy factor)
        {
            return (MatMaxAbs(in A) + (fProxy)1) * factor * Consts.fProxySqrtEps;
        }

        fProxy MatMaxAbs(in fProxyMxN A)
        {
            fProxy mx = (fProxy)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    fProxy v = math.abs(A[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        void CheckMatClose(in fProxyMxN X, in fProxyMxN Y, fProxy tol)
        {
            for (int i = 0; i < X.M_Rows; i++)
                for (int c = 0; c < X.N_Cols; c++)
                    AssertClose(X[i, c], Y[i, c], tol);
        }

        void CheckColClose(in fProxyMxN X, int c, in fProxyN v, fProxy tol)
        {
            for (int i = 0; i < v.N; i++)
                AssertClose(X[i, c], v[i], tol);
        }

        // ‖A·X − B‖ (max abs entry) ≤ band.
        void CheckResidual(in fProxyMxN A, in fProxyMxN X, in fProxyMxN B, fProxy factor)
        {
            var AX = new fProxyMxN(A.M_Rows, X.N_Cols, Allocator.Temp);
            Blas.dot(in A, in X, ref AX);
            fProxy tol = Band(in A, factor);
            for (int i = 0; i < B.M_Rows; i++)
                for (int c = 0; c < B.N_Cols; c++)
                    AssertClose(AX[i, c], B[i, c], tol);
        }

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = (fProxy)0; Fail[2] = (fProxy)1; Fail[3] = (fProxy)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void MultiRHSSolveTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
