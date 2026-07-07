using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: arena.fProxyLaplacian1D(n), arena.fProxyLauchli(n,eps), ...

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
            var arena = new Arena(Allocator.Persistent);

            int n = 6, k = 3;

            // Well-conditioned upper- and lower-triangular factors (unit-ish diagonal + small off-diag).
            var U = arena.fProxyMat(n, n);   // zero-init
            var L = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++)
            {
                U[i, i] = (fProxy)(i + 2);
                L[i, i] = (fProxy)(i + 2);
                for (int j = i + 1; j < n; j++) U[i, j] = (fProxy)0.5 / (fProxy)(j - i + 1);
                for (int j = 0; j < i; j++)     L[i, j] = (fProxy)0.5 / (fProxy)(i - j + 1);
            }

            var Xtrue = MakeX(ref arena, n, k);

            // Upper: B = U·Xtrue, solve U X = B, check.
            var Bu = arena.fProxyMat(n, k); Blas.dot(in U, in Xtrue, ref Bu);
            var Xu = Bu.Copy();
            Blas.triUpper(ref U, ref Xu);
            CheckMatClose(in Xu, in Xtrue, Band(in U, (fProxy)200));
            CheckResidual(ref arena, in U, in Xu, in Bu, (fProxy)200);

            // Lower: B = L·Xtrue, solve L X = B, check.
            var Bl = arena.fProxyMat(n, k); Blas.dot(in L, in Xtrue, ref Bl);
            var Xl = Bl.Copy();
            Blas.triLower(ref L, ref Xl);
            CheckMatClose(in Xl, in Xtrue, Band(in L, (fProxy)200));
            CheckResidual(ref arena, in L, in Xl, in Bl, (fProxy)200);

            arena.Dispose();
        }

        // =====================================================================
        // LU — compact decompSolve + solveInPlace, cross-checked vs the vector path.
        // =====================================================================

        void LUMultiRHS()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6, k = 3;
            var A = arena.fProxyLaplacian1D(n);
            var Xtrue = MakeX(ref arena, n, k);
            var B = arena.fProxyMat(n, k); Blas.dot(in A, in Xtrue, ref B);

            // decompInPlace once, decompSolve on the whole block.
            var LUm = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref LUm, ref P);
            var X = B.Copy();
            LU.decompSolve(ref LUm, in P, ref X);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)200));
            CheckResidual(ref arena, in A, in X, in B, (fProxy)200);

            // Column cross-check vs the vector decompSolve (same factor).
            for (int c = 0; c < k; c++)
            {
                var bc = GetCol(ref arena, in B, c);
                LU.decompSolve(ref LUm, in P, ref bc);   // bc := column solution
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)200));
            }

            // solveInPlace (factor+solve fused) matches too.
            var A2 = A.Copy();
            var P2 = new Pivot(n, Allocator.Temp);
            var X2 = B.Copy();
            LU.solveInPlace(ref A2, ref P2, ref X2);
            CheckMatClose(in X2, in X, Band(in A, (fProxy)200));

            P2.Dispose();
            P.Dispose();
            arena.Dispose();
        }

        // =====================================================================
        // CHO — decompSolve + solveInPlace on an SPD system.
        // =====================================================================

        void CHOMultiRHS()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6, k = 3;
            var A = arena.fProxyLaplacian1D(n);
            var Xtrue = MakeX(ref arena, n, k);
            var B = arena.fProxyMat(n, k); Blas.dot(in A, in Xtrue, ref B);

            var L = arena.fProxyMat(n, n);
            AssertTrue(CHO.decomp(in A, ref L));
            var X = B.Copy();
            CHO.decompSolve(ref L, ref X);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)200));
            CheckResidual(ref arena, in A, in X, in B, (fProxy)200);

            for (int c = 0; c < k; c++)
            {
                var bc = GetCol(ref arena, in B, c);
                CHO.decompSolve(ref L, ref bc);
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)200));
            }

            var A2 = A.Copy();
            var X2 = B.Copy();
            AssertTrue(CHO.solveInPlace(ref A2, ref X2));
            CheckMatClose(in X2, in X, Band(in A, (fProxy)200));

            arena.Dispose();
        }

        // =====================================================================
        // QR — decompSolve (reuse Q,R) + fused solveInPlace, square and tall LS.
        // =====================================================================

        void QRMultiRHSSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6, k = 3;
            var A = arena.fProxyLaplacian1D(n);
            var Xtrue = MakeX(ref arena, n, k);
            var B = arena.fProxyMat(n, k); Blas.dot(in A, in Xtrue, ref B);

            // decompSolve from a precomputed QR (B preserved).
            var Q = A.Copy();
            var R = arena.fProxyMat(n, n);
            QR.decompInPlace(ref Q, ref R);
            var Xd = arena.fProxyMat(n, k);
            QR.decompSolve(ref Q, ref R, ref B, ref Xd);
            CheckMatClose(in Xd, in Xtrue, Band(in A, (fProxy)300));
            CheckResidual(ref arena, in A, in Xd, in B, (fProxy)300);

            // fused solveInPlace (destroys A and B) matches, and matches the vector path per column.
            var Aw = A.Copy();
            var Bw = B.Copy();
            var Xs = arena.fProxyMat(n, k);
            QR.solveInPlace(ref Aw, ref Bw, ref Xs);
            CheckMatClose(in Xs, in Xd, Band(in A, (fProxy)300));

            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(n);
                QR.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in Xd, c, in xc, Band(in A, (fProxy)300));
            }

            arena.Dispose();
        }

        void QRMultiRHSTall()
        {
            var arena = new Arena(Allocator.Persistent);

            int nc = 4, k = 3;
            var A = arena.fProxyLauchli(nc, (fProxy)0.5);   // (nc+1) x nc, tall full-column-rank
            int m = A.M_Rows;
            var Xtrue = MakeX(ref arena, nc, k);
            var B = arena.fProxyMat(m, k); Blas.dot(in A, in Xtrue, ref B);   // consistent, in range(A)

            var Aw = A.Copy();
            var Bw = B.Copy();
            var X = arena.fProxyMat(nc, k);
            QR.solveInPlace(ref Aw, ref Bw, ref X);

            // consistent LS ⇒ solution is exactly Xtrue, residual ≈ 0.
            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)100));
            CheckResidual(ref arena, in A, in X, in B, (fProxy)100);

            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(nc);
                QR.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)100));
            }

            arena.Dispose();
        }

        // =====================================================================
        // QRCP — rank-safe multi-RHS: full rank + rank-deficient.
        // =====================================================================

        void QRCPMultiRHS()
        {
            var arena = new Arena(Allocator.Persistent);

            int nc = 4, k = 3;
            var A = arena.fProxyLauchli(nc, (fProxy)0.5);   // tall full rank
            int m = A.M_Rows;
            var Xtrue = MakeX(ref arena, nc, k);
            var B = arena.fProxyMat(m, k); Blas.dot(in A, in Xtrue, ref B);

            var Aq = A.Copy();
            var X = arena.fProxyMat(nc, k);
            RankInfo info = QRCP.solveInPlace(ref Aq, ref B, ref X);   // A_to_Q -> Q, B preserved
            AssertTrue(info.Solved);
            RecordEq(info.rank, nc);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)300));
            CheckResidual(ref arena, in A, in X, in B, (fProxy)300);

            // Cross-check vs the vector fused solveInPlace per column (destroys its own copies).
            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(nc);
                QRCP.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)300));
            }

            arena.Dispose();
        }

        void QRCPMultiRHSRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4, k = 2;
            var A = arena.fProxyPei(n, (fProxy)0);   // all-ones, rank 1
            var B = MakeX(ref arena, n, k);           // arbitrary RHS

            var Aq = A.Copy();
            var X = arena.fProxyMat(n, k);
            RankInfo info = QRCP.solveInPlace(ref Aq, ref B, ref X);
            RecordEq(info.rank, 1);

            // Cross-check vs the vector path per column — the truncated (basic) solution must agree.
            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(n);
                QRCP.solveInPlace(ref Ac, ref bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)300));
            }

            arena.Dispose();
        }

        // =====================================================================
        // LQ — minimum-norm multi-RHS on a wide, full-row-rank system.
        // =====================================================================

        void LQMultiRHSMinNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            int k = 3;
            var Tall = arena.fProxyLauchli(4, (fProxy)0.5);   // 5 x 4
            var A = arena.fProxyMat(Tall.N_Cols, Tall.M_Rows); // 4 x 5, wide full row rank
            Blas.trans(in Tall, ref A);
            int m = A.M_Rows;   // 4
            int n = A.N_Cols;   // 5

            var B = MakeX(ref arena, m, k);   // any RHS (A full row rank ⇒ consistent)

            var X = arena.fProxyMat(n, k);
            LQ.minNormSolve(ref A, ref B, ref X);

            CheckResidual(ref arena, in A, in X, in B, (fProxy)200);   // A·X ≈ B

            for (int c = 0; c < k; c++)
            {
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(n);
                LQ.minNormSolve(ref A, ref bc, ref xc);   // A not modified
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)200));
            }

            arena.Dispose();
        }

        // =====================================================================
        // CHOP — pivoted Cholesky multi-RHS: full rank + rank-deficient (min-norm).
        // =====================================================================

        void CHOPMultiRHSFullRank()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6, k = 3;
            var A = arena.fProxyLaplacian1D(n);   // SPD full rank
            var Xtrue = MakeX(ref arena, n, k);
            var B = arena.fProxyMat(n, k); Blas.dot(in A, in Xtrue, ref B);

            var Aw = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            var X = B.Copy();
            RankInfo info = CHOP.solveInPlace(ref Aw, ref P, ref X);
            AssertTrue(info.Solved);
            RecordEq(info.rank, n);

            CheckMatClose(in X, in Xtrue, Band(in A, (fProxy)300));
            CheckResidual(ref arena, in A, in X, in B, (fProxy)300);

            // Cross-check vs the vector solveInPlace per column.
            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var Pc = new Pivot(n, Allocator.Temp);
                var bc = GetCol(ref arena, in B, c);
                CHOP.solveInPlace(ref Ac, ref Pc, ref bc);
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)300));
                Pc.Dispose();
            }

            P.Dispose();
            arena.Dispose();
        }

        void CHOPMultiRHSRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4, k = 2;
            var A = arena.fProxyPei(n, (fProxy)0);   // all-ones PSD, rank 1
            var B = MakeX(ref arena, n, k);

            var Aw = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            var X = B.Copy();
            RankInfo info = CHOP.solveInPlace(ref Aw, ref P, ref X);
            RecordEq(info.rank, 1);

            // Cross-check vs the vector min-norm path per column.
            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var Pc = new Pivot(n, Allocator.Temp);
                var bc = GetCol(ref arena, in B, c);
                CHOP.solveInPlace(ref Ac, ref Pc, ref bc);
                CheckColClose(in X, c, in bc, Band(in A, (fProxy)300));
                Pc.Dispose();
            }

            P.Dispose();
            arena.Dispose();
        }

        // =====================================================================
        // SVD — minimum-norm least-squares multi-RHS: tall and wide.
        // =====================================================================

        void SVDMultiRHSTall()
        {
            var arena = new Arena(Allocator.Persistent);

            int nc = 4, k = 3;
            var A = arena.fProxyLauchli(nc, (fProxy)1E-2);   // (nc+1) x nc tall
            int m = A.M_Rows;
            var Xtrue = MakeX(ref arena, nc, k);
            var B = arena.fProxyMat(m, k); Blas.dot(in A, in Xtrue, ref B);

            var Aw = A.Copy();
            var X = arena.fProxyMat(nc, k);
            RankInfo info = SVD.pinvSolve(ref Aw, in B, ref X);
            AssertTrue(info.Solved);

            CheckResidual(ref arena, in A, in X, in B, (fProxy)500);

            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(nc);
                SVD.pinvSolve(ref Ac, in bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)500));
            }

            arena.Dispose();
        }

        void SVDMultiRHSWide()
        {
            var arena = new Arena(Allocator.Persistent);

            int k = 3;
            var Tall = arena.fProxyLauchli(4, (fProxy)0.5);   // 5 x 4
            var A = arena.fProxyMat(Tall.N_Cols, Tall.M_Rows); // 4 x 5 wide
            Blas.trans(in Tall, ref A);
            int m = A.M_Rows;
            int n = A.N_Cols;

            var B = MakeX(ref arena, m, k);

            var Aw = A.Copy();
            var X = arena.fProxyMat(n, k);
            RankInfo info = SVD.pinvSolve(ref Aw, in B, ref X);
            AssertTrue(info.Solved);

            CheckResidual(ref arena, in A, in X, in B, (fProxy)500);   // consistent underdetermined

            for (int c = 0; c < k; c++)
            {
                var Ac = A.Copy();
                var bc = GetCol(ref arena, in B, c);
                var xc = arena.fProxyVec(n);
                SVD.pinvSolve(ref Ac, in bc, ref xc);
                CheckColClose(in X, c, in xc, Band(in A, (fProxy)500));
            }

            arena.Dispose();
        }

        // =====================================================================
        // helpers
        // =====================================================================

        fProxyMxN MakeX(ref Arena arena, int n, int k)
        {
            var X = arena.fProxyMat(n, k);
            for (int i = 0; i < n; i++)
                for (int c = 0; c < k; c++)
                    X[i, c] = (fProxy)(1 + i - (fProxy)0.5 * c + (fProxy)0.25 * (c + 1) * (i % 3));
            return X;
        }

        fProxyN GetCol(ref Arena arena, in fProxyMxN M, int c)
        {
            int n = M.M_Rows;
            var v = arena.fProxyVec(n);
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
        void CheckResidual(ref Arena arena, in fProxyMxN A, in fProxyMxN X, in fProxyMxN B, fProxy factor)
        {
            var AX = arena.fProxyMat(A.M_Rows, X.N_Cols);
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
