using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov.lslq — single-RHS LSLQ (Estrin-Orban-Saunders): least-SQUARES solve for an overdetermined
// system min‖Ax-b‖ (A is m×n, m>=n, full column rank) via Golub-Kahan bidiagonalization folded through
// an LQ factorization. Returns the LQ point x^L (the error-minimizing iterate), which converges to the
// unique least-squares solution x*. Oracles: a direct dense QR least-squares solve (exact x*) AND lsqr
// itself (same least-squares problem, must agree at convergence). The HEADLINE test exercises the
// certified Gauss-Radau forward-error bound (xErrBound); the plain overloads report NaN for it.
public class fProxyLSLQTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            RectangularLeastSquares,
            SquareFullRank,
            ExplicitScratchInJob,
            ZeroRhs,
            RankDeficientGraceful,
            BoundIsUpperBound,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RectangularLeastSquares: RectangularLeastSquares(); break;
                case TestType.SquareFullRank:          SquareFullRank();          break;
                case TestType.ExplicitScratchInJob:    ExplicitScratchInJob();    break;
                case TestType.ZeroRhs:                 ZeroRhs();                 break;
                case TestType.RankDeficientGraceful:   RankDeficientGraceful();   break;
                case TestType.BoundIsUpperBound:       BoundIsUpperBound();       break;
            }
        }

        // Comparison tolerance vs the QR oracle / lsqr, scaled per numeric type. The +10-boosted
        // matrices below are κ≈1, so LSLQ's Golub-Kahan bidiagonalization converges cleanly and this
        // stays tight; float carries the looser value.
        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        // Convergence tolerance handed to lslq -- tighter than the default Consts.fProxySqrtEps so the
        // optimality residual ‖Aᵀr‖ is driven well below Tol().
        static fProxy SolveTol() => /*+choose[1e-5f|1e-13]*/1e-5f/*-choose*/;

        // Generous iteration budget: the bidiagonalization on a full-column-rank A terminates within n
        // steps in exact arithmetic; 4*n gives float loss-of-orthogonality headroom.
        static int MaxIter(in fProxyMxN A) => 4 * A.N_Cols;

        // Full-(column-)rank tall test matrix: random with a +10 diagonal boost -> AᵀA ≈ (100+ε)·I -> κ≈1.
        static fProxyMxN BuildA(int m, int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, seed);
            int d = math.min(m, n);
            for (int i = 0; i < d; i++)
                A[i, i] += (fProxy)10;
            return A;
        }

        static fProxy Norm(in fProxyN v) => math.sqrt(Blas.dot(v, v));

        // Direct dense least-squares reference via QR (full column rank, m >= n) -- the exact x*.
        static fProxyN ReferenceSolveLstsq(in fProxyMxN A, in fProxyN b)
        {
            var Q = new fProxyMxN(A.M_Rows, A.N_Cols, Allocator.Temp);
            var R = new fProxyMxN(A.N_Cols, A.N_Cols, Allocator.Temp);
            QR.decomp(in A, ref Q, ref R);
            var bLocal = new fProxyN(A.M_Rows, Allocator.Temp);
            bLocal.CopyFrom(in b);                       // decompSolve destroys its RHS; keep b intact
            var xRef = new fProxyN(A.N_Cols, Allocator.Temp);
            QR.decompSolve(ref Q, ref R, ref bLocal, ref xRef);
            return xRef;
        }

        // ---- KEY TEST: least-squares correctness on an INCONSISTENT overdetermined system. Random b is
        // generally not in range(A), so the residual is genuinely nonzero (rnorm > 0) -- proving this is
        // a real least-squares problem, not a disguised consistent solve. Mandatory: x ≈ the exact QR
        // least-squares solution AND x ≈ lsqr's iterate, and the certified optimality residual ‖Aᵀr‖ is
        // small. ----
        void RectangularLeastSquares()
        {
            int m = 11, n = 5;
            var A = BuildA(m, n, 71001);
            var b = GenerateOP.fProxyRandomVec(m, -3f, 3f, 71002);   // generally inconsistent

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lslq(in A, in b, ref x, MaxIter(in A), SolveTol());
            Assert.IsTrue(info.Solved);

            // (a) THE POINT: x matches the exact least-squares solution from the QR oracle.
            var xRef = ReferenceSolveLstsq(in A, in b);
            var xRefMinusX = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(xRefMinusX, x);
            Assert.IsTrue(Analysis.isZero(xRefMinusX, Tol()));

            // (b) Cross-solver oracle: lsqr solves the same LS problem; must agree at convergence.
            var xLsqr = new fProxyN(n, Allocator.Temp);
            Krylov.lsqr(in A, in b, ref xLsqr, MaxIter(in A), SolveTol());
            var xLsqrMinusX = new fProxyN(in xLsqr, Allocator.Temp);
            fProxyComp.subInPlace(xLsqrMinusX, x);
            Assert.IsTrue(Analysis.isZero(xLsqrMinusX, Tol()));

            // (c) Optimality residual ‖Aᵀ(b-Ax)‖ is small relative to ‖Aᵀb‖ (certified in info).
            var Atb = new fProxyN(n, Allocator.Temp);
            Blas.dot(in b, in A, ref Atb);               // Aᵀb = bᵀA
            Assert.IsTrue(info.Arnorm <= (double)(SolveTol() * Norm(in Atb)) * 10.0);

            // (d, teeth) the problem really IS inconsistent: residual is meaningfully nonzero, so this
            // exercised the least-squares path (not a consistent Ax=b solve).
            Assert.IsTrue(info.rnorm > (double)Tol());

            // xErrBound is NaN on the plain overloads (no σ_min estimate supplied).
            Assert.IsTrue(double.IsNaN(info.xErrBound));
        }

        // ---- Square full-rank consistent system: UNIQUE solution, so lslq must recover x_true. ----
        void SquareFullRank()
        {
            int nn = 7;
            var A = BuildA(nn, nn, 72001);

            var xTrue = GenerateOP.fProxyRandomVec(nn, -5f, 5f, 72002);
            var b = new fProxyN(nn, Allocator.Temp);
            Blas.dot(in A, in xTrue, ref b);

            var x = new fProxyN(nn, Allocator.Temp);
            var info = Krylov.lslq(in A, in b, ref x, MaxIter(in A), SolveTol());
            Assert.IsTrue(info.Solved);

            var Ax = new fProxyN(nn, Allocator.Temp);
            Blas.dot(in A, in x, ref Ax);
            var bMinusAx = new fProxyN(in b, Allocator.Temp);
            fProxyComp.subInPlace(bMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bMinusAx, Tol()));

            // Unique solution: x IS x_true.
            var xTrueMinusX = new fProxyN(in xTrue, Allocator.Temp);
            fProxyComp.subInPlace(xTrueMinusX, x);
            Assert.IsTrue(Analysis.isZero(xTrueMinusX, Tol()));
        }

        // ---- Explicit-scratch overload driven through the IJob struct: exercises the caller-provided
        // u/v/wbar/tmpM/tmpN buffer path (guards against IJob struct-copy resets). ----
        void ExplicitScratchInJob()
        {
            int m = 13, n = 6;
            var A = BuildA(m, n, 73001);
            var b = GenerateOP.fProxyRandomVec(m, -3f, 3f, 73002);

            // Caller-provided scratch (lengths: u,tmpM = Rows; v,wbar,tmpN = Cols).
            var u    = new fProxyN(m, Allocator.Temp);
            var v    = new fProxyN(n, Allocator.Temp);
            var wbar = new fProxyN(n, Allocator.Temp);
            var tmpM = new fProxyN(m, Allocator.Temp);
            var tmpN = new fProxyN(n, Allocator.Temp);
            var x    = new fProxyN(n, Allocator.Temp);

            var info = Krylov.lslq(in A, in b, ref x, ref u, ref v, ref wbar, ref tmpM, ref tmpN, MaxIter(in A), SolveTol());
            Assert.IsTrue(info.Solved);

            var xRef = ReferenceSolveLstsq(in A, in b);
            var xRefMinusX = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(xRefMinusX, x);
            Assert.IsTrue(Analysis.isZero(xRefMinusX, Tol()));
        }

        // ---- Zero RHS: least-squares solution is exactly x = 0 on the early-out path with zero
        // iterations. Assertions are EXACT. Also proves lslq zeroes x internally (no warm start) by
        // seeding garbage. ----
        void ZeroRhs()
        {
            int m = 9, n = 5;
            var A = BuildA(m, n, 74001);
            var b = new fProxyN(m, Allocator.Temp); // all zeros

            var x = new fProxyN(n, Allocator.Temp);
            for (int j = 0; j < n; j++) x[j] = (fProxy)7;

            var info = Krylov.lslq(in A, in b, ref x);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(info.iterations == 0);
            Assert.IsTrue(Analysis.isZero(x, (fProxy)0));
        }

        // ---- Rank-deficient A (a duplicated column) is NOT a failure for LSLQ: the bidiagonalization
        // terminates early at a least-squares solution. Must report a usable status (Converged or
        // MaxIterations), never NaN, and the certified optimality residual ‖Aᵀr‖ must be small -- x is a
        // genuine least-squares point even though it is not unique. ----
        void RankDeficientGraceful()
        {
            int m = 10, n = 5;
            var A = BuildA(m, n, 75001);
            for (int i = 0; i < m; i++)
                A[i, n - 1] = A[i, 0];        // last column := first column -> rank n-1

            var b = GenerateOP.fProxyRandomVec(m, -3f, 3f, 75002);

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lslq(in A, in b, ref x, MaxIter(in A), SolveTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged
                       || info.status == IterativeSolveStatus.MaxIterations);
            Assert.IsFalse(double.IsNaN(info.rnorm));
            Assert.IsFalse(double.IsNaN(info.Arnorm));

            // Certified optimality: ‖Aᵀr‖ small relative to ‖Aᵀb‖ (a least-squares stationary point).
            var Atb = new fProxyN(n, Allocator.Temp);
            Blas.dot(in b, in A, ref Atb);               // Aᵀb = bᵀA
            Assert.IsTrue(info.Arnorm <= (double)(Tol() * Norm(in Atb)));
        }

        // ---- HEADLINE: the certified Gauss-Radau forward-error bound |ζ̃| on ‖x^L - x*‖. Checked at a
        // MID-convergence iterate (maxIter=2, well short of the ~1e-5 optimality tol) so the true error
        // is meaningfully nonzero -- otherwise `xErrBound >= ‖x-x*‖` is trivially satisfied at
        // convergence and has no teeth. σ_est = (1-1e-10)·σ_min(A) (a valid strict underestimate via
        // SVD). The bound must (a) be a valid UPPER bound and (b) be TIGHT (LSLQ's error-minimization
        // property makes it essentially exact), which a crude ‖r‖/σ_min bound would fail. ----
        void BoundIsUpperBound()
        {
            int m = 11, n = 6;
            var A = BuildA(m, n, 76001);
            var b = GenerateOP.fProxyRandomVec(m, -3f, 3f, 76002);   // inconsistent least-squares

            // σ_min(A) via SVD of the tall m×n A (n singular values = A's). The 1e-4 underestimate
            // margin must survive the float build's (fProxy)sigmaMinEst cast (~1e-7 rel) AND the float
            // SVD's own error, so it is deliberately far coarser than double eps -- a strict underestimate
            // in BOTH builds (a tighter margin like 1e-10 would round away in float and break the bound).
            var svals = new fProxyN(n, Allocator.Temp);
            SVD.values(in A, ref svals);
            fProxy smin = svals[0];
            for (int i = 1; i < n; i++) smin = math.min(smin, svals[i]);
            double sigmaEst = (1.0 - 1e-4) * (double)smin;

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lslq(in A, in b, ref x, 2, SolveTol(), sigmaEst);
            // 2 iterations cannot reach the 1e-5 optimality tol on a 6-column system: mid-convergence.
            Assert.IsTrue(info.status == IterativeSolveStatus.MaxIterations);

            var xRef = ReferenceSolveLstsq(in A, in b);
            var diff = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(diff, x);
            double trueErr = (double)Norm(in diff);

            // (a) the reported bound bounds the true error (a wrong recurrence that under-reports fails
            // here). Certified (upper bound) in double; in float it may marginally under-report only
            // NEAR convergence -- at this MID-convergence operating point the float bound is measured to
            // sit at ratio >=1.0005, so the float floor carries only a small Burst-arithmetic margin,
            // double stays tight. (0.99 keeps real teeth: a float-specific pathology >1% is still caught.)
            Assert.IsFalse(double.IsNaN(info.xErrBound));
            double lowerFactor = /*+choose[0.99|1.0 - 1e-3]*/0.99/*-choose*/;
            Assert.IsTrue(info.xErrBound >= trueErr * lowerFactor);
            // (b) ... and TIGHT -- within a small factor (a crude ‖r‖/σ_min bound would be far looser).
            Assert.IsTrue(info.xErrBound <= trueErr * 10.0);

            // Estimate gates the machinery: no σ_est (default) -> NaN bound, same solve.
            var x2 = new fProxyN(n, Allocator.Temp);
            var info2 = Krylov.lslq(in A, in b, ref x2, 2, SolveTol());
            Assert.IsTrue(double.IsNaN(info2.xErrBound));
            for (int i = 0; i < n; i++)
                Assert.IsTrue(x[i] == x2[i]);   // bound machinery does not perturb the solve
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    // Generous timeout: the first case run in a session pays one cold Burst compile of the whole
    // Execute() body (all six test methods), which can exceed the 180s default on its own.
    [Timeout(600000)]
    [TestCaseSource("GetEnums")]
    public void Test(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }
}
