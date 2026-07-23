using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov.craig — single-RHS CRAIG (Craig 1955 / Paige-Saunders BIT 1995): the least-NORM solver
// for underdetermined consistent systems A x = b (A is m×n, m<=n, full row rank). Among all x with
// A x = b it returns the minimum-‖x‖ one, x* = Aᵀ(AAᵀ)⁻¹b, via Golub-Kahan bidiagonalization.
//
// Oracle for the min-norm value: LQ.minNormSolve (exact x* via LQ factorization).
public class fProxyCRAIGTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            RectangularMinNorm,
            SquareFullRank,
            ExplicitScratchInJob,
            ZeroRhs,
            RankDeficientBreakdown,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RectangularMinNorm:     RectangularMinNorm();     break;
                case TestType.SquareFullRank:         SquareFullRank();         break;
                case TestType.ExplicitScratchInJob:   ExplicitScratchInJob();   break;
                case TestType.ZeroRhs:                ZeroRhs();                break;
                case TestType.RankDeficientBreakdown: RankDeficientBreakdown(); break;
            }
        }

        // Comparison tolerance for craig vs LQ oracle / Ax≈b, scaled per numeric type. CRAIG's
        // bidiagonalization is finite (m steps in exact arithmetic) and the diagonal-boosted systems
        // below are well-conditioned, so this stays reasonably tight; float carries the looser value.
        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        // Convergence tolerance handed to craig — tighter than the default Consts.fProxySqrtEps
        // (~3.4e-4 for float), which would only drive ‖b-Ax‖ down to ~1e-2 absolute here and never
        // satisfy Tol(). Loose enough that a well-conditioned full-row-rank system still converges
        // within its m bidiagonalization steps (default maxIter = A.M_Rows), so no MaxIterations.
        static fProxy SolveTol() => /*+choose[1e-5f|1e-13]*/1e-5f/*-choose*/;

        // Full-(row-)rank test matrix: random with a diagonal boost (mirrors LQMinNormInPlaceTests).
        static fProxyMxN BuildA(int m, int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, seed, Allocator.Temp);
            for (int d = 0; d < m; d++)
                A[d, d] += (fProxy)10;
            return A;
        }

        static fProxy Norm(in fProxyN v) => math.sqrt(Blas.dot(v, v));

        // ---- KEY TEST: least-norm correctness on an underdetermined consistent system. ----
        // Ax=b alone is satisfied by any solution; the mandatory assertion is x ≈ the LQ min-norm
        // oracle xRef (and x is verifiably NOT the arbitrary x_true used to build b).
        void RectangularMinNorm()
        {
            int m = 5, n = 9;
            var A = BuildA(m, n, 51001);

            // Arbitrary true solution; b = A x_true makes the system consistent. x_true is generally
            // NOT in row(A), so the min-norm solution differs from it.
            var xTrue = GenerateOP.fProxyRandomVec(n, -5f, 5f, 51002, Allocator.Temp);
            var b = new fProxyN(m, Allocator.Temp);
            Blas.dot(in A, in xTrue, ref b);

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.craig(in A, in b, ref x, A.M_Rows, SolveTol());
            Assert.IsTrue(info.Solved);

            // (a) Ax ≈ b — necessary but not sufficient.
            var Ax = new fProxyN(m, Allocator.Temp);
            Blas.dot(in A, in x, ref Ax);
            var bMinusAx = new fProxyN(in b, Allocator.Temp);
            fProxyComp.subInPlace(bMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bMinusAx, Tol()));

            // (b) THE POINT: x matches the exact min-2-norm solution from the LQ oracle.
            var xRef = new fProxyN(n, Allocator.Temp);
            LQ.minNormSolve(in A, in b, ref xRef);
            var xRefMinusX = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(xRefMinusX, x);
            Assert.IsTrue(Analysis.isZero(xRefMinusX, Tol()));

            // (b, softer) ‖x‖ <= ‖x_true‖ (x is minimal among all solutions incl. x_true).
            fProxy nx = Norm(in x);
            fProxy nxTrue = Norm(in xTrue);
            Assert.IsTrue(nx <= nxTrue + Tol());

            // (b, negative guard) craig did NOT merely echo x_true or some arbitrary solution:
            // x_true is a solution but not the min-norm one, so x != x_true at any sane tolerance.
            var xTrueMinusX = new fProxyN(in xTrue, Allocator.Temp);
            fProxyComp.subInPlace(xTrueMinusX, x);
            Assert.IsFalse(Analysis.isZero(xTrueMinusX, (fProxy)0.1));
        }

        // ---- Square full-rank: the system has a UNIQUE solution, so craig must recover x_true. ----
        void SquareFullRank()
        {
            int nn = 7;
            var A = BuildA(nn, nn, 52001);

            var xTrue = GenerateOP.fProxyRandomVec(nn, -5f, 5f, 52002, Allocator.Temp);
            var b = new fProxyN(nn, Allocator.Temp);
            Blas.dot(in A, in xTrue, ref b);

            var x = new fProxyN(nn, Allocator.Temp);
            var info = Krylov.craig(in A, in b, ref x, A.M_Rows, SolveTol());
            Assert.IsTrue(info.Solved);

            var Ax = new fProxyN(nn, Allocator.Temp);
            Blas.dot(in A, in x, ref Ax);
            var bMinusAx = new fProxyN(in b, Allocator.Temp);
            fProxyComp.subInPlace(bMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bMinusAx, Tol()));

            // Unique solution: x IS x_true (direct comparison, unlike the rectangular case).
            var xTrueMinusX = new fProxyN(in xTrue, Allocator.Temp);
            fProxyComp.subInPlace(xTrueMinusX, x);
            Assert.IsTrue(Analysis.isZero(xTrueMinusX, Tol()));
        }

        // ---- Explicit-scratch overload driven through the IJob struct: exercises the
        // caller-provided u/v/tmpM/tmpN buffer path (guards against IJob struct-copy resets of the
        // solver's internal ping-pong buffers) on the rectangular min-norm case. ----
        void ExplicitScratchInJob()
        {
            int m = 6, n = 11;
            var A = BuildA(m, n, 53001);

            var xTrue = GenerateOP.fProxyRandomVec(n, -4f, 4f, 53002, Allocator.Temp);
            var b = new fProxyN(m, Allocator.Temp);
            Blas.dot(in A, in xTrue, ref b);

            // Caller-provided scratch (lengths: u,tmpM = Rows; v,tmpN = Cols).
            var u    = new fProxyN(m, Allocator.Temp);
            var v    = new fProxyN(n, Allocator.Temp);
            var tmpM = new fProxyN(m, Allocator.Temp);
            var tmpN = new fProxyN(n, Allocator.Temp);
            var x    = new fProxyN(n, Allocator.Temp);

            var info = Krylov.craig(in A, in b, ref x, ref u, ref v, ref tmpM, ref tmpN, A.M_Rows, SolveTol());
            Assert.IsTrue(info.Solved);

            var Ax = new fProxyN(m, Allocator.Temp);
            Blas.dot(in A, in x, ref Ax);
            var bMinusAx = new fProxyN(in b, Allocator.Temp);
            fProxyComp.subInPlace(bMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bMinusAx, Tol()));

            var xRef = new fProxyN(n, Allocator.Temp);
            LQ.minNormSolve(in A, in b, ref xRef);
            var xRefMinusX = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(xRefMinusX, x);
            Assert.IsTrue(Analysis.isZero(xRefMinusX, Tol()));
        }

        // ---- Zero RHS: min-norm solution is exactly x = 0, returned on the early-out path with
        // zero iterations. Assertions are EXACT, not approximate. ----
        void ZeroRhs()
        {
            int m = 5, n = 9;
            var A = BuildA(m, n, 54001);
            var b = new fProxyN(m, Allocator.Temp); // all zeros

            // Seed x with garbage to prove craig zeroes it internally (no warm start).
            var x = new fProxyN(n, Allocator.Temp);
            for (int j = 0; j < n; j++) x[j] = (fProxy)7;

            var info = Krylov.craig(in A, in b, ref x);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(info.iterations == 0);
            Assert.IsTrue(Analysis.isZero(x, (fProxy)0));
        }

        // ---- BONUS: rank-deficient A with b ∉ range(A) -> the very first Aᵀu step collapses
        // (alfa == 0), the documented "v collapsed on the first step" branch. Constructed bit-exactly
        // (row 1 all zeros, b nonzero ONLY in that row's component) so u1 weights only the zero row
        // and Aᵀu1 == 0 EXACTLY in both float and double -- craig must report Breakdown (never
        // Converged, never NaN). x is UNDEFINED per the Breakdown contract, so it is NOT asserted. ----
        void RankDeficientBreakdown()
        {
            int m = 2, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 55001, Allocator.Temp);
            for (int j = 0; j < n; j++)
                A[1, j] = (fProxy)0; // row 1 = 0 -> rank-deficient (not full row rank)

            // b = e_2: nonzero only where A's row is zero, so b is orthogonal to range(A) and
            // u1 = b/‖b‖ makes Aᵀu1 = 0 exactly on the first bidiagonalization step.
            var b = new fProxyN(m, Allocator.Temp);
            b[1] = (fProxy)1;

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.craig(in A, in b, ref x);

            Assert.IsTrue(info.status == IterativeSolveStatus.Breakdown);
            // Norms are finite (no NaN escapes the collapse path).
            Assert.IsFalse(double.IsNaN(info.rnorm));
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void Test(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }
}
