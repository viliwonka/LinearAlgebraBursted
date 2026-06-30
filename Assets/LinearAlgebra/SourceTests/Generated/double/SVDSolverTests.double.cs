using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class doubleSVDSolverTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            PinvSquareFullRank,
            PinvOverdeterminedFullRank,
            PinvOverdeterminedResidual,
            PinvRankDeficientMinNorm,
            PinvUnderdetermined,
            PinvZeroMatrix,
            PseudoInverseSquareInvertible,
            PseudoInverseDiag,
            PseudoInverseMoorePenrose
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.PinvSquareFullRank:
                    PinvSquareFullRank();
                break;
                case TestType.PinvOverdeterminedFullRank:
                    PinvOverdeterminedFullRank();
                break;
                case TestType.PinvOverdeterminedResidual:
                    PinvOverdeterminedResidual();
                break;
                case TestType.PinvRankDeficientMinNorm:
                    PinvRankDeficientMinNorm();
                break;
                case TestType.PinvUnderdetermined:
                    PinvUnderdetermined();
                break;
                case TestType.PinvZeroMatrix:
                    PinvZeroMatrix();
                break;
                case TestType.PseudoInverseSquareInvertible:
                    PseudoInverseSquareInvertible();
                break;
                case TestType.PseudoInverseDiag:
                    PseudoInverseDiag();
                break;
                case TestType.PseudoInverseMoorePenrose:
                    PseudoInverseMoorePenrose();
                break;
            }
        }

        // Case 1: Square full-rank. Well-conditioned 8x8 (diagonal-boosted), known x_orig,
        // b = A*x_orig. pinvSolve no longer modifies A; copies are kept for clarity and residual
        // checks use the saved copy.
        public void PinvSquareFullRank()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var A = arena.doubleRandomMatrix(dim, dim, -5f, 5f, 314221);
            // boost the diagonal to ensure good conditioning (see OrthoOpTests / SolversTests)
            for (int d = 0; d < dim; d++)
                A[d, d] += (double)10f;

            var A_copy = A.Copy();

            var xOrig = arena.doubleRandomVector(dim, -3f, 3f, 1337);
            var b = double_OP.dot(A_copy, xOrig);

            var x = arena.doubleVec(dim);

            int rank = SVD.pinvSolve(ref A, in b, ref x, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(8, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in x));

            for (int k = 0; k < dim; k++)
                AssertClose(x[k], xOrig[k], 1E-3f);

            arena.Dispose();
        }

        // Case 2a: Overdetermined full column rank, b exactly in range(A). x recovers x_orig.
        public void PinvOverdeterminedFullRank()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12;
            int n = 4;

            var A = arena.doubleRandomMatrix(m, n, -5f, 5f, 778231);
            // boost the leading diagonal block to ensure full column rank / conditioning
            for (int d = 0; d < n; d++)
                A[d, d] += (double)10f;

            var A_copy = A.Copy();

            var xOrig = arena.doubleRandomVector(n, -3f, 3f, 4242);
            var b = double_OP.dot(A_copy, xOrig);

            var x = arena.doubleVec(n);

            int rank = SVD.pinvSolve(ref A, in b, ref x, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(4, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in x));

            for (int k = 0; k < n; k++)
                AssertClose(x[k], xOrig[k], 1E-3f);

            arena.Dispose();
        }

        // Case 2b: Overdetermined, b has a component outside range(A). Least-squares solution
        // must satisfy the normal equations: A^T (A x - b) = 0. Check ||A^T r||_inf < 1e-2.
        public void PinvOverdeterminedResidual()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12;
            int n = 4;

            var A = arena.doubleRandomMatrix(m, n, -5f, 5f, 778231);
            for (int d = 0; d < n; d++)
                A[d, d] += (double)10f;

            var A_copy = A.Copy();

            // b is a generic random vector in R^m, almost surely not in range(A)
            var b = arena.doubleRandomVector(m, -5f, 5f, 9090);
            var b_copy = b.Copy();

            var x = arena.doubleVec(n);

            int rank = SVD.pinvSolve(ref A, in b, ref x, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(4, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in x));

            // residual r = A x - b   (length m), using the saved copy of A
            var Ax = double_OP.dot(A_copy, x);
            doubleN r = arena.doubleVec(m);
            for (int i = 0; i < m; i++)
                r[i] = Ax[i] - b_copy[i];

            // A^T r  (length n): vecMatDot computes r^T A = A^T r
            var Atr = double_OP.dot(r, A_copy);

            double maxAbs = (double)0f;
            for (int k = 0; k < n; k++)
                maxAbs = Unity.Mathematics.math.max(maxAbs, Unity.Mathematics.math.abs(Atr[k]));

            // looser tolerance: errors get squared through A^T A
            // Fail layout: [1]=maxAbs, [2]=limit 1E-2, [3]=diff
            if (!(maxAbs < (double)1E-2f) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = maxAbs;
                Fail[2] = (double)1E-2f;
                Fail[3] = maxAbs - (double)1E-2f;
            }
            Assert.IsTrue(maxAbs < (double)1E-2f);

            arena.Dispose();
        }

        // Case 3: Rank-deficient minimum-norm. A = 4x2, both columns = (1,1,1,1)^T,
        // b = (1,1,1,1)^T -> rank 1. A x = (x0+x1) * ones, so any x0+x1 = 1 solves exactly;
        // the minimum-norm solution is x = (0.5, 0.5).
        public void PinvRankDeficientMinNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4;
            int n = 2;

            var A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
            {
                A[i, 0] = (double)1f;
                A[i, 1] = (double)1f;
            }

            var b = arena.doubleVec(m);
            for (int i = 0; i < m; i++)
                b[i] = (double)1f;

            var x = arena.doubleVec(n);

            int rank = SVD.pinvSolve(ref A, in b, ref x, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(1, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in x));

            AssertClose(x[0], (double)0.5f, 1E-4f);
            AssertClose(x[1], (double)0.5f, 1E-4f);

            arena.Dispose();
        }

        // Case 4: Underdetermined (m < n branch). A = [[1,0,0],[0,1,0]] (2x3), b = (2,3)
        // -> minimum-norm x = (2,3,0), rank 2.
        public void PinvUnderdetermined()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 2;
            int n = 3;

            var A = arena.doubleMat(m, n);
            A[0, 0] = (double)1f;
            A[1, 1] = (double)1f;

            var b = arena.doubleVec(m);
            b[0] = (double)2f;
            b[1] = (double)3f;

            var x = arena.doubleVec(n);

            int rank = SVD.pinvSolve(ref A, in b, ref x, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(2, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in x));

            AssertClose(x[0], (double)2f, 1E-4f);
            AssertClose(x[1], (double)3f, 1E-4f);
            AssertClose(x[2], (double)0f, 1E-4f);

            arena.Dispose();
        }

        // Case 5: Zero matrix 5x3 -> rank 0, x all zeros, converged true, no NaN/Inf.
        public void PinvZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5;
            int n = 3;

            var A = arena.doubleMat(m, n);

            var b = arena.doubleRandomVector(m, -5f, 5f, 5151);

            var x = arena.doubleVec(n);

            int rank = SVD.pinvSolve(ref A, in b, ref x, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(0, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in x));

            for (int k = 0; k < n; k++)
                AssertClose(x[k], (double)0f, 1E-4f);

            arena.Dispose();
        }

        // Case 6: pseudoInverse of a square invertible 3x3. Aplus ~= A^{-1},
        // verified by A_copy * Aplus ~= I, rank 3.
        public void PseudoInverseSquareInvertible()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 3;

            // simple well-conditioned invertible matrix
            var A = arena.doubleMat(dim, dim);
            A[0, 0] = (double)4f; A[0, 1] = (double)1f; A[0, 2] = (double)0f;
            A[1, 0] = (double)1f; A[1, 1] = (double)3f; A[1, 2] = (double)1f;
            A[2, 0] = (double)0f; A[2, 1] = (double)1f; A[2, 2] = (double)2f;

            var A_copy = A.Copy();

            var Aplus = arena.doubleMat(dim, dim);

            int rank = SVD.pseudoInverse(ref A, ref Aplus, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(3, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in Aplus));

            var prod = double_OP.dot(A_copy, Aplus);
            Assert.IsTrue(Analysis_OP.IsIdentity(in prod, 1E-3f));

            arena.Dispose();
        }

        // Case 7: pseudoInverse of diag(2, 0) 2x2 -> diag(0.5, 0), rank 1, no NaN.
        public void PseudoInverseDiag()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 2;

            var A = arena.doubleMat(dim, dim);
            A[0, 0] = (double)2f;
            A[1, 1] = (double)0f;

            var Aplus = arena.doubleMat(dim, dim);

            int rank = SVD.pseudoInverse(ref A, ref Aplus, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(1, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in Aplus));

            AssertClose(Aplus[0, 0], (double)0.5f, 1E-4f);
            AssertClose(Aplus[0, 1], (double)0f, 1E-4f);
            AssertClose(Aplus[1, 0], (double)0f, 1E-4f);
            AssertClose(Aplus[1, 1], (double)0f, 1E-4f);

            arena.Dispose();
        }

        // Case 8: Moore-Penrose property on a rank-deficient case (case 3's A, 4x2 rank 1):
        // Aplus * A_copy * Aplus ~= Aplus.
        public void PseudoInverseMoorePenrose()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4;
            int n = 2;

            var A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
            {
                A[i, 0] = (double)1f;
                A[i, 1] = (double)1f;
            }

            var A_copy = A.Copy();

            // Aplus is n x m
            var Aplus = arena.doubleMat(n, m);

            int rank = SVD.pseudoInverse(ref A, ref Aplus, out bool converged);

            Assert.IsTrue(converged);
            Assert.AreEqual(1, rank);

            Assert.IsFalse(Analysis_OP.IsAnyNan(in Aplus));

            // Aplus * A_copy * Aplus  (n x m)
            var AplusA = double_OP.dot(Aplus, A_copy);          // n x n
            var AplusAAplus = double_OP.dot(AplusA, Aplus);     // n x m

            doubleMxN shouldBeZero = Aplus - AplusAAplus;
            Assert.IsTrue(Analysis_OP.IsZero(in shouldBeZero, 1E-3f));

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(double a, double b, double precision)
        {
            double diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void SVDSolverTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    // Managed throw-tests: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void PinvThrowsOnWrongBLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 3);
        var b = arena.doubleVec(3); // should be 4
        var x = arena.doubleVec(3);

        Assert.Catch<ArgumentException>(() => SVD.pinvSolve(ref A, in b, ref x, out bool converged));

        arena.Dispose();
    }

    [Test]
    public void PinvThrowsOnWrongXLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 3);
        var b = arena.doubleVec(4);
        var x = arena.doubleVec(2); // should be 3

        Assert.Catch<ArgumentException>(() => SVD.pinvSolve(ref A, in b, ref x, out bool converged));

        arena.Dispose();
    }

    [Test]
    public void PinvThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 3);
        var b = arena.doubleVec(4);
        var x = arena.doubleVec(3);

        Assert.Catch<ArgumentException>(() => SVD.pinvSolve(ref A, in b, ref x, out bool converged, (double)(-1f), 0));

        arena.Dispose();
    }

    [Test]
    public void PseudoInverseThrowsOnWrongShape()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 3);
        var Aplus = arena.doubleMat(4, 3); // should be 3 x 4

        Assert.Catch<ArgumentException>(() => SVD.pseudoInverse(ref A, ref Aplus, out bool converged));

        arena.Dispose();
    }

    [Test]
    public void PseudoInverseThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 3);
        var Aplus = arena.doubleMat(3, 4);

        Assert.Catch<ArgumentException>(() => SVD.pseudoInverse(ref A, ref Aplus, out bool converged, (double)(-1f), 0));

        arena.Dispose();
    }

}
