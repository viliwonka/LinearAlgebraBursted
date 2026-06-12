using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class floatSVDTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            SVDIdentity,
            SVDDiagonal,
            SVDKnown2x2,
            SVDRandomSquare,
            SVDRectangularTall,
            SVDRankDeficient,
            SVDZero,
            SVDSingleColumn,
            SVDNonConvergence
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.SVDIdentity:
                    SVDIdentity();
                break;
                case TestType.SVDDiagonal:
                    SVDDiagonal();
                break;
                case TestType.SVDKnown2x2:
                    SVDKnown2x2();
                break;
                case TestType.SVDRandomSquare:
                    SVDRandomSquare();
                break;
                case TestType.SVDRectangularTall:
                    SVDRectangularTall();
                break;
                case TestType.SVDRankDeficient:
                    SVDRankDeficient();
                break;
                case TestType.SVDZero:
                    SVDZero();
                break;
                case TestType.SVDSingleColumn:
                    SVDSingleColumn();
                break;
                case TestType.SVDNonConvergence:
                    SVDNonConvergence();
                break;
            }
        }

        public void SVDIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.floatIdentityMatrix(dim);
            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            for (int i = 0; i < dim; i++)
                AssertClose(S[i], (float)1f, 1E-4f);

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.floatMat(dim, dim);
            U[0, 0] = 3f;
            U[1, 1] = -2f;
            U[2, 2] = 0.5f;
            U[3, 3] = 5f;

            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // Singular values are |eigenvalues|, sorted descending: 5, 3, 2, 0.5
            AssertClose(S[0], (float)5f, 1E-4f);
            AssertClose(S[1], (float)3f, 1E-4f);
            AssertClose(S[2], (float)2f, 1E-4f);
            AssertClose(S[3], (float)0.5f, 1E-4f);

            // descending and non-negative
            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 2;

            // A = [[3, 0], [4, 5]]
            var U = arena.floatMat(dim, dim);
            U[0, 0] = 3f; U[0, 1] = 0f;
            U[1, 0] = 4f; U[1, 1] = 5f;

            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // singular values: sqrt(45) ~ 6.7082039, sqrt(5) ~ 2.2360680
            AssertClose(S[0], (float)6.7082039f, 1E-3f);
            AssertClose(S[1], (float)2.2360680f, 1E-3f);

            AssertDescendingNonNegative(in S, dim);

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDRandomSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.floatRandomMatrix(dim, dim, -10f, 10f, 314221);
            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDRectangularTall()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6;
            int n = 3;

            var U = arena.floatRandomMatrix(m, n, -10f, 10f, 778231);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            // U is 6x3 with orthonormal columns -> U^T U = I_3
            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertDescendingNonNegative(in S, n);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.floatRandomMatrix(dim, dim, -5f, 5f, 559013);
            // make column 2 an exact copy of column 0 -> rank deficient
            for (int i = 0; i < dim; i++)
                U[i, 2] = U[i, 0];

            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            // exactly one zero singular value (smallest), the third is non-trivial
            Assert.IsTrue(S[3] < 1E-4f);
            Assert.IsTrue(S[2] > 1E-4f);

            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            // first 3 columns of U are orthonormal (the column matching zero sigma is zeroed)
            for (int a = 0; a < 3; a++) {
                for (int b = a; b < 3; b++) {
                    float dotcol = (float)0f;
                    for (int i = 0; i < dim; i++)
                        dotcol += U[i, a] * U[i, b];
                    float expected = (a == b) ? (float)1f : (float)0f;
                    AssertClose(dotcol, expected, 1E-4f);
                }
            }

            arena.Dispose();
        }

        public void SVDZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 3;

            var U = arena.floatMat(dim, dim);
            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            for (int i = 0; i < dim; i++)
                AssertClose(S[i], (float)0f, 1E-4f);

            // U is all zeros (every column matches a zero singular value)
            Assert.IsTrue(Analysis.IsZero(in U, 1E-4f));

            // V stays identity (no rotations applied)
            Assert.IsTrue(Analysis.IsIdentity(in V, 1E-4f));

            arena.Dispose();
        }

        public void SVDSingleColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5;
            int n = 1;

            var U = arena.floatMat(m, n);
            U[0, 0] = 1f;
            U[1, 0] = 2f;
            U[2, 0] = 3f;
            U[3, 0] = 4f;
            U[4, 0] = 5f;

            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // S[0] == column 2-norm = sqrt(1+4+9+16+25) = sqrt(55) ~ 7.4161985
            AssertClose(S[0], (float)7.4161985f, 1E-3f);

            // U column has unit norm
            float normSq = (float)0f;
            for (int i = 0; i < m; i++)
                normSq += U[i, 0] * U[i, 0];
            AssertClose(normSq, (float)1f, 1E-4f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDNonConvergence()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.floatHilbertMatrix(dim);
            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);

            // maxSweeps = 1: regardless of convergence bool, outputs must be finite,
            // S descending and non-negative.
            SVD.svdDecomposition(ref U, ref S, ref V, 1);

            // The return value is intentionally not asserted (may or may not converge).

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            AssertDescendingNonNegative(in S, dim);

            arena.Dispose();
        }

        private void AssertReconstruct(in floatMxN A, in floatMxN U, in floatN S, in floatMxN V, ref Arena arena, float precision)
        {
            // A ~= U * diag(S) * V^T
            var diagS = arena.floatDiagonalMatrix(in S);
            var US = floatOP.dot(U, diagS);
            var Vt = floatOP.trans(V);
            var recon = floatOP.dot(US, Vt);

            floatMxN shouldBeZero = A - recon;

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            // Fail layout: [1]=zeroError, [2]=precision, [3]=diff
            if (!(zeroError <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
        }

        // Fail layout: [1]=S[i] (offending element), [2]=bound or S[i-1], [3]=index cast to float
        private void AssertDescendingNonNegative(in floatN S, int n)
        {
            for (int i = 0; i < n; i++)
            {
                bool nonNeg = S[i] >= (float)(-1E-6f);
                if (!nonNeg && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = S[i];
                    Fail[2] = (float)(-1E-6f);
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(nonNeg);
            }

            for (int i = 1; i < n; i++)
            {
                bool descending = S[i] <= S[i - 1] + (float)1E-6f;
                if (!descending && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = S[i];
                    Fail[2] = S[i - 1];
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(descending);
            }
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(float a, float b, float precision)
        {
            float diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
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
    public void SVDDecompTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    // Managed throw-tests: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void SVDThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.floatMat(2, 3);
        var S = arena.floatVec(3);
        var V = arena.floatMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnWrongSLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.floatMat(4, 3);
        var S = arena.floatVec(2);
        var V = arena.floatMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnWrongVSize()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.floatMat(4, 3);
        var S = arena.floatVec(3);
        var V = arena.floatMat(2, 2);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.floatMat(4, 3);
        var S = arena.floatVec(3);
        var V = arena.floatMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V, 0));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnBadEps()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.floatMat(4, 3);
        var S = arena.floatVec(3);
        var V = arena.floatMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V, 30, (float)0f));

        arena.Dispose();
    }

}
