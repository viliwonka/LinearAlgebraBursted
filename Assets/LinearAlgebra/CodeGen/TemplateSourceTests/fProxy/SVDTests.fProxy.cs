using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class fProxySVDTests
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

            var U = arena.fProxyIdentityMatrix(dim);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            for (int i = 0; i < dim; i++)
                AssertClose(S[i], (fProxy)1f, 1E-4f);

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.fProxyMat(dim, dim);
            U[0, 0] = 3f;
            U[1, 1] = -2f;
            U[2, 2] = 0.5f;
            U[3, 3] = 5f;

            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // Singular values are |eigenvalues|, sorted descending: 5, 3, 2, 0.5
            AssertClose(S[0], (fProxy)5f, 1E-4f);
            AssertClose(S[1], (fProxy)3f, 1E-4f);
            AssertClose(S[2], (fProxy)2f, 1E-4f);
            AssertClose(S[3], (fProxy)0.5f, 1E-4f);

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
            var U = arena.fProxyMat(dim, dim);
            U[0, 0] = 3f; U[0, 1] = 0f;
            U[1, 0] = 4f; U[1, 1] = 5f;

            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // singular values: sqrt(45) ~ 6.7082039, sqrt(5) ~ 2.2360680
            AssertClose(S[0], (fProxy)6.7082039f, 1E-3f);
            AssertClose(S[1], (fProxy)2.2360680f, 1E-3f);

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

            var U = arena.fProxyRandomMatrix(dim, dim, -10f, 10f, 314221);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

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

            var U = arena.fProxyRandomMatrix(m, n, -10f, 10f, 778231);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);

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

            var U = arena.fProxyRandomMatrix(dim, dim, -5f, 5f, 559013);
            // make column 2 an exact copy of column 0 -> rank deficient
            for (int i = 0; i < dim; i++)
                U[i, 2] = U[i, 0];

            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            // exactly one zero singular value (smallest), the third is non-trivial
            Assert.IsTrue(S[3] < 1E-4f, $"Expected S[3] < tol, got {S[3]}");
            Assert.IsTrue(S[2] > 1E-4f, $"Expected S[2] > tol, got {S[2]}");

            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            // first 3 columns of U are orthonormal (the column matching zero sigma is zeroed)
            for (int a = 0; a < 3; a++) {
                for (int b = a; b < 3; b++) {
                    fProxy dotcol = (fProxy)0f;
                    for (int i = 0; i < dim; i++)
                        dotcol += U[i, a] * U[i, b];
                    fProxy expected = (a == b) ? (fProxy)1f : (fProxy)0f;
                    AssertClose(dotcol, expected, 1E-4f);
                }
            }

            arena.Dispose();
        }

        public void SVDZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 3;

            var U = arena.fProxyMat(dim, dim);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            for (int i = 0; i < dim; i++)
                AssertClose(S[i], (fProxy)0f, 1E-4f);

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

            var U = arena.fProxyMat(m, n);
            U[0, 0] = 1f;
            U[1, 0] = 2f;
            U[2, 0] = 3f;
            U[3, 0] = 4f;
            U[4, 0] = 5f;

            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // S[0] == column 2-norm = sqrt(1+4+9+16+25) = sqrt(55) ~ 7.4161985
            AssertClose(S[0], (fProxy)7.4161985f, 1E-3f);

            // U column has unit norm
            fProxy normSq = (fProxy)0f;
            for (int i = 0; i < m; i++)
                normSq += U[i, 0] * U[i, 0];
            AssertClose(normSq, (fProxy)1f, 1E-4f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDNonConvergence()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.fProxyHilbertMatrix(dim);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            // maxSweeps = 1: regardless of convergence bool, outputs must be finite,
            // S descending and non-negative.
            bool success = SVD.svdDecomposition(ref U, ref S, ref V, 1);

            // The return value is intentionally not asserted (may or may not converge).
            Debug.Log($"SVD maxSweeps=1 converged: {success}");

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            AssertDescendingNonNegative(in S, dim);

            arena.Dispose();
        }

        private void AssertReconstruct(in fProxyMxN A, in fProxyMxN U, in fProxyN S, in fProxyMxN V, ref Arena arena, fProxy precision)
        {
            // A ~= U * diag(S) * V^T
            var diagS = arena.fProxyDiagonalMatrix(in S);
            var US = fProxyOP.dot(U, diagS);
            var Vt = fProxyOP.trans(V);
            var recon = fProxyOP.dot(US, Vt);

            fProxyMxN shouldBeZero = A - recon;

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            Debug.Log($"Error of max(abs(A - U*diag(S)*V^T)): {zeroError}");

            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
        }

        private void AssertDescendingNonNegative(in fProxyN S, int n)
        {
            for (int i = 0; i < n; i++)
                Assert.IsTrue(S[i] >= (fProxy)(-1E-6f), $"S[{i}] = {S[i]} is negative");

            for (int i = 1; i < n; i++)
                Assert.IsTrue(S[i] <= S[i - 1] + (fProxy)1E-6f, $"S not descending: S[{i - 1}]={S[i - 1]} < S[{i}]={S[i]}");
        }

        private void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = Unity.Mathematics.math.abs(a - b);
            Assert.IsTrue(diff <= precision, $"Expected {b} got {a} (diff {diff})");
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void SVDDecompTests(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }

    // Managed throw-tests: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void SVDThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(2, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnWrongSLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(2);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnWrongVSize()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(2, 2);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V, 0));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnBadEps()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V, 30, (fProxy)0f));

        arena.Dispose();
    }

}
