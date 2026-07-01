using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for SVD.nullspaceBasis / SVD.rangeBasis: from A = U diag(S) Vᵀ on known-rank matrices,
// verify (1) the reported dimension/rank, (2) A·v ≈ 0 for every nullspace vector, (3) both bases are
// orthonormal in their used columns, (4) every column of A is reconstructable from the range basis.
public class fProxySVDSubspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            FullRankSquare6,
            Rank1_8x5,
            RankDeficient8x5r3,
            ZeroMatrix7x4,
            Identity6,
            TallFullRank10x4,
        }

        public TestType Type;

        // [0] flag (1 = failure), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.FullRankSquare6:    FullRankSquare6();    break;
                case TestType.Rank1_8x5:          Rank1_8x5();          break;
                case TestType.RankDeficient8x5r3: RankDeficient8x5r3(); break;
                case TestType.ZeroMatrix7x4:      ZeroMatrix7x4();      break;
                case TestType.Identity6:          Identity6();          break;
                case TestType.TallFullRank10x4:   TallFullRank10x4();   break;
            }
        }

        // ---- failure-recording asserts ----

        void Record(fProxy got, fProxy expected, fProxy diff)
        {
            if (Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = diff;
            }
        }

        void AssertClose(fProxy got, fProxy expected, fProxy tol)
        {
            fProxy d = math.abs(got - expected);
            if (!(d <= tol)) Record(got, expected, d);
            Assert.IsTrue(d <= tol);
        }

        void AssertLE(fProxy val, fProxy limit)
        {
            if (!(val <= limit)) Record(val, limit, val - limit);
            Assert.IsTrue(val <= limit);
        }

        void AssertIntEq(int got, int expected)
        {
            if (got != expected) Record((fProxy)got, (fProxy)expected, (fProxy)(got - expected));
            Assert.AreEqual(expected, got);
        }

        // The first `cols` columns of `basis` (rows 0..rows-1) must be orthonormal: colᵢ·colⱼ ≈ δᵢⱼ.
        void AssertOrthoCols(in fProxyMxN basis, int rows, int cols, fProxy tol)
        {
            for (int a = 0; a < cols; a++)
                for (int b = a; b < cols; b++)
                {
                    fProxy dot = (fProxy)0;
                    for (int i = 0; i < rows; i++) dot += basis[i, a] * basis[i, b];
                    fProxy expected = (a == b) ? (fProxy)1 : (fProxy)0;
                    AssertClose(dot, expected, tol);
                }
        }

        // Full property check for one matrix of known rank.
        void CheckSubspaces(in fProxyMxN A, int expectedRank, ref Arena arena, fProxy tol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // ---- nullspace ----
            var nbasis = arena.fProxyMat(n, n);
            int dim = SVD.nullspaceBasis(in A, ref nbasis, out bool cN);
            Assert.IsTrue(cN);
            AssertIntEq(dim, n - expectedRank);

            // A·v ≈ 0 for each nullspace vector v.
            for (int col = 0; col < dim; col++)
            {
                fProxy nrm2 = (fProxy)0;
                for (int i = 0; i < m; i++)
                {
                    fProxy s = (fProxy)0;
                    for (int k = 0; k < n; k++) s += A[i, k] * nbasis[k, col];
                    nrm2 += s * s;
                }
                AssertLE(math.sqrt(nrm2), tol);
            }
            AssertOrthoCols(in nbasis, n, dim, tol);

            // ---- range ----
            var rbasis = arena.fProxyMat(m, n);
            int rank = SVD.rangeBasis(in A, ref rbasis, out bool cR);
            Assert.IsTrue(cR);
            AssertIntEq(rank, expectedRank);
            AssertOrthoCols(in rbasis, m, rank, tol);

            // Every column of A lies in span(range basis): Q Qᵀ a_c ≈ a_c.
            var coeff = arena.fProxyVec(n);
            for (int col = 0; col < n; col++)
            {
                for (int k = 0; k < rank; k++)
                {
                    fProxy c = (fProxy)0;
                    for (int t = 0; t < m; t++) c += rbasis[t, k] * A[t, col];
                    coeff[k] = c;
                }
                for (int i = 0; i < m; i++)
                {
                    fProxy recon = (fProxy)0;
                    for (int k = 0; k < rank; k++) recon += coeff[k] * rbasis[i, k];
                    AssertClose(recon, A[i, col], tol);
                }
            }
        }

        // ---- cases ----

        void FullRankSquare6()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = arena.fProxyRandomMat(n, n, (fProxy)(-2f), (fProxy)2f, 9001);
            for (int d = 0; d < n; d++) A[d, d] += (fProxy)8f;   // ensure full rank / conditioning
            CheckSubspaces(in A, n, ref arena, (fProxy)1E-3f);
            arena.Dispose();
        }

        void Rank1_8x5()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 5;
            var u = arena.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 4242);
            var v = arena.fProxyRandomVec(n, (fProxy)(-2f), (fProxy)2f, 2424);
            var A = arena.fProxyMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = u[i] * v[j];   // rank 1
            CheckSubspaces(in A, 1, ref arena, (fProxy)1E-3f);
            arena.Dispose();
        }

        void RankDeficient8x5r3()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 5, r = 3;
            var B = arena.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, 13579);
            var C = arena.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, 24680);
            var A = fProxy_OP.dot(B, C);   // m x n, rank r (generic)
            CheckSubspaces(in A, r, ref arena, (fProxy)1E-3f);
            arena.Dispose();
        }

        void ZeroMatrix7x4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 7, n = 4;
            var A = arena.fProxyMat(m, n);   // all zeros
            CheckSubspaces(in A, 0, ref arena, (fProxy)1E-3f);
            arena.Dispose();
        }

        void Identity6()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = arena.fProxyIdentityMat(n);
            CheckSubspaces(in A, n, ref arena, (fProxy)1E-3f);
            arena.Dispose();
        }

        void TallFullRank10x4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 271828);
            CheckSubspaces(in A, n, ref arena, (fProxy)1E-3f);
            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void SubspaceTests(TestJob.TestType type)
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
        finally
        {
            fail.Dispose();
        }
    }

    // ---- argument-validation tests (managed) ----

    [Test]
    public void NullspaceThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 5);
        var basis = arena.fProxyMat(5, 5);
        Assert.Catch<ArgumentException>(() => SVD.nullspaceBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }

    [Test]
    public void NullspaceThrowsOnWrongBasisShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var basis = arena.fProxyMat(6, 4);   // must be n x n = 4 x 4
        Assert.Catch<ArgumentException>(() => SVD.nullspaceBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }

    [Test]
    public void RangeThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 5);
        var basis = arena.fProxyMat(3, 5);
        Assert.Catch<ArgumentException>(() => SVD.rangeBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }

    [Test]
    public void RangeThrowsOnWrongBasisShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var basis = arena.fProxyMat(4, 4);   // must be m x n = 6 x 4
        Assert.Catch<ArgumentException>(() => SVD.rangeBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }
}
