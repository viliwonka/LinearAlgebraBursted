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
public class floatSVDSubspaceTests
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
        public NativeArray<float> Fail;

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

        void Record(float got, float expected, float diff)
        {
            if (Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = diff;
            }
        }

        void AssertClose(float got, float expected, float tol)
        {
            float d = math.abs(got - expected);
            if (!(d <= tol)) Record(got, expected, d);
            Assert.IsTrue(d <= tol);
        }

        void AssertLE(float val, float limit)
        {
            if (!(val <= limit)) Record(val, limit, val - limit);
            Assert.IsTrue(val <= limit);
        }

        void AssertIntEq(int got, int expected)
        {
            if (got != expected) Record((float)got, (float)expected, (float)(got - expected));
            Assert.AreEqual(expected, got);
        }

        // The first `cols` columns of `basis` (rows 0..rows-1) must be orthonormal: colᵢ·colⱼ ≈ δᵢⱼ.
        void AssertOrthoCols(in floatMxN basis, int rows, int cols, float tol)
        {
            for (int a = 0; a < cols; a++)
                for (int b = a; b < cols; b++)
                {
                    float dot = (float)0;
                    for (int i = 0; i < rows; i++) dot += basis[i, a] * basis[i, b];
                    float expected = (a == b) ? (float)1 : (float)0;
                    AssertClose(dot, expected, tol);
                }
        }

        // Full property check for one matrix of known rank.
        void CheckSubspaces(in floatMxN A, int expectedRank, ref Arena arena, float tol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // ---- nullspace ----
            var nbasis = arena.floatMat(n, n);
            int dim = SVD.nullspaceBasis(in A, ref nbasis, out bool cN);
            Assert.IsTrue(cN);
            AssertIntEq(dim, n - expectedRank);

            // A·v ≈ 0 for each nullspace vector v.
            for (int col = 0; col < dim; col++)
            {
                float nrm2 = (float)0;
                for (int i = 0; i < m; i++)
                {
                    float s = (float)0;
                    for (int k = 0; k < n; k++) s += A[i, k] * nbasis[k, col];
                    nrm2 += s * s;
                }
                AssertLE(math.sqrt(nrm2), tol);
            }
            AssertOrthoCols(in nbasis, n, dim, tol);

            // ---- range ----
            var rbasis = arena.floatMat(m, n);
            int rank = SVD.rangeBasis(in A, ref rbasis, out bool cR);
            Assert.IsTrue(cR);
            AssertIntEq(rank, expectedRank);
            AssertOrthoCols(in rbasis, m, rank, tol);

            // Every column of A lies in span(range basis): Q Qᵀ a_c ≈ a_c.
            var coeff = arena.floatVec(n);
            for (int col = 0; col < n; col++)
            {
                for (int k = 0; k < rank; k++)
                {
                    float c = (float)0;
                    for (int t = 0; t < m; t++) c += rbasis[t, k] * A[t, col];
                    coeff[k] = c;
                }
                for (int i = 0; i < m; i++)
                {
                    float recon = (float)0;
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
            var A = arena.floatRandomMat(n, n, (float)(-2f), (float)2f, 9001);
            for (int d = 0; d < n; d++) A[d, d] += (float)8f;   // ensure full rank / conditioning
            CheckSubspaces(in A, n, ref arena, (float)1E-3f);
            arena.Dispose();
        }

        void Rank1_8x5()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 5;
            var u = arena.floatRandomVec(m, (float)(-2f), (float)2f, 4242);
            var v = arena.floatRandomVec(n, (float)(-2f), (float)2f, 2424);
            var A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = u[i] * v[j];   // rank 1
            CheckSubspaces(in A, 1, ref arena, (float)1E-3f);
            arena.Dispose();
        }

        void RankDeficient8x5r3()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 5, r = 3;
            var B = arena.floatRandomMat(m, r, (float)(-2f), (float)2f, 13579);
            var C = arena.floatRandomMat(r, n, (float)(-2f), (float)2f, 24680);
            var A = Linear_OP.dot(B, C);   // m x n, rank r (generic)
            CheckSubspaces(in A, r, ref arena, (float)1E-3f);
            arena.Dispose();
        }

        void ZeroMatrix7x4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 7, n = 4;
            var A = arena.floatMat(m, n);   // all zeros
            CheckSubspaces(in A, 0, ref arena, (float)1E-3f);
            arena.Dispose();
        }

        void Identity6()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = arena.floatIdentityMat(n);
            CheckSubspaces(in A, n, ref arena, (float)1E-3f);
            arena.Dispose();
        }

        void TallFullRank10x4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 4;
            var A = arena.floatRandomMat(m, n, (float)(-3f), (float)3f, 271828);
            CheckSubspaces(in A, n, ref arena, (float)1E-3f);
            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void SubspaceTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
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
        var A = arena.floatMat(3, 5);
        var basis = arena.floatMat(5, 5);
        Assert.Catch<ArgumentException>(() => SVD.nullspaceBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }

    [Test]
    public void NullspaceThrowsOnWrongBasisShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(6, 4);
        var basis = arena.floatMat(6, 4);   // must be n x n = 4 x 4
        Assert.Catch<ArgumentException>(() => SVD.nullspaceBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }

    [Test]
    public void RangeThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(3, 5);
        var basis = arena.floatMat(3, 5);
        Assert.Catch<ArgumentException>(() => SVD.rangeBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }

    [Test]
    public void RangeThrowsOnWrongBasisShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(6, 4);
        var basis = arena.floatMat(4, 4);   // must be m x n = 6 x 4
        Assert.Catch<ArgumentException>(() => SVD.rangeBasis(in A, ref basis, out bool _));
        arena.Dispose();
    }
}
