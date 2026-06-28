using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for SVD.svdRandomized (Halko-Martinsson-Tropp). svdValues is the true-spectrum oracle.
// Invariants checked on every case: Uk/Vk orthonormal columns; Sk descending; the compression bound
// Sk[i] <= σ_i(A) (singular values of QᵀA never exceed A's); the leading value is recovered well.
// For exactly low-rank A with k >= rank the reconstruction Uk diag(Sk) Vkᵀ ≈ A.
public class doubleSVDRandomizedTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ExactRank3_24x12,
            ExactRank5_40x16,
            GeneralRandom20x10,
        }

        public TestType Type;
        public NativeArray<double> Fail;   // [0] flag, [1] got, [2] expected/limit, [3] diff

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExactRank3_24x12:  ExactRank3_24x12();  break;
                case TestType.ExactRank5_40x16:  ExactRank5_40x16();  break;
                case TestType.GeneralRandom20x10: GeneralRandom20x10(); break;
            }
        }

        void Record(double got, double expected, double diff)
        {
            if (Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = got; Fail[2] = expected; Fail[3] = diff; }
        }

        void AssertClose(double got, double expected, double tol)
        {
            double d = math.abs(got - expected);
            if (!(d <= tol)) Record(got, expected, d);
            Assert.IsTrue(d <= tol);
        }

        void AssertLE(double val, double limit)
        {
            if (!(val <= limit)) Record(val, limit, val - limit);
            Assert.IsTrue(val <= limit);
        }

        void AssertGE(double val, double limit)
        {
            if (!(val >= limit)) Record(val, limit, limit - val);
            Assert.IsTrue(val >= limit);
        }

        void AssertOrthoCols(in doubleMxN basis, int rows, int cols, double tol)
        {
            for (int a = 0; a < cols; a++)
                for (int b = a; b < cols; b++)
                {
                    double dot = (double)0;
                    for (int i = 0; i < rows; i++) dot += basis[i, a] * basis[i, b];
                    AssertClose(dot, (a == b) ? (double)1 : (double)0, tol);
                }
        }

        void CheckRandomized(in doubleMxN A, int k, int oversample, int powerIters, uint seed,
                             bool expectExact, ref Arena arena)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // oracle spectrum + ||A||_F
            var fullS = arena.doubleVec(n);
            SVD.svdValues(in A, ref fullS);
            double normA = (double)0;
            for (int i = 0; i < n; i++) normA += fullS[i] * fullS[i];
            normA = math.sqrt(normA);

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            bool ok = SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, oversample, powerIters, seed, 75);
            Assert.IsTrue(ok);

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            // Sk descending; compression bound Sk[i] <= σ_i(A); leading value recovered.
            double slack = (double)1E-3f * (fullS[0] + (double)1);
            for (int i = 0; i < k; i++)
            {
                if (i + 1 < k) AssertGE(Sk[i] + slack, Sk[i + 1]);
                AssertLE(Sk[i], fullS[i] + slack);
            }
            AssertGE(Sk[0], (double)0.9f * fullS[0]);

            if (expectExact)
            {
                // ||A - Uk diag(Sk) Vkᵀ||_F  <=  1e-2 * ||A||_F
                double err2 = (double)0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double recon = (double)0;
                        for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                        double d = A[i, j] - recon;
                        err2 += d * d;
                    }
                AssertLE(math.sqrt(err2), (double)1E-2f * (normA + (double)1E-6f));
            }
        }

        void ExactRank3_24x12()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 24, n = 12, r = 3;
            var B = arena.doubleRandomMatrix(m, r, (double)(-2f), (double)2f, 1001);
            var C = arena.doubleRandomMatrix(r, n, (double)(-2f), (double)2f, 2002);
            var A = doubleOP.dot(B, C);   // rank 3
            CheckRandomized(in A, 3, 6, 2, 12345u, true, ref arena);
            arena.Dispose();
        }

        void ExactRank5_40x16()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 40, n = 16, r = 5;
            var B = arena.doubleRandomMatrix(m, r, (double)(-2f), (double)2f, 3003);
            var C = arena.doubleRandomMatrix(r, n, (double)(-2f), (double)2f, 4004);
            var A = doubleOP.dot(B, C);   // rank 5
            CheckRandomized(in A, 5, 8, 2, 67890u, true, ref arena);
            arena.Dispose();
        }

        void GeneralRandom20x10()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 20, n = 10;
            var A = arena.doubleRandomMatrix(m, n, (double)(-2f), (double)2f, 555);
            // flat-ish spectrum: only assert invariants + leading value (power iters sharpen it).
            CheckRandomized(in A, 4, 8, 3, 24680u, false, ref arena);
            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void RandomizedTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
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
    public void RandomizedThrowsOnBadK()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(6, 4);
        var Uk = arena.doubleMat(6, 5);
        var Sk = arena.doubleVec(5);
        var Vk = arena.doubleMat(4, 5);
        Assert.Catch<ArgumentException>(() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, 5)); // k=5 > n=4
        arena.Dispose();
    }

    [Test]
    public void RandomizedThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(3, 5);
        var Uk = arena.doubleMat(3, 2);
        var Sk = arena.doubleVec(2);
        var Vk = arena.doubleMat(5, 2);
        Assert.Catch<ArgumentException>(() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, 2));
        arena.Dispose();
    }
}
