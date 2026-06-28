using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for SVD.svdTruncated / SVD.lowRankApprox. The spectrum from svdValues is the oracle:
// truncated Sk must equal the leading full singular values; Uk/Vk must be orthonormal; and the
// rank-k approximation's Frobenius error must equal the spectral tail sqrt(Σ_{i>=k} σ_i²)
// (Eckart-Young). lowRankApprox must agree with Uk diag(Sk) Vkᵀ.
public class fProxySVDLowRankTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            RandomTall12x5,
            RandomSquare8,
            LowRank10x6r3,
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RandomTall12x5: RandomTall12x5(); break;
                case TestType.RandomSquare8:  RandomSquare8();  break;
                case TestType.LowRank10x6r3:  LowRank10x6r3();  break;
            }
        }

        void Record(fProxy got, fProxy expected, fProxy diff)
        {
            if (Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = expected; Fail[3] = diff; }
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

        void AssertOrthoCols(in fProxyMxN basis, int rows, int cols, fProxy tol)
        {
            for (int a = 0; a < cols; a++)
                for (int b = a; b < cols; b++)
                {
                    fProxy dot = (fProxy)0;
                    for (int i = 0; i < rows; i++) dot += basis[i, a] * basis[i, b];
                    AssertClose(dot, (a == b) ? (fProxy)1 : (fProxy)0, tol);
                }
        }

        // Run all truncated/low-rank checks for one matrix at one k. fullS = full spectrum (length n).
        void CheckAtK(in fProxyMxN A, in fProxyN fullS, int k, fProxy normA2, ref Arena arena)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            fProxy tol = (fProxy)1E-3f * (normA2 + (fProxy)1);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, out bool cT);
            Assert.IsTrue(cT);

            // Sk equals the leading full singular values.
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], fullS[t], tol);

            // Uk, Vk orthonormal columns.
            AssertOrthoCols(in Uk, m, k, (fProxy)1E-3f);
            AssertOrthoCols(in Vk, n, k, (fProxy)1E-3f);

            // Rank-k approximation.
            var Ak = arena.fProxyMat(m, n);
            SVD.lowRankApprox(in A, ref Ak, k, out bool cL);
            Assert.IsTrue(cL);

            // Frobenius error squared == spectral tail Σ_{i>=k} σ_i² (Eckart-Young).
            fProxy efro2 = (fProxy)0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy d = A[i, j] - Ak[i, j];
                    efro2 += d * d;
                }
            fProxy tail = (fProxy)0;
            for (int i = k; i < n; i++) tail += fullS[i] * fullS[i];
            AssertLE(math.abs(efro2 - tail), tol);

            // lowRankApprox agrees with Uk diag(Sk) Vkᵀ.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy recon = (fProxy)0;
                    for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                    AssertClose(recon, Ak[i, j], tol);
                }
        }

        // Compute the full spectrum (oracle) and ||A||_F² once for a matrix.
        fProxyN Spectrum(in fProxyMxN A, ref Arena arena, out fProxy normA2)
        {
            int n = A.N_Cols;
            var fullS = arena.fProxyVec(n);
            SVD.svdValues(in A, ref fullS);
            normA2 = (fProxy)0;
            for (int i = 0; i < n; i++) normA2 += fullS[i] * fullS[i];
            return fullS;
        }

        void RandomTall12x5()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMatrix(12, 5, (fProxy)(-2f), (fProxy)2f, 555111);
            var fullS = Spectrum(in A, ref arena, out fProxy normA2);
            CheckAtK(in A, in fullS, 1, normA2, ref arena);
            CheckAtK(in A, in fullS, 2, normA2, ref arena);
            CheckAtK(in A, in fullS, 3, normA2, ref arena);
            CheckAtK(in A, in fullS, 5, normA2, ref arena);
            arena.Dispose();
        }

        void RandomSquare8()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMatrix(8, 8, (fProxy)(-3f), (fProxy)3f, 909090);
            var fullS = Spectrum(in A, ref arena, out fProxy normA2);
            CheckAtK(in A, in fullS, 1, normA2, ref arena);
            CheckAtK(in A, in fullS, 4, normA2, ref arena);
            CheckAtK(in A, in fullS, 8, normA2, ref arena);
            arena.Dispose();
        }

        void LowRank10x6r3()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 6, r = 3;
            var B = arena.fProxyRandomMatrix(m, r, (fProxy)(-2f), (fProxy)2f, 121212);
            var C = arena.fProxyRandomMatrix(r, n, (fProxy)(-2f), (fProxy)2f, 343434);
            var A = fProxyOP.dot(B, C);   // rank 3
            var fullS = Spectrum(in A, ref arena, out fProxy normA2);
            // k=3 should already capture all energy (tail ~ 0); k=2 leaves σ_2; k=6 exact.
            CheckAtK(in A, in fullS, 2, normA2, ref arena);
            CheckAtK(in A, in fullS, 3, normA2, ref arena);
            CheckAtK(in A, in fullS, 6, normA2, ref arena);
            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void LowRankTests(TestJob.TestType type)
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
    public void TruncatedThrowsOnBadK()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var Uk = arena.fProxyMat(6, 5);
        var Sk = arena.fProxyVec(5);
        var Vk = arena.fProxyMat(4, 5);
        Assert.Catch<ArgumentException>(() => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, 5, out bool _)); // k=5 > n=4
        arena.Dispose();
    }

    [Test]
    public void TruncatedThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 5);
        var Uk = arena.fProxyMat(3, 2);
        var Sk = arena.fProxyVec(2);
        var Vk = arena.fProxyMat(5, 2);
        Assert.Catch<ArgumentException>(() => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, 2, out bool _));
        arena.Dispose();
    }

    [Test]
    public void LowRankApproxThrowsOnWrongAkShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var Ak = arena.fProxyMat(4, 6);   // must be m x n = 6 x 4
        Assert.Catch<ArgumentException>(() => SVD.lowRankApprox(in A, ref Ak, 2, out bool _));
        arena.Dispose();
    }
}
