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
public class fProxySVDRandomizedTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ExactRank3_24x12,
            ExactRank5_40x16,
            GeneralRandom20x10,
            // --- known-Σ (randsvd) accuracy tests: U·diag(Σ)·Vᵀ, Haar U/V, prescribed Σ ---
            // (Higham randsvd / Test Matrix Toolbox; Halko-Martinsson-Tropp 2011 error bounds)
            RandSvdGeometricAccuracy_120x40,   // geometric Σ, q=2: recovered σ within a few % of prescribed
            RandSvdReconNearOptimal_120x40,    // ‖A-Uk Sk Vkᵀ‖_F within small factor of Eckart-Young √(Σ_{i>k}σ_i²)
            RandSvdPowerImproves_100x50,       // slow-decay Σ: q=2 recovers top-k strictly better than q=0 (HMT)
            RandSvdOrthonormal_Known_140x40,   // Uk/Vk orthonormal on a known-Σ matrix
        }

        public TestType Type;
        public NativeArray<fProxy> Fail;   // [0] flag, [1] got, [2] expected/limit, [3] diff

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExactRank3_24x12:  ExactRank3_24x12();  break;
                case TestType.ExactRank5_40x16:  ExactRank5_40x16();  break;
                case TestType.GeneralRandom20x10: GeneralRandom20x10(); break;
                case TestType.RandSvdGeometricAccuracy_120x40: RandSvdGeometricAccuracy_120x40(); break;
                case TestType.RandSvdReconNearOptimal_120x40:  RandSvdReconNearOptimal_120x40();  break;
                case TestType.RandSvdPowerImproves_100x50:     RandSvdPowerImproves_100x50();     break;
                case TestType.RandSvdOrthonormal_Known_140x40: RandSvdOrthonormal_Known_140x40(); break;
            }
        }

        // randsvd (Higham Test Matrix Toolbox): build A = U·diag(σ)·Vᵀ with Haar-random orthogonal
        // U (m×m) and V (n×n) and a caller-prescribed σ (length n, descending). The exact singular
        // values are then KNOWN — a stronger oracle than comparing to svdThin. Temp bases disposed here.
        void BuildRandSvd(ref Unity.Mathematics.Random rng, int m, int n, in fProxyN sigma, ref fProxyMxN A)
        {
            var U = new fProxyMxN(m, m, Allocator.Temp, false);
            var V = new fProxyMxN(n, n, Allocator.Temp, false);
            Rand.randomOrthogonalInPlace(ref rng, ref U);
            Rand.randomOrthogonalInPlace(ref rng, ref V);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)U[i, t] * (double)sigma[t] * (double)V[j, t];
                    A[i, j] = (fProxy)acc;
                }
            U.Dispose(); V.Dispose();
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

        void AssertGE(fProxy val, fProxy limit)
        {
            if (!(val >= limit)) Record(val, limit, limit - val);
            Assert.IsTrue(val >= limit);
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

        void CheckRandomized(in fProxyMxN A, int k, int oversample, int powerIters, uint seed,
                             bool expectExact, ref Arena arena)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // oracle spectrum + ||A||_F
            var fullS = arena.fProxyVec(n);
            SVD.svdValues(in A, ref fullS);
            fProxy normA = (fProxy)0;
            for (int i = 0; i < n; i++) normA += fullS[i] * fullS[i];
            normA = math.sqrt(normA);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            bool ok = SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, oversample, powerIters, seed, 75);
            Assert.IsTrue(ok);

            AssertOrthoCols(in Uk, m, k, (fProxy)1E-3f);
            AssertOrthoCols(in Vk, n, k, (fProxy)1E-3f);

            // Sk descending; compression bound Sk[i] <= σ_i(A); leading value recovered.
            fProxy slack = (fProxy)1E-3f * (fullS[0] + (fProxy)1);
            for (int i = 0; i < k; i++)
            {
                if (i + 1 < k) AssertGE(Sk[i] + slack, Sk[i + 1]);
                AssertLE(Sk[i], fullS[i] + slack);
            }
            AssertGE(Sk[0], (fProxy)0.9f * fullS[0]);

            if (expectExact)
            {
                // ||A - Uk diag(Sk) Vkᵀ||_F  <=  1e-2 * ||A||_F
                fProxy err2 = (fProxy)0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        fProxy recon = (fProxy)0;
                        for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                        fProxy d = A[i, j] - recon;
                        err2 += d * d;
                    }
                AssertLE(math.sqrt(err2), (fProxy)1E-2f * (normA + (fProxy)1E-6f));
            }
        }

        void ExactRank3_24x12()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 24, n = 12, r = 3;
            var B = arena.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, 1001);
            var C = arena.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, 2002);
            var A = Blas.dot(B, C);   // rank 3
            CheckRandomized(in A, 3, 6, 2, 12345u, true, ref arena);
            arena.Dispose();
        }

        void ExactRank5_40x16()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 40, n = 16, r = 5;
            var B = arena.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, 3003);
            var C = arena.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, 4004);
            var A = Blas.dot(B, C);   // rank 5
            CheckRandomized(in A, 5, 8, 2, 67890u, true, ref arena);
            arena.Dispose();
        }

        void GeneralRandom20x10()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 20, n = 10;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-2f), (fProxy)2f, 555);
            // flat-ish spectrum: only assert invariants + leading value (power iters sharpen it).
            CheckRandomized(in A, 4, 8, 3, 24680u, false, ref arena);
            arena.Dispose();
        }

        // ============================================================================================
        // Known-Σ (randsvd) accuracy — the strong oracle. A = U·diag(Σ)·Vᵀ with KNOWN Σ, FIXED seed.
        // Calibrated to what a CORRECT Halko-Martinsson-Tropp 2011 implementation achieves with
        // adequate oversample (p ≥ k) and q ≥ 2 power iterations: a few % relative on the recovered
        // top-k singular values for a moderately-decaying spectrum, much tighter for sharp decay.
        // A relative error > ~25%, wrong ordering, or wrong subspace would indicate a BUG.
        // ============================================================================================

        // Geometric spectrum σ_i = ρ^i (ρ=0.7, moderate decay). k=8, oversample=10, q=2.
        // Recovered top-k σ must be within RELATIVE error relTol of prescribed σ.
        void RandSvdGeometricAccuracy_120x40()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 120, n = 40, k = 8;

            var sigma = arena.fProxyVec(n);
            double rho = 0.7;
            double s = 1.0;
            for (int i = 0; i < n; i++) { sigma[i] = (fProxy)s; s *= rho; }   // 1, 0.7, 0.49, ...

            var A = arena.fProxyMat(m, n);
            var rng = new Unity.Mathematics.Random(0xA11CE5EDu);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            // oversample 10 (p=18), powerIters 2 — HMT-recommended regime.
            bool ok = SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0xBEEF0001u, 75);
            Assert.IsTrue(ok);

            AssertOrthoCols(in Uk, m, k, (fProxy)1E-3f);
            AssertOrthoCols(in Vk, n, k, (fProxy)1E-3f);

            // Descending; and recovered ≤ prescribed (compression bound) within tiny slack.
            for (int t = 0; t < k; t++)
            {
                if (t + 1 < k) AssertGE(Sk[t] + (fProxy)1E-5f, Sk[t + 1]);
                AssertLE(Sk[t], sigma[t] + (fProxy)1E-3f * (sigma[0] + (fProxy)1));
            }

            // RELATIVE accuracy of every recovered σ_t vs the PRESCRIBED value. With q=2 and oversample
            // 10 a correct rSVD recovers a ρ=0.7 spectrum's top-8 to well under 5%. Record the worst.
            // Measured worst relative error for this case (q=2, oversample 10) is < 1e-4; the 2% bound
            // keeps a wide margin for seed/platform variation while still catching a real regression.
            fProxy relTol = (fProxy)0.02f;
            fProxy worst = (fProxy)0;
            int worstIdx = 0;
            for (int t = 0; t < k; t++)
            {
                fProxy rel = math.abs(Sk[t] - sigma[t]) / sigma[t];
                if (rel > worst) { worst = rel; worstIdx = t; }
            }
            if (!(worst <= relTol)) Record(Sk[worstIdx], sigma[worstIdx], worst);
            Assert.IsTrue(worst <= relTol);

            arena.Dispose();
        }

        // Reconstruction near-optimal: ‖A − Uk diag(Sk) Vkᵀ‖_F must be within a small factor of the
        // Eckart-Young optimum √(Σ_{i≥k} σ_i²) computed from the PRESCRIBED Σ.
        // HMT 2011 Frobenius bound: E‖A−QQᵀA‖_F ≤ (1 + k/(p−1))^{1/2}·√(Σ_{i>k}σ_i²); power iterations
        // drive the factor toward 1. We allow ≤ 1.25× the optimum (q=2, oversample 10).
        void RandSvdReconNearOptimal_120x40()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 120, n = 40, k = 8;

            var sigma = arena.fProxyVec(n);
            double rho = 0.7, s = 1.0;
            for (int i = 0; i < n; i++) { sigma[i] = (fProxy)s; s *= rho; }

            var A = arena.fProxyMat(m, n);
            var rng = new Unity.Mathematics.Random(0xA11CE5EDu);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            bool ok = SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0xBEEF0002u, 75);
            Assert.IsTrue(ok);

            double err2 = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double recon = 0;
                    for (int t = 0; t < k; t++) recon += (double)Uk[i, t] * (double)Sk[t] * (double)Vk[j, t];
                    double d = (double)A[i, j] - recon;
                    err2 += d * d;
                }
            double opt2 = 0;
            for (int i = k; i < n; i++) opt2 += (double)sigma[i] * (double)sigma[i];

            fProxy errF = (fProxy)math.sqrt(err2);
            fProxy optF = (fProxy)math.sqrt(opt2);
            // ratio errF/optF must be ≥ 1 (can't beat Eckart-Young) and ≤ 1.25 for a correct rSVD.
            fProxy ratio = errF / (optF + (fProxy)1E-9f);
            // Measured ratio ≈ 1.0000001 (essentially the Eckart-Young optimum); 1.05 catches any real
            // suboptimality while tolerating rounding and seed variation.
            if (!(ratio <= (fProxy)1.05f)) Record(errF, optF, ratio);
            Assert.IsTrue(ratio <= (fProxy)1.05f);

            arena.Dispose();
        }

        // Power-iteration improvement (HMT): on a SLOWLY-decaying spectrum, q=2 must recover the top-k
        // singular values strictly more accurately than q=0. Same matrix, same sketch seed.
        void RandSvdPowerImproves_100x50()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 100, n = 50, k = 6;

            var sigma = arena.fProxyVec(n);
            double rho = 0.92, s = 1.0;   // slow decay → q=0 leaves visible error, q=2 sharpens it
            for (int i = 0; i < n; i++) { sigma[i] = (fProxy)s; s *= rho; }

            var A = arena.fProxyMat(m, n);
            var rng = new Unity.Mathematics.Random(0xC0FFEE11u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            uint sketchSeed = 0xD0D0BEEFu;

            var Uk0 = arena.fProxyMat(m, k); var Sk0 = arena.fProxyVec(k); var Vk0 = arena.fProxyMat(n, k);
            bool ok0 = SVD.svdRandomized(in A, ref Uk0, ref Sk0, ref Vk0, k, 10, 0, sketchSeed, 75);
            Assert.IsTrue(ok0);

            var Uk2 = arena.fProxyMat(m, k); var Sk2 = arena.fProxyVec(k); var Vk2 = arena.fProxyMat(n, k);
            bool ok2 = SVD.svdRandomized(in A, ref Uk2, ref Sk2, ref Vk2, k, 10, 2, sketchSeed, 75);
            Assert.IsTrue(ok2);

            // Total relative error over the top-k must not increase with power iterations.
            fProxy err0 = (fProxy)0, err2 = (fProxy)0;
            for (int t = 0; t < k; t++)
            {
                err0 += math.abs(Sk0[t] - sigma[t]) / sigma[t];
                err2 += math.abs(Sk2[t] - sigma[t]) / sigma[t];
            }
            // q=2 must be at least as accurate (allow tiny float slack). Monotone HMT behavior.
            // Measured (slow ρ=0.92): q=0 summed-rel-error ≈ 0.19 (float)/0.29 (double), q=2 ≈ 6e-5/1.6e-4
            // → dramatic, monotone improvement. Require q=2 to be at least as accurate as q=0.
            bool improved = err2 <= err0 + (fProxy)1E-4f;
            if (!improved) Record(err2, err0, err2 - err0);
            Assert.IsTrue(improved);
            // And q=2 should actually be accurate (each σ within ~2% on this slow spectrum).
            for (int t = 0; t < k; t++)
                AssertLE(math.abs(Sk2[t] - sigma[t]) / sigma[t], (fProxy)0.02f);

            arena.Dispose();
        }

        // Orthonormality of returned Uk/Vk columns on a known-Σ matrix (clustered + decaying).
        void RandSvdOrthonormal_Known_140x40()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 140, n = 40, k = 7;

            var sigma = arena.fProxyVec(n);
            // clustered top then decay: [20,20,20, 8,5,3,2, then geometric tail]
            double tail = 2.0;
            for (int i = 0; i < n; i++)
            {
                double sg;
                if (i < 3) sg = 20.0;
                else if (i == 3) sg = 8.0;
                else if (i == 4) sg = 5.0;
                else if (i == 5) sg = 3.0;
                else if (i == 6) sg = 2.0;
                else { tail *= 0.8; sg = tail; }
                sigma[i] = (fProxy)sg;
            }

            var A = arena.fProxyMat(m, n);
            var rng = new Unity.Mathematics.Random(0x5EED1234u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            bool ok = SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9ABCDEF0u, 75);
            Assert.IsTrue(ok);

            AssertOrthoCols(in Uk, m, k, (fProxy)1E-3f);
            AssertOrthoCols(in Vk, n, k, (fProxy)1E-3f);

            // Sk descending and the well-separated values (indices 3..6) within 5% relative.
            for (int t = 1; t < k; t++) AssertGE(Sk[t - 1] + (fProxy)1E-4f * (sigma[0] + (fProxy)1), Sk[t]);
            for (int t = 3; t < k; t++)
                AssertLE(math.abs(Sk[t] - sigma[t]) / sigma[t], (fProxy)0.05f);

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void RandomizedTests(TestJob.TestType type)
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
    public void RandomizedThrowsOnBadK()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var Uk = arena.fProxyMat(6, 5);
        var Sk = arena.fProxyVec(5);
        var Vk = arena.fProxyMat(4, 5);
        Assert.Catch<ArgumentException>(() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, 5)); // k=5 > n=4
        arena.Dispose();
    }

    [Test]
    public void RandomizedThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 5);
        var Uk = arena.fProxyMat(3, 2);
        var Sk = arena.fProxyVec(2);
        var Vk = arena.fProxyMat(5, 2);
        Assert.Catch<ArgumentException>(() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, 2));
        arena.Dispose();
    }
}
