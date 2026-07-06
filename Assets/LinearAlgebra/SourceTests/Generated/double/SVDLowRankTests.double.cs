using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for SVD.truncated / SVD.lowRankApprox. The spectrum from values is the oracle:
// truncated Sk must equal the leading full singular values; Uk/Vk must be orthonormal; and the
// rank-k approximation's Frobenius error must equal the spectral tail sqrt(Σ_{i>=k} σ_i²)
// (Eckart-Young). lowRankApprox must agree with Uk diag(Sk) Vkᵀ.
public class doubleSVDLowRankTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            RandomTall12x5,
            RandomSquare8,
            LowRank10x6r3,
            GklVsThin_40x12,
            GklExactLowRank_24x8r4,
            GklReconBound_30x10,
            GklOrthonormal_20x7,
            // True-truncation (p < n) tests — these exercise the regime GKL exists for
            GklTruncated_80x40_k3,           // n=40, k=3, os=4  → p=7
            GklTruncated_120x30_k5,          // n=30, k=5, os=4  → p=9
            GklClusteredSpectrum_50x30,      // clustered σ=[10,10,10,3,…], k=3, os=4 → p=7
            GklIllConditioned_60x20,         // cond~1e3, k=3, os=4 → p=7
            GklConvergedFalse_RankDeficient, // rank-3 A, k=5 → tail σ near-zero, graceful handling
            GklConvergedFalse_TooFewSteps,   // oversample=0 → p=k → betaLast large → converged=false
            GklConvergedFalse_MaxIter1,      // maxIter=1 → inner thin fails → converged=false
            // --- known-Σ (randsvd, Higham Test Matrix Toolbox) truncation in the p<n regime ---
            GklGeometricTrunc_80x30,         // geometric Σ=ρ^i (ρ=0.5), k=3, p=23<30
            GklFlatCliffTrunc_70x25,         // Σ=[100,80,60,1e-3,…], k=3, p=9<25
            GklOneSmallTrunc_50x20,          // Σ=[1,…,1,1e-4] (κ=1e4), k=3 inside flat top, full Krylov
            GklClusterProjector_50x24,       // cluster Σ=[10,10,10,…], k=3 → rank-3 PROJECTOR matches oracle
            // --- partial reorthogonalization (de74c48): partial≡full + no ghost singular values ---
            PartialVsFull_Geometric_80x30,   // partialReorth true vs false on geometric Σ, both Eckart-Young
            PartialVsFull_FlatCliff_70x25,   // partialReorth true vs false on flat-then-cliff Σ
            NoGhost_Geometric_80x30,         // every returned σ (partial) matches SOME true σ (k=3,4,5)
            PartialOrthonormal_70x25,        // UkᵀUk≈I, VkᵀVk≈I under partial reorth (broken recurrence shows here)
            PartialClustered_50x30,          // stress: tight σ cluster (hardest case for reorth), p=15<30
            PartialIllConditioned_60x20,     // stress: κ≈1e4 one-small spectrum, p=7<20
            PartialLargeP_120x60,            // stress: k=15 on 120×60 → p=35<60, reorth fires repeatedly
            PartialCloseClusterSeeds_64x24,  // stress: close-but-resolvable top cluster × 8 seeds → no ghost
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RandomTall12x5:               RandomTall12x5();               break;
                case TestType.RandomSquare8:                RandomSquare8();                break;
                case TestType.LowRank10x6r3:               LowRank10x6r3();                break;
                case TestType.GklVsThin_40x12:              GklVsThin_40x12();              break;
                case TestType.GklExactLowRank_24x8r4:      GklExactLowRank_24x8r4();       break;
                case TestType.GklReconBound_30x10:         GklReconBound_30x10();          break;
                case TestType.GklOrthonormal_20x7:         GklOrthonormal_20x7();          break;
                case TestType.GklTruncated_80x40_k3:       GklTruncated_80x40_k3();        break;
                case TestType.GklTruncated_120x30_k5:      GklTruncated_120x30_k5();       break;
                case TestType.GklClusteredSpectrum_50x30:  GklClusteredSpectrum_50x30();   break;
                case TestType.GklIllConditioned_60x20:     GklIllConditioned_60x20();      break;
                case TestType.GklConvergedFalse_RankDeficient: GklConvergedFalse_RankDeficient(); break;
                case TestType.GklConvergedFalse_TooFewSteps:   GklConvergedFalse_TooFewSteps();   break;
                case TestType.GklConvergedFalse_MaxIter1:  GklConvergedFalse_MaxIter1();   break;
                case TestType.GklGeometricTrunc_80x30:     GklGeometricTrunc_80x30();      break;
                case TestType.GklFlatCliffTrunc_70x25:     GklFlatCliffTrunc_70x25();      break;
                case TestType.GklOneSmallTrunc_50x20:      GklOneSmallTrunc_50x20();       break;
                case TestType.GklClusterProjector_50x24:   GklClusterProjector_50x24();    break;
                case TestType.PartialVsFull_Geometric_80x30: PartialVsFull_Geometric_80x30(); break;
                case TestType.PartialVsFull_FlatCliff_70x25: PartialVsFull_FlatCliff_70x25(); break;
                case TestType.NoGhost_Geometric_80x30:     NoGhost_Geometric_80x30();      break;
                case TestType.PartialOrthonormal_70x25:    PartialOrthonormal_70x25();     break;
                case TestType.PartialClustered_50x30:      PartialClustered_50x30();       break;
                case TestType.PartialIllConditioned_60x20: PartialIllConditioned_60x20();  break;
                case TestType.PartialLargeP_120x60:        PartialLargeP_120x60();         break;
                case TestType.PartialCloseClusterSeeds_64x24: PartialCloseClusterSeeds_64x24(); break;
            }
        }

        // randsvd (Higham Test Matrix Toolbox): A = U·diag(σ)·Vᵀ, Haar U(m×m)/V(n×n), prescribed σ
        // (length n). The exact singular values are KNOWN. Temp bases disposed here.
        void BuildRandSvd(ref Unity.Mathematics.Random rng, int m, int n, in doubleN sigma, ref doubleMxN A)
        {
            var U = new doubleMxN(m, m, Allocator.Temp, false);
            var V = new doubleMxN(n, n, Allocator.Temp, false);
            Rand.orthogonalInPlace(ref rng, ref U);
            Rand.orthogonalInPlace(ref rng, ref V);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)U[i, t] * (double)sigma[t] * (double)V[j, t];
                    A[i, j] = (double)acc;
                }
            U.Dispose(); V.Dispose();
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

        // Run all truncated/low-rank checks for one matrix at one k. fullS = full spectrum (length n).
        void CheckAtK(in doubleMxN A, in doubleN fullS, int k, double normA2, ref Arena arena)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            double tol = (double)1E-3f * (normA2 + (double)1);

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k);
            Assert.IsTrue(cT);

            // Sk equals the leading full singular values.
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], fullS[t], tol);

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            // Rank-k approximation.
            var Ak = arena.doubleMat(m, n);
            bool cL = SVD.lowRankApprox(in A, ref Ak, k);
            Assert.IsTrue(cL);

            // Frobenius error squared == spectral tail Σ_{i>=k} σ_i² (Eckart-Young).
            double efro2 = (double)0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double d = A[i, j] - Ak[i, j];
                    efro2 += d * d;
                }
            double tail = (double)0;
            for (int i = k; i < n; i++) tail += fullS[i] * fullS[i];
            AssertLE(math.abs(efro2 - tail), tol);

            // lowRankApprox agrees with Uk diag(Sk) Vkᵀ.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double recon = (double)0;
                    for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                    AssertClose(recon, Ak[i, j], tol);
                }
        }

        // Compute the full spectrum (oracle) and ||A||_F² once for a matrix.
        doubleN Spectrum(in doubleMxN A, ref Arena arena, out double normA2)
        {
            int n = A.N_Cols;
            var fullS = arena.doubleVec(n);
            SVD.values(in A, ref fullS);
            normA2 = (double)0;
            for (int i = 0; i < n; i++) normA2 += fullS[i] * fullS[i];
            return fullS;
        }

        void RandomTall12x5()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleRandomMat(12, 5, (double)(-2f), (double)2f, 555111);
            var fullS = Spectrum(in A, ref arena, out double normA2);
            CheckAtK(in A, in fullS, 1, normA2, ref arena);
            CheckAtK(in A, in fullS, 2, normA2, ref arena);
            CheckAtK(in A, in fullS, 3, normA2, ref arena);
            CheckAtK(in A, in fullS, 5, normA2, ref arena);
            arena.Dispose();
        }

        void RandomSquare8()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleRandomMat(8, 8, (double)(-3f), (double)3f, 909090);
            var fullS = Spectrum(in A, ref arena, out double normA2);
            CheckAtK(in A, in fullS, 1, normA2, ref arena);
            CheckAtK(in A, in fullS, 4, normA2, ref arena);
            CheckAtK(in A, in fullS, 8, normA2, ref arena);
            arena.Dispose();
        }

        void LowRank10x6r3()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 6, r = 3;
            var B = arena.doubleRandomMat(m, r, (double)(-2f), (double)2f, 121212);
            var C = arena.doubleRandomMat(r, n, (double)(-2f), (double)2f, 343434);
            var A = Blas.dot(B, C);   // rank 3
            var fullS = Spectrum(in A, ref arena, out double normA2);
            // k=3 captures all energy (tail ~ 0); k=2 leaves σ_2. k > rank(A) is not tested here —
            // GKL correctly signals converged=false for k > rank (rank-deficiency detected via Krylov
            // exhaustion), so CheckAtK (which asserts cT=true) does not apply.
            CheckAtK(in A, in fullS, 2, normA2, ref arena);
            CheckAtK(in A, in fullS, 3, normA2, ref arena);
            arena.Dispose();
        }

        // ---- GKL oracle tests ----

        void GklVsThin_40x12()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 40, n = 12;
            var A = arena.doubleRandomMat(m, n, (double)(-2f), (double)2f, 777111);

            // Full thin SVD as oracle
            var Ufull = arena.doubleMat(m, n);
            var Sfull = arena.doubleVec(n);
            var Vfull = arena.doubleMat(n, n);
            bool okFull = SVD.thin(in A, ref Ufull, ref Sfull, ref Vfull);
            Assert.IsTrue(okFull);

            // k=3 with oversample=9 → p=min(12,12)=12 (full Krylov, exact result)
            CheckGklVsThin(in A, in Ufull, in Sfull, in Vfull, 3, 9, 0x1234ABCDu, m, n, ref arena);
            // k=5 with oversample=7 → p=12 (full Krylov)
            CheckGklVsThin(in A, in Ufull, in Sfull, in Vfull, 5, 7, 0xDEADBEEFu, m, n, ref arena);

            arena.Dispose();
        }

        void CheckGklVsThin(in doubleMxN A, in doubleMxN Ufull, in doubleN Sfull, in doubleMxN Vfull,
                             int k, int oversample, uint seed, int m, int n, ref Arena arena)
        {
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75);
            Assert.IsTrue(cT);

            double svTol = (double)8 * Consts.doubleSqrtEps * (Sfull[0] + (double)1);

            for (int t = 0; t < k; t++)
            {
                // Singular value agreement
                AssertClose(Sk[t], Sfull[t], svTol);

                // Left singular vector agreement (sign-insensitive)
                double dotU = (double)0;
                for (int i = 0; i < m; i++) dotU += Uk[i, t] * Ufull[i, t];
                double absDotU = math.abs(dotU);
                AssertClose(absDotU, (double)1, (double)1E-2f);

                // Right singular vector agreement (sign-insensitive)
                double dotV = (double)0;
                for (int i = 0; i < n; i++) dotV += Vk[i, t] * Vfull[i, t];
                double absDotV = math.abs(dotV);
                AssertClose(absDotV, (double)1, (double)1E-2f);
            }
        }

        void GklExactLowRank_24x8r4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 24, n = 8, r = 4;
            var B = arena.doubleRandomMat(m, r, (double)(-2f), (double)2f, 8881);
            var C = arena.doubleRandomMat(r, n, (double)(-2f), (double)2f, 9992);
            var A = Blas.dot(B, C);   // exactly rank 4

            // Full spectrum oracle
            var Sfull = arena.doubleVec(n);
            SVD.values(in A, ref Sfull);

            int k = 4;
            int oversample = 4;   // p = min(8,8) = 8 = n: full Krylov
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xCAFEBABEu, 75);
            Assert.IsTrue(cT);

            double svTol = (double)8 * Consts.doubleSqrtEps * (Sfull[0] + (double)1);

            // Top r=4 singular values must match
            for (int t = 0; t < r; t++)
                AssertClose(Sk[t], Sfull[t], svTol);

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            arena.Dispose();
        }

        void GklReconBound_30x10()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 30, n = 10;
            var A = arena.doubleRandomMat(m, n, (double)(-3f), (double)3f, 313131);

            // Full spectrum oracle
            var Sfull = arena.doubleVec(n);
            SVD.values(in A, ref Sfull);

            int k = 4;
            int oversample = 6;  // p = min(10,10) = 10 = n: full Krylov
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xBEEFCAFEu, 75);
            Assert.IsTrue(cT);

            // Reconstruction error squared
            double errFro2 = (double)0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double recon = (double)0;
                    for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                    double d = A[i, j] - recon;
                    errFro2 += d * d;
                }

            // Eckart-Young tail
            double tail = (double)0;
            for (int i = k; i < n; i++) tail += Sfull[i] * Sfull[i];

            // Allow tolerance relative to tail magnitude
            double tol = (double)1E-2f * (tail + (double)1);
            AssertLE(math.abs(errFro2 - tail), tol);

            arena.Dispose();
        }

        void GklOrthonormal_20x7()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 20, n = 7;
            var A = arena.doubleRandomMat(m, n, (double)(-4f), (double)4f, 202020);

            int k = 4;
            int oversample = 3;  // p = min(7,7) = 7 = n: full Krylov
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x12345678u, 75);
            Assert.IsTrue(cT);

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            // Sk sorted descending and non-negative
            for (int t = 0; t < k; t++)
            {
                bool nonNeg = Sk[t] >= (double)0;
                if (!nonNeg) Record(Sk[t], (double)0, Sk[t]);
                Assert.IsTrue(nonNeg);
                if (t > 0)
                {
                    bool desc = Sk[t] <= Sk[t-1] + (double)1E-6f;
                    if (!desc) Record(Sk[t], Sk[t-1], Sk[t] - Sk[t-1]);
                    Assert.IsTrue(desc);
                }
            }

            arena.Dispose();
        }

        // ---- TRUE TRUNCATION tests: p < n (the regime GKL exists for) ----
        //
        // For converged=true to be reliable, these tests use matrices with PRESCRIBED spectra
        // (large clear spectral gap between σ_k and σ_{k+1}). With gap ratio γ = σ_{k+1}/σ_k:
        // residual after p steps ≈ σ_0·γ^(p-k), which must be < 8·√ε·σ_0. With γ small and
        // p-k ≥ 4, this is easily satisfied. Random matrices have no gap → converged is unreliable.

        // Build A (m×n) as U·diag(S)·Vᵀ where U (m×m), V (n×n) are Haar-random (Temp) and
        // S is prescribed inline in the caller's triple loop. U and V are disposed by the caller.
        // Returns Ularge (m×m) and Vmat (n×n) in the Temp allocator — caller must Dispose.
        void BuildOrthoBases(ref Unity.Mathematics.Random rng, int m, int n,
                             out doubleMxN Ularge, out doubleMxN Vmat)
        {
            Ularge = new doubleMxN(m, m, Allocator.Temp, false);
            Vmat   = new doubleMxN(n, n, Allocator.Temp, false);
            Rand.orthogonalInPlace(ref rng, ref Ularge);
            Rand.orthogonalInPlace(ref rng, ref Vmat);
        }

        // Check GKL top-k against thin oracle. Asserts converged=true, σ within svTol of oracle,
        // |dot| for singular vectors ≥ 1-vecTol, orthonormality within orthoTol.
        void CheckTruncVsThin(in doubleMxN A, int k, int oversample, uint seed, int m, int n,
                              double svTol, double vecTol, double orthoTol, ref Arena arena)
        {
            var Ufull = arena.doubleMat(m, n);
            var Sfull = arena.doubleVec(n);
            var Vfull = arena.doubleMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref Ufull, ref Sfull, ref Vfull));

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75);
            Assert.IsTrue(cT);  // clear spectral gap guarantees convergence

            for (int t = 0; t < k; t++)
            {
                AssertClose(Sk[t], Sfull[t], svTol);
                double dU = (double)0; for (int i = 0; i < m; i++) dU += Uk[i,t]*Ufull[i,t];
                AssertClose(math.abs(dU), (double)1, vecTol);
                double dV = (double)0; for (int i = 0; i < n; i++) dV += Vk[i,t]*Vfull[i,t];
                AssertClose(math.abs(dV), (double)1, vecTol);
            }
            AssertOrthoCols(in Uk, m, k, orthoTol);
            AssertOrthoCols(in Vk, n, k, orthoTol);
        }

        // Test 1: 80×40, k=3, oversample=4 → p=min(7,40)=7 ≪ n=40. Genuine truncation.
        // Prescribed: S=[100,80,60,0.1,…,0.1] → gap σ₃/σ₂≈1.67e-3 → residual after 4 extra steps ≈ 2e-11
        void GklTruncated_80x40_k3()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 80, n = 40;
            // p = min(3+4, 40) = 7. Confirmed p=7 < n=40.
            var rng = new Unity.Mathematics.Random(0xABCDEF01u);
            BuildOrthoBases(ref rng, m, n, out var UL, out var VL);
            var A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double s = (t == 0) ? 100.0 : (t == 1) ? 80.0 : (t == 2) ? 60.0 : 0.1;
                        acc += (double)UL[i,t] * s * (double)VL[j,t];
                    }
                    A[i,j] = (double)acc;
                }
            UL.Dispose(); VL.Dispose();
            CheckTruncVsThin(in A, 3, 4, 0x11223344u, m, n,
                             (double)1f, (double)0.01f, (double)1E-3f, ref arena);
            arena.Dispose();
        }

        // Test 2: 120×30, k=5, oversample=4 → p=min(9,30)=9 ≪ n=30. Genuine truncation.
        // Prescribed: S=[100,80,60,40,20,0.1,…,0.1] → gap σ₅/σ₄≈5e-3 → residual after 4 extra steps ≈ 6e-10
        void GklTruncated_120x30_k5()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 120, n = 30;
            // p = min(5+4, 30) = 9. Confirmed p=9 < n=30.
            var rng = new Unity.Mathematics.Random(0xDEAD5678u);
            BuildOrthoBases(ref rng, m, n, out var UL, out var VL);
            var A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double s = (t==0)?100.0:(t==1)?80.0:(t==2)?60.0:(t==3)?40.0:(t==4)?20.0:0.1;
                        acc += (double)UL[i,t] * s * (double)VL[j,t];
                    }
                    A[i,j] = (double)acc;
                }
            UL.Dispose(); VL.Dispose();
            CheckTruncVsThin(in A, 5, 4, 0x87654321u, m, n,
                             (double)1f, (double)0.01f, (double)1E-3f, ref arena);
            arena.Dispose();
        }

        // Test 3: clustered spectrum σ₀=σ₁=σ₂=10, rest decay. k=3, oversample=12 → p=15 < n=30.
        // Oversample 12 gives 12 extra Lanczos steps beyond k=3.
        // Gap σ₃/σ₂=3/10=0.3; residual after 12 extra steps ≈ 10·0.3^12 ≈ 5e-6 → converged=true.
        // Stresses DGKS reorthogonalization on the degenerate cluster.
        void GklClusteredSpectrum_50x30()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 50, n = 30;
            // k=3, oversample=12 → p = min(3+12, 30) = 15. Confirmed p=15 < n=30.
            var rng = new Unity.Mathematics.Random(0xC1A3B1EDu);
            BuildOrthoBases(ref rng, m, n, out var UL, out var VL);
            var A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double sig;
                        if      (t < 3)  sig = 10.0;
                        else if (t == 3) sig = 3.0;
                        else if (t == 4) sig = 2.0;
                        else if (t == 5) sig = 1.5;
                        else if (t == 6) sig = 1.0;
                        else             sig = 0.5 / (t - 5);
                        acc += (double)UL[i,t] * sig * (double)VL[j,t];
                    }
                    A[i,j] = (double)acc;
                }
            UL.Dispose(); VL.Dispose();

            int k = 3, oversample = 12;  // p=15 < n=30
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xF00DB4BEu, 75);
            Assert.IsTrue(cT);

            // All three recovered σ must be close to 10 (the cluster value)
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], (double)10, (double)0.05f);

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            // Compare against thin oracle (all three should give ≈10)
            var Uf = arena.doubleMat(m, n);
            var Sf = arena.doubleVec(n);
            var Vf = arena.doubleMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref Uf, ref Sf, ref Vf));
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], Sf[t], (double)0.05f);

            arena.Dispose();
        }

        // Test 4: ill-conditioned matrix (κ~1e4), k=3, oversample=4 → p=7 < n=20.
        // GKL operates on A directly (avoids κ² of AᵀA), so large condition number doesn't hurt accuracy.
        // Prescribed: S=[1000,500,200,0.1,…,0.1] → κ=1e4, gap σ₃/σ₂=0.1/200=5e-4
        //             → residual after 4 extra steps ≈ 1000·(5e-4)^4 ≈ 6e-8 → converged=true.
        void GklIllConditioned_60x20()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 60, n = 20;
            // p = min(3+4, 20) = 7. Confirmed p=7 < n=20.
            var rng = new Unity.Mathematics.Random(0x6789ABCDu);
            BuildOrthoBases(ref rng, m, n, out var UL, out var VL);
            var A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double s = (t==0)?1000.0:(t==1)?500.0:(t==2)?200.0:0.1;
                        acc += (double)UL[i,t] * s * (double)VL[j,t];
                    }
                    A[i,j] = (double)acc;
                }
            UL.Dispose(); VL.Dispose();
            CheckTruncVsThin(in A, 3, 4, 0xFEEDC0DEu, m, n,
                             (double)5f, (double)0.01f, (double)1E-3f, ref arena);
            arena.Dispose();
        }

        // Test 5: rank-3 matrix, k=5. The tail σ (indices 3,4) should be near-zero;
        // the algorithm handles rank-deficiency gracefully (correct top-r, tiny tail).
        // Whether converged is true/false depends on floating-point; we check OUTPUT quality.
        // Also exercises FIX 2 (alpha-breakdown betaLast=0) when it triggers.
        void GklConvergedFalse_RankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 30, n = 20, r = 3;
            // k=5, oversample=5 → p = min(10, 20) = 10 < n=20.
            var B = arena.doubleRandomMat(m, r, (double)(-2f), (double)2f, 0xBADC0DE0u);
            var C = arena.doubleRandomMat(r, n, (double)(-2f), (double)2f, 0xBADC0DE1u);
            var A = Blas.dot(B, C);  // exactly rank 3

            // True top-r singular values (oracle)
            var Sfull = arena.doubleVec(n);
            SVD.values(in A, ref Sfull);

            int k = 5, oversample = 5;
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo _cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x99887766u, 75);

            // Top-r singular values must be captured correctly
            double svTol = (double)1E-2f * (Sfull[0] + (double)1);
            for (int t = 0; t < r; t++)
                AssertClose(Sk[t], Sfull[t], svTol);

            // Tail σ (indices r..k-1) must be much smaller than Sk[0]:
            // the rank-3 null space dimensions produce near-zero Ritz values.
            double tailTol = (double)1E-3f * Sk[0];
            for (int t = r; t < k; t++)
                AssertLE(Sk[t], tailTol);

            arena.Dispose();
        }

        // Test 6: oversample=0 → p=k=3 (NO extra Lanczos steps). For any non-trivial
        // large matrix, betaLast·|U[p-1,t]| / σ₀ ≫ 8·√ε → converged=false.
        // Directly exercises FIX 1's residual check (the path that was computing V instead of U).
        void GklConvergedFalse_TooFewSteps()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 80, n = 30;
            // k=3, oversample=0 → p = min(3+0, 30) = 3. p=k with ZERO oversampling.
            // betaLast·|U[2,t]|/σ₀ ≈ σ₄/σ₀ ≈ 0.8 ≫ 8·√ε → converged=false guaranteed.
            var A = arena.doubleRandomMat(m, n, (double)(-2f), (double)2f, 0x55AA77BBu);

            int k = 3, oversample = 0;
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xAABBCCDDu, 75);

            // With p=k and no oversampling the residual is far from zero → converged=false
            Assert.IsFalse(cT);

            arena.Dispose();
        }

        // Test 7: maxIter=1 forces the inner bidiagonal QR (on p×p B) to not converge.
        // Exercises the thin-failure branch → converged=false even with p < n.
        void GklConvergedFalse_MaxIter1()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 40, n = 20;
            // k=5, oversample=5 → p = min(10, 20) = 10. Confirmed p=10 < n=20.
            // maxIter=1: the inner 10×10 bidiagonal QR almost certainly needs > 1 sweep.
            var A = arena.doubleRandomMat(m, n, (double)(-3f), (double)3f, 0xDEADBEEFu);

            int k = 5, oversample = 5;
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x12AB34CDu, 1);

            // With maxIter=1 the inner bidiagonal QR will not converge for a non-trivial matrix
            Assert.IsFalse(cT);

            arena.Dispose();
        }

        // ============================================================================================
        // Known-Σ (randsvd) truncation — exact-converging GKL must return the EXACT top-k of a matrix
        // whose singular values are prescribed. svTol scales with the type via Consts.doubleSqrtEps
        // (float ~3.45e-4, double ~1.49e-8) so the same assertion holds across float/double expansions.
        // ============================================================================================

        // Geometric Σ_i = ρ^i (ρ=0.5). k=3, oversample=20 → p=min(23,30)=23 < n=30: genuine truncation.
        // 20 extra Lanczos steps drive the top-3 Ritz values to machine precision.
        void GklGeometricTrunc_80x30()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 80, n = 30, k = 3;

            var sigma = arena.doubleVec(n);
            double s = 1.0;
            for (int i = 0; i < n; i++) { sigma[i] = (double)s; s *= 0.5; }   // 1, 0.5, 0.25, ...

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0x10AD5EEDu);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, 20, 0x1111AAAAu, 75);
            Assert.IsTrue(cT);

            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);
            for (int t = 0; t < k; t++) AssertClose(Sk[t], sigma[t], svTol);
            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            arena.Dispose();
        }

        // Flat-then-cliff Σ=[100,80,60, 1e-3,…]. k=3, oversample=6 → p=9 < n=25. Huge spectral gap at
        // index 3 → top-3 converge almost immediately; compare values + vectors against thin oracle.
        void GklFlatCliffTrunc_70x25()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 70, n = 25;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i == 0) ? (double)100 : (i == 1) ? (double)80 : (i == 2) ? (double)60 : (double)1E-3f;

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xC11FF00Du);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            // svTol scaled to σ_max=100; vector + orthonormality tolerances as in the other GKL tests.
            CheckTruncVsThin(in A, 3, 6, 0x2222BBBBu, m, n,
                             (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1),
                             (double)0.01f, (double)1E-3f, ref arena);
            arena.Dispose();
        }

        // One-small Σ=[1,…,1,1e-4] (κ=1e4). Targeting k=3 inside the FLAT top block: the individual
        // singular vectors are non-unique, so we only assert the recovered VALUES are all ≈1 and the
        // columns are orthonormal. oversample=17 → p=min(20,20)=20=n (full Krylov) → exact, converged.
        void GklOneSmallTrunc_50x20()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 50, n = 20, k = 3;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++) sigma[i] = (i == n - 1) ? (double)1E-4f : (double)1;

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0x0E5A3411u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, 17, 0x3333CCCCu, 75);
            Assert.IsTrue(cT);

            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);
            for (int t = 0; t < k; t++) AssertClose(Sk[t], (double)1, svTol);
            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            arena.Dispose();
        }

        // Clustered Σ=[10,10,10, 3,2,1.5,1, …]: the top-3 are a degenerate cluster, so individual u/v
        // are non-unique but the rank-3 SUBSPACE is well-defined. Assert the rank-3 projector
        // Pk = Uk·Ukᵀ matches the thin oracle's top-3 projector (and likewise for V), plus Sk≈10.
        // k=3, oversample=15 → p=18 < n=24. Gap 3/10=0.3 → residual ≈10·0.3^15 ≈1.4e-6 → converged.
        void GklClusterProjector_50x24()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 50, n = 24, k = 3;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
            {
                double sg;
                if (i < 3) sg = 10.0;
                else if (i == 3) sg = 3.0;
                else if (i == 4) sg = 2.0;
                else if (i == 5) sg = 1.5;
                else if (i == 6) sg = 1.0;
                else sg = 0.5 / (i - 5);
                sigma[i] = (double)sg;
            }

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xC1051E33u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, 15, 0x4444DDDDu, 75);
            Assert.IsTrue(cT);

            for (int t = 0; t < k; t++) AssertClose(Sk[t], (double)10, (double)0.05f);

            // thin oracle; compare the rank-3 projectors Σ_t u_t u_tᵀ (subspace, not per-vector).
            var Uf = arena.doubleMat(m, n);
            var Sf = arena.doubleVec(n);
            var Vf = arena.doubleMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref Uf, ref Sf, ref Vf));

            AssertProjectorMatch(in Uk, in Uf, m, k, (double)1E-2f);
            AssertProjectorMatch(in Vk, in Vf, n, k, (double)1E-2f);

            arena.Dispose();
        }

        // ‖B Bᵀ (first k cols) − Ref Refᵀ (first k cols)‖_max over the rank-k projectors.
        void AssertProjectorMatch(in doubleMxN B, in doubleMxN Ref, int rows, int k, double tol)
        {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < rows; j++)
                {
                    double pb = (double)0, pr = (double)0;
                    for (int t = 0; t < k; t++) { pb += B[i, t] * B[j, t]; pr += Ref[i, t] * Ref[j, t]; }
                    AssertClose(pb, pr, tol);
                }
        }

        // ====================================================================================
        // PARTIAL REORTHOGONALIZATION (de74c48). The bool partialReorth on the core overload
        // selects the ω-recurrence/ELR path (true, default) vs the original full-DGKS path
        // (false). Both must return the EXACT top-k triplets with no spurious ("ghost")
        // singular values and orthonormal factors. All matrices below use genuine p < n
        // truncation (NOT p == n) so the Lanczos recurrence is actually exercised.
        // Tolerances reuse the GklTruncated style: svTol = 8·√ε·(σ₀+1), orthoTol = 1e-3.
        // ====================================================================================

        // Run truncated with an explicit partialReorth flag against a prescribed (oracle) Σ.
        // Asserts converged, per-index σ match, orthonormal Uk/Vk, and the Eckart-Young optimum
        // ‖A − UkΣkVkᵀ‖_F² == Σ_{i≥k} σ_i². Recovered Sk are copied into SkOut for cross-compare.
        void RunTruncWithReorth(in doubleMxN A, in doubleN sigmaTrue, int k, int oversample, uint seed,
                                bool partialReorth, ref doubleN SkOut, int m, int n,
                                double svTol, ref Arena arena)
        {
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, partialReorth);
            Assert.IsTrue(cT);

            for (int t = 0; t < k; t++) { AssertClose(Sk[t], sigmaTrue[t], svTol); SkOut[t] = Sk[t]; }

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            // Eckart-Young: rank-k Frobenius error squared equals the spectral tail.
            double err2 = (double)0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double recon = (double)0;
                    for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                    double d = A[i, j] - recon;
                    err2 += d * d;
                }
            double tail = (double)0;
            for (int i = k; i < n; i++) tail += sigmaTrue[i] * sigmaTrue[i];
            AssertLE(math.abs(err2 - tail), (double)1E-2f * (tail + (double)1));
        }

        // partialReorth=true: assert converged, per-index σ match to oracle, no ghost (each
        // returned σ_t is within svTol of SOME true σ), and orthonormal Uk/Vk.
        void CheckPartialReorth(in doubleMxN A, in doubleN sigmaTrue, int k, int oversample, uint seed,
                                int m, int n, double svTol, ref Arena arena)
        {
            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, true);
            Assert.IsTrue(cT);

            for (int t = 0; t < k; t++) AssertClose(Sk[t], sigmaTrue[t], svTol);

            // No ghost: every recovered σ matches some true σ.
            for (int t = 0; t < k; t++)
            {
                double best = math.abs(Sk[t] - sigmaTrue[0]);
                for (int i = 1; i < n; i++)
                {
                    double d = math.abs(Sk[t] - sigmaTrue[i]);
                    if (d < best) best = d;
                }
                AssertLE(best, svTol);
            }

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);
        }

        // Test A: partial≡full on a geometric spectrum Σ_i = 0.5^i. k=3, oversample=20 → p=23<30.
        // Both paths must recover the top-3 to oracle tol AND agree with each other; both hit the
        // Eckart-Young optimum. (Mirrors the proven GklGeometricTrunc_80x30 oversampling.)
        void PartialVsFull_Geometric_80x30()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 80, n = 30, k = 3;

            var sigma = arena.doubleVec(n);
            double s = 1.0;
            for (int i = 0; i < n; i++) { sigma[i] = (double)s; s *= 0.5; }

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0x5A1B0C0Du);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);

            var SkP = arena.doubleVec(k);
            var SkF = arena.doubleVec(k);
            RunTruncWithReorth(in A, in sigma, k, 20, 0xA11CE001u, true,  ref SkP, m, n, svTol, ref arena);
            RunTruncWithReorth(in A, in sigma, k, 20, 0xA11CE001u, false, ref SkF, m, n, svTol, ref arena);

            // Partial and full agree to the SAME tolerance.
            for (int t = 0; t < k; t++) AssertClose(SkP[t], SkF[t], svTol);

            arena.Dispose();
        }

        // Test B: partial≡full on a flat-then-cliff spectrum Σ=[100,80,60,1e-3,…]. k=3, oversample=6
        // → p=9<25. Huge gap at index 3 → both paths converge immediately and must agree.
        void PartialVsFull_FlatCliff_70x25()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 70, n = 25, k = 3;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i == 0) ? (double)100 : (i == 1) ? (double)80 : (i == 2) ? (double)60 : (double)1E-3f;

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xB22DEEF1u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);

            var SkP = arena.doubleVec(k);
            var SkF = arena.doubleVec(k);
            RunTruncWithReorth(in A, in sigma, k, 6, 0xB22DEEF1u, true,  ref SkP, m, n, svTol, ref arena);
            RunTruncWithReorth(in A, in sigma, k, 6, 0xB22DEEF1u, false, ref SkF, m, n, svTol, ref arena);

            for (int t = 0; t < k; t++) AssertClose(SkP[t], SkF[t], svTol);

            arena.Dispose();
        }

        // Test C: no ghost. Geometric Σ (distinct values) → any spurious / in-between σ would fail the
        // nearest-true-σ check. Checked at k=3,4,5, all with oversample=20 (p=23,24,25 < n=30).
        void NoGhost_Geometric_80x30()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 80, n = 30;

            var sigma = arena.doubleVec(n);
            double s = 1.0;
            for (int i = 0; i < n; i++) { sigma[i] = (double)s; s *= 0.5; }

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xC33FACE2u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);
            CheckPartialReorth(in A, in sigma, 3, 20, 0x0BADF00Du, m, n, svTol, ref arena);
            CheckPartialReorth(in A, in sigma, 4, 20, 0x0BADF00Eu, m, n, svTol, ref arena);
            CheckPartialReorth(in A, in sigma, 5, 20, 0x0BADF00Fu, m, n, svTol, ref arena);

            arena.Dispose();
        }

        // Test D: orthonormality under partial reorth. Σ=[100,80,60,40,1e-2,…], k=4, oversample=8
        // → p=12<25. A broken ω-recurrence first manifests as loss of orthonormality in Uk/Vk.
        void PartialOrthonormal_70x25()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 70, n = 25, k = 4;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i == 0) ? (double)100 : (i == 1) ? (double)80 : (i == 2) ? (double)60 :
                           (i == 3) ? (double)40 : (double)1E-2f;

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xD44B0B0Du);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            var Uk = arena.doubleMat(m, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            SVDInfo cT = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, 8, 0xD44B0B0Du, 75, true);
            Assert.IsTrue(cT);

            AssertOrthoCols(in Uk, m, k, (double)1E-3f);
            AssertOrthoCols(in Vk, n, k, (double)1E-3f);

            // Sk non-negative, descending.
            for (int t = 0; t < k; t++)
            {
                bool nonNeg = Sk[t] >= (double)0;
                if (!nonNeg) Record(Sk[t], (double)0, Sk[t]);
                Assert.IsTrue(nonNeg);
                if (t > 0)
                {
                    bool desc = Sk[t] <= Sk[t - 1] + (double)1E-4f;
                    if (!desc) Record(Sk[t], Sk[t - 1], Sk[t] - Sk[t - 1]);
                    Assert.IsTrue(desc);
                }
            }

            arena.Dispose();
        }

        // Test E (stress a): clustered spectrum σ₀=σ₁=σ₂=10 — the hardest case for reorth, where a
        // broken recurrence spawns ghost copies of the converged value. k=3, oversample=12 → p=15<30.
        void PartialClustered_50x30()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 50, n = 30, k = 3;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
            {
                double sg;
                if (i < 3) sg = 10.0;
                else if (i == 3) sg = 3.0;
                else if (i == 4) sg = 2.0;
                else if (i == 5) sg = 1.5;
                else if (i == 6) sg = 1.0;
                else sg = 0.5 / (i - 5);
                sigma[i] = (double)sg;
            }

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xE55C1A57u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            // Cluster: individual σ recovered to ~10 within 0.05 (matches GklClusteredSpectrum tol).
            CheckPartialReorth(in A, in sigma, k, 12, 0xC1057E12u, m, n, (double)0.05f, ref arena);

            arena.Dispose();
        }

        // Test E (stress b): ill-conditioned κ≈1e4, Σ=[1000,500,200,0.1,…]. k=3, oversample=4 → p=7<20.
        void PartialIllConditioned_60x20()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 60, n = 20, k = 3;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i == 0) ? (double)1000 : (i == 1) ? (double)500 : (i == 2) ? (double)200 : (double)0.1f;

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0xF66D1A60u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            // svTol=5 mirrors GklIllConditioned_60x20; distinct top values are far apart.
            CheckPartialReorth(in A, in sigma, k, 4, 0x111C0AD3u, m, n, (double)5f, ref arena);

            arena.Dispose();
        }

        // Test E (stress c): LARGE p so reorth fires repeatedly. k=15 (≈n/4) on 120×60, oversample=20
        // → p=35<60. Top-15 distinct (Σ_i = 100−2i) then a cliff to 1e-2, so the wanted block has a
        // clean gap and converges; the 35-step Lanczos run reorthogonalizes many times.
        void PartialLargeP_120x60()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 120, n = 60, k = 15;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i < 15) ? (double)(100.0 - 2.0 * i) : (double)1E-2f;

            var A = arena.doubleMat(m, n);
            var rng = new Unity.Mathematics.Random(0x12A6E057u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);

            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);
            CheckPartialReorth(in A, in sigma, k, 20, 0x1A36E099u, m, n, svTol, ref arena);

            arena.Dispose();
        }

        // Test E (stress d): close-but-RESOLVABLE top cluster σ=[10,9.7,9.4] then a clean drop, swept
        // over 8 (matrix, start-vector) seeds. Close Ritz values are exactly where Lanczos is prone to
        // emit a "ghost" (a spurious duplicate of a converged σ). The svTol = 8·√ε·(σ₀+1) is SMALLER
        // than the 0.3 intra-cluster gap, so CheckPartialReorth's per-index AssertClose(Sk[t],σ[t])
        // genuinely catches a duplicate (a ghost copy of σ₀ at index 1 fails the 9.7 match), unlike an
        // exactly-equal cluster. Recovers cleanly because the wanted block has a large gap to the tail.
        void PartialCloseClusterSeeds_64x24()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 64, n = 24, k = 3;

            var sigma = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i < 3) ? (double)(10.0 - 0.3 * i) : (double)(0.5 / (i - 1));

            var A = arena.doubleMat(m, n);
            double svTol = (double)8 * Consts.doubleSqrtEps * (sigma[0] + (double)1);

            for (uint g = 0; g < 8; g++)
            {
                var rng = new Unity.Mathematics.Random(0x5EED0001u + g * 0x9E3779B1u);
                BuildRandSvd(ref rng, m, n, in sigma, ref A);
                CheckPartialReorth(in A, in sigma, k, 12, 0x57A27000u + g, m, n, svTol, ref arena);
            }

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void LowRankTests(TestJob.TestType type)
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
    public void TruncatedThrowsOnBadK()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(6, 4);
        var Uk = arena.doubleMat(6, 5);
        var Sk = arena.doubleVec(5);
        var Vk = arena.doubleMat(4, 5);
        Assert.Catch<ArgumentException>(() => SVD.truncated(in A, ref Uk, ref Sk, ref Vk, 5)); // k=5 > n=4
        arena.Dispose();
    }

    [Test]
    public void TruncatedThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(3, 5);
        var Uk = arena.doubleMat(3, 2);
        var Sk = arena.doubleVec(2);
        var Vk = arena.doubleMat(5, 2);
        Assert.Catch<ArgumentException>(() => SVD.truncated(in A, ref Uk, ref Sk, ref Vk, 2));
        arena.Dispose();
    }

    [Test]
    public void LowRankApproxThrowsOnWrongAkShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(6, 4);
        var Ak = arena.doubleMat(4, 6);   // must be m x n = 6 x 4
        Assert.Catch<ArgumentException>(() => SVD.lowRankApprox(in A, ref Ak, 2));
        arena.Dispose();
    }
}
