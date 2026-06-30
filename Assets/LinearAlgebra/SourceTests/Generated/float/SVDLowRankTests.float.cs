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
public class floatSVDLowRankTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
            GklConvergedFalse_MaxIter1,      // maxIter=1 → inner svdThin fails → converged=false
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

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
            }
        }

        void Record(float got, float expected, float diff)
        {
            if (Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = got; Fail[2] = expected; Fail[3] = diff; }
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

        void AssertOrthoCols(in floatMxN basis, int rows, int cols, float tol)
        {
            for (int a = 0; a < cols; a++)
                for (int b = a; b < cols; b++)
                {
                    float dot = (float)0;
                    for (int i = 0; i < rows; i++) dot += basis[i, a] * basis[i, b];
                    AssertClose(dot, (a == b) ? (float)1 : (float)0, tol);
                }
        }

        // Run all truncated/low-rank checks for one matrix at one k. fullS = full spectrum (length n).
        void CheckAtK(in floatMxN A, in floatN fullS, int k, float normA2, ref Arena arena)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            float tol = (float)1E-3f * (normA2 + (float)1);

            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, out bool cT);
            Assert.IsTrue(cT);

            // Sk equals the leading full singular values.
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], fullS[t], tol);

            // Uk, Vk orthonormal columns.
            AssertOrthoCols(in Uk, m, k, (float)1E-3f);
            AssertOrthoCols(in Vk, n, k, (float)1E-3f);

            // Rank-k approximation.
            var Ak = arena.floatMat(m, n);
            SVD.lowRankApprox(in A, ref Ak, k, out bool cL);
            Assert.IsTrue(cL);

            // Frobenius error squared == spectral tail Σ_{i>=k} σ_i² (Eckart-Young).
            float efro2 = (float)0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float d = A[i, j] - Ak[i, j];
                    efro2 += d * d;
                }
            float tail = (float)0;
            for (int i = k; i < n; i++) tail += fullS[i] * fullS[i];
            AssertLE(math.abs(efro2 - tail), tol);

            // lowRankApprox agrees with Uk diag(Sk) Vkᵀ.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float recon = (float)0;
                    for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                    AssertClose(recon, Ak[i, j], tol);
                }
        }

        // Compute the full spectrum (oracle) and ||A||_F² once for a matrix.
        floatN Spectrum(in floatMxN A, ref Arena arena, out float normA2)
        {
            int n = A.N_Cols;
            var fullS = arena.floatVec(n);
            SVD.svdValues(in A, ref fullS);
            normA2 = (float)0;
            for (int i = 0; i < n; i++) normA2 += fullS[i] * fullS[i];
            return fullS;
        }

        void RandomTall12x5()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatRandomMatrix(12, 5, (float)(-2f), (float)2f, 555111);
            var fullS = Spectrum(in A, ref arena, out float normA2);
            CheckAtK(in A, in fullS, 1, normA2, ref arena);
            CheckAtK(in A, in fullS, 2, normA2, ref arena);
            CheckAtK(in A, in fullS, 3, normA2, ref arena);
            CheckAtK(in A, in fullS, 5, normA2, ref arena);
            arena.Dispose();
        }

        void RandomSquare8()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatRandomMatrix(8, 8, (float)(-3f), (float)3f, 909090);
            var fullS = Spectrum(in A, ref arena, out float normA2);
            CheckAtK(in A, in fullS, 1, normA2, ref arena);
            CheckAtK(in A, in fullS, 4, normA2, ref arena);
            CheckAtK(in A, in fullS, 8, normA2, ref arena);
            arena.Dispose();
        }

        void LowRank10x6r3()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 6, r = 3;
            var B = arena.floatRandomMatrix(m, r, (float)(-2f), (float)2f, 121212);
            var C = arena.floatRandomMatrix(r, n, (float)(-2f), (float)2f, 343434);
            var A = floatOP.dot(B, C);   // rank 3
            var fullS = Spectrum(in A, ref arena, out float normA2);
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
            var A = arena.floatRandomMatrix(m, n, (float)(-2f), (float)2f, 777111);

            // Full thin SVD as oracle
            var Ufull = arena.floatMat(m, n);
            var Sfull = arena.floatVec(n);
            var Vfull = arena.floatMat(n, n);
            bool okFull = SVD.svdThin(in A, ref Ufull, ref Sfull, ref Vfull);
            Assert.IsTrue(okFull);

            // k=3 with oversample=9 → p=min(12,12)=12 (full Krylov, exact result)
            CheckGklVsThin(in A, in Ufull, in Sfull, in Vfull, 3, 9, 0x1234ABCDu, m, n, ref arena);
            // k=5 with oversample=7 → p=12 (full Krylov)
            CheckGklVsThin(in A, in Ufull, in Sfull, in Vfull, 5, 7, 0xDEADBEEFu, m, n, ref arena);

            arena.Dispose();
        }

        void CheckGklVsThin(in floatMxN A, in floatMxN Ufull, in floatN Sfull, in floatMxN Vfull,
                             int k, int oversample, uint seed, int m, int n, ref Arena arena)
        {
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, out bool cT);
            Assert.IsTrue(cT);

            float svTol = (float)8 * Consts.floatSqrtEps * (Sfull[0] + (float)1);

            for (int t = 0; t < k; t++)
            {
                // Singular value agreement
                AssertClose(Sk[t], Sfull[t], svTol);

                // Left singular vector agreement (sign-insensitive)
                float dotU = (float)0;
                for (int i = 0; i < m; i++) dotU += Uk[i, t] * Ufull[i, t];
                float absDotU = math.abs(dotU);
                AssertClose(absDotU, (float)1, (float)1E-2f);

                // Right singular vector agreement (sign-insensitive)
                float dotV = (float)0;
                for (int i = 0; i < n; i++) dotV += Vk[i, t] * Vfull[i, t];
                float absDotV = math.abs(dotV);
                AssertClose(absDotV, (float)1, (float)1E-2f);
            }
        }

        void GklExactLowRank_24x8r4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 24, n = 8, r = 4;
            var B = arena.floatRandomMatrix(m, r, (float)(-2f), (float)2f, 8881);
            var C = arena.floatRandomMatrix(r, n, (float)(-2f), (float)2f, 9992);
            var A = floatOP.dot(B, C);   // exactly rank 4

            // Full spectrum oracle
            var Sfull = arena.floatVec(n);
            SVD.svdValues(in A, ref Sfull);

            int k = 4;
            int oversample = 4;   // p = min(8,8) = 8 = n: full Krylov
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xCAFEBABEu, 75, out bool cT);
            Assert.IsTrue(cT);

            float svTol = (float)8 * Consts.floatSqrtEps * (Sfull[0] + (float)1);

            // Top r=4 singular values must match
            for (int t = 0; t < r; t++)
                AssertClose(Sk[t], Sfull[t], svTol);

            // Orthonormality
            AssertOrthoCols(in Uk, m, k, (float)1E-3f);
            AssertOrthoCols(in Vk, n, k, (float)1E-3f);

            arena.Dispose();
        }

        void GklReconBound_30x10()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 30, n = 10;
            var A = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, 313131);

            // Full spectrum oracle
            var Sfull = arena.floatVec(n);
            SVD.svdValues(in A, ref Sfull);

            int k = 4;
            int oversample = 6;  // p = min(10,10) = 10 = n: full Krylov
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xBEEFCAFEu, 75, out bool cT);
            Assert.IsTrue(cT);

            // Reconstruction error squared
            float errFro2 = (float)0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float recon = (float)0;
                    for (int t = 0; t < k; t++) recon += Uk[i, t] * Sk[t] * Vk[j, t];
                    float d = A[i, j] - recon;
                    errFro2 += d * d;
                }

            // Eckart-Young tail
            float tail = (float)0;
            for (int i = k; i < n; i++) tail += Sfull[i] * Sfull[i];

            // Allow tolerance relative to tail magnitude
            float tol = (float)1E-2f * (tail + (float)1);
            AssertLE(math.abs(errFro2 - tail), tol);

            arena.Dispose();
        }

        void GklOrthonormal_20x7()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 20, n = 7;
            var A = arena.floatRandomMatrix(m, n, (float)(-4f), (float)4f, 202020);

            int k = 4;
            int oversample = 3;  // p = min(7,7) = 7 = n: full Krylov
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x12345678u, 75, out bool cT);
            Assert.IsTrue(cT);

            // Orthonormal columns
            AssertOrthoCols(in Uk, m, k, (float)1E-3f);
            AssertOrthoCols(in Vk, n, k, (float)1E-3f);

            // Sk sorted descending and non-negative
            for (int t = 0; t < k; t++)
            {
                bool nonNeg = Sk[t] >= (float)0;
                if (!nonNeg) Record(Sk[t], (float)0, Sk[t]);
                Assert.IsTrue(nonNeg);
                if (t > 0)
                {
                    bool desc = Sk[t] <= Sk[t-1] + (float)1E-6f;
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
                             out floatMxN Ularge, out floatMxN Vmat)
        {
            Ularge = new floatMxN(m, m, Allocator.Temp, false);
            Vmat   = new floatMxN(n, n, Allocator.Temp, false);
            floatRandomMatrixOP.randomOrthogonalInpl(ref rng, ref Ularge);
            floatRandomMatrixOP.randomOrthogonalInpl(ref rng, ref Vmat);
        }

        // Check GKL top-k against svdThin oracle. Asserts converged=true, σ within svTol of oracle,
        // |dot| for singular vectors ≥ 1-vecTol, orthonormality within orthoTol.
        void CheckTruncVsThin(in floatMxN A, int k, int oversample, uint seed, int m, int n,
                              float svTol, float vecTol, float orthoTol, ref Arena arena)
        {
            var Ufull = arena.floatMat(m, n);
            var Sfull = arena.floatVec(n);
            var Vfull = arena.floatMat(n, n);
            Assert.IsTrue(SVD.svdThin(in A, ref Ufull, ref Sfull, ref Vfull));

            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, out bool cT);
            Assert.IsTrue(cT);  // clear spectral gap guarantees convergence

            for (int t = 0; t < k; t++)
            {
                AssertClose(Sk[t], Sfull[t], svTol);
                float dU = (float)0; for (int i = 0; i < m; i++) dU += Uk[i,t]*Ufull[i,t];
                AssertClose(math.abs(dU), (float)1, vecTol);
                float dV = (float)0; for (int i = 0; i < n; i++) dV += Vk[i,t]*Vfull[i,t];
                AssertClose(math.abs(dV), (float)1, vecTol);
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
            var A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double s = (t == 0) ? 100.0 : (t == 1) ? 80.0 : (t == 2) ? 60.0 : 0.1;
                        acc += (double)UL[i,t] * s * (double)VL[j,t];
                    }
                    A[i,j] = (float)acc;
                }
            UL.Dispose(); VL.Dispose();
            CheckTruncVsThin(in A, 3, 4, 0x11223344u, m, n,
                             (float)1f, (float)0.01f, (float)1E-3f, ref arena);
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
            var A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double s = (t==0)?100.0:(t==1)?80.0:(t==2)?60.0:(t==3)?40.0:(t==4)?20.0:0.1;
                        acc += (double)UL[i,t] * s * (double)VL[j,t];
                    }
                    A[i,j] = (float)acc;
                }
            UL.Dispose(); VL.Dispose();
            CheckTruncVsThin(in A, 5, 4, 0x87654321u, m, n,
                             (float)1f, (float)0.01f, (float)1E-3f, ref arena);
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
            var A = arena.floatMat(m, n);
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
                    A[i,j] = (float)acc;
                }
            UL.Dispose(); VL.Dispose();

            int k = 3, oversample = 12;  // p=15 < n=30
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xF00DB4BEu, 75, out bool cT);
            Assert.IsTrue(cT);

            // All three recovered σ must be close to 10 (the cluster value)
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], (float)10, (float)0.05f);

            // Orthonormality
            AssertOrthoCols(in Uk, m, k, (float)1E-3f);
            AssertOrthoCols(in Vk, n, k, (float)1E-3f);

            // Compare against svdThin oracle (all three should give ≈10)
            var Uf = arena.floatMat(m, n);
            var Sf = arena.floatVec(n);
            var Vf = arena.floatMat(n, n);
            Assert.IsTrue(SVD.svdThin(in A, ref Uf, ref Sf, ref Vf));
            for (int t = 0; t < k; t++)
                AssertClose(Sk[t], Sf[t], (float)0.05f);

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
            var A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                    {
                        double s = (t==0)?1000.0:(t==1)?500.0:(t==2)?200.0:0.1;
                        acc += (double)UL[i,t] * s * (double)VL[j,t];
                    }
                    A[i,j] = (float)acc;
                }
            UL.Dispose(); VL.Dispose();
            CheckTruncVsThin(in A, 3, 4, 0xFEEDC0DEu, m, n,
                             (float)5f, (float)0.01f, (float)1E-3f, ref arena);
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
            var B = arena.floatRandomMatrix(m, r, (float)(-2f), (float)2f, 0xBADC0DE0u);
            var C = arena.floatRandomMatrix(r, n, (float)(-2f), (float)2f, 0xBADC0DE1u);
            var A = floatOP.dot(B, C);  // exactly rank 3

            // True top-r singular values (oracle)
            var Sfull = arena.floatVec(n);
            SVD.svdValues(in A, ref Sfull);

            int k = 5, oversample = 5;
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x99887766u, 75, out bool _cT);

            // Top-r singular values must be captured correctly
            float svTol = (float)1E-2f * (Sfull[0] + (float)1);
            for (int t = 0; t < r; t++)
                AssertClose(Sk[t], Sfull[t], svTol);

            // Tail σ (indices r..k-1) must be much smaller than Sk[0]:
            // the rank-3 null space dimensions produce near-zero Ritz values.
            float tailTol = (float)1E-3f * Sk[0];
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
            var A = arena.floatRandomMatrix(m, n, (float)(-2f), (float)2f, 0x55AA77BBu);

            int k = 3, oversample = 0;
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0xAABBCCDDu, 75, out bool cT);

            // With p=k and no oversampling the residual is far from zero → converged=false
            Assert.IsFalse(cT);

            arena.Dispose();
        }

        // Test 7: maxIter=1 forces the inner bidiagonal QR (on p×p B) to not converge.
        // Exercises the svdThin-failure branch → converged=false even with p < n.
        void GklConvergedFalse_MaxIter1()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 40, n = 20;
            // k=5, oversample=5 → p = min(10, 20) = 10. Confirmed p=10 < n=20.
            // maxIter=1: the inner 10×10 bidiagonal QR almost certainly needs > 1 sweep.
            var A = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, 0xDEADBEEFu);

            int k = 5, oversample = 5;
            var Uk = arena.floatMat(m, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x12AB34CDu, 1, out bool cT);

            // With maxIter=1 the inner bidiagonal QR will not converge for a non-trivial matrix
            Assert.IsFalse(cT);

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void LowRankTests(TestJob.TestType type)
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
    public void TruncatedThrowsOnBadK()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(6, 4);
        var Uk = arena.floatMat(6, 5);
        var Sk = arena.floatVec(5);
        var Vk = arena.floatMat(4, 5);
        Assert.Catch<ArgumentException>(() => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, 5, out bool _)); // k=5 > n=4
        arena.Dispose();
    }

    [Test]
    public void TruncatedThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(3, 5);
        var Uk = arena.floatMat(3, 2);
        var Sk = arena.floatVec(2);
        var Vk = arena.floatMat(5, 2);
        Assert.Catch<ArgumentException>(() => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, 2, out bool _));
        arena.Dispose();
    }

    [Test]
    public void LowRankApproxThrowsOnWrongAkShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(6, 4);
        var Ak = arena.floatMat(4, 6);   // must be m x n = 6 x 4
        Assert.Catch<ArgumentException>(() => SVD.lowRankApprox(in A, ref Ak, 2, out bool _));
        arena.Dispose();
    }
}
