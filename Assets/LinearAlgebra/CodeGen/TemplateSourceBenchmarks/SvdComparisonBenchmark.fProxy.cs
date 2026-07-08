using System;
using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

using Random = Unity.Mathematics.Random;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of SvdComparisonBenchmark (setup + timing IJobs, per-size measure
    // methods, and managed accuracy helpers). The dtype-agnostic harness (sizes, Run, Section, the
    // dedicated-case orchestration) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/SvdComparisonBenchmark.cs; shared formatters + KVals + BuildSeed
    // live in the public SvdCmpFmt helper there.

    // ---- setup job: build A from a known SVD (not timed) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpBuildJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN SigmaTrue;
        public uint seed;

        public void Execute()
        {
            int m = A.M_Rows, n = A.N_Cols;
            var rng = new Random(seed);

            // Sigma[i] = 100 * 0.95^i  (geometric decay; 0.95 keeps Sigma[255] ~ 2e-4,
            // so kappa ~ 5e5 — realistic and well above both float and double epsilon).
            fProxy sig = (fProxy)100;
            for (int i = 0; i < n; i++) { SigmaTrue[i] = sig; sig *= (fProxy)0.95; }

            // U (m x n) via QR of a random Gaussian m x n matrix → orthonormal columns (Stiefel)
            var G = new fProxyMxN(m, n, Allocator.Temp, false);
            var R = new fProxyMxN(n, n, Allocator.Temp, false);
            var gauss = new fProxyGaussian(0f, 1f);
            Rand.randomInPlace(ref rng, ref G, ref gauss);
            QR.decompInPlace(ref G, ref R);   // G → Q in-place

            // V (n x n) Haar-uniform orthogonal
            var V = new fProxyMxN(n, n, Allocator.Temp, false);
            Rand.orthogonalInPlace(ref rng, ref V);

            // A[i,j] = Σ_t  Sigma[t] · G[i,t] · V[j,t]   (double accumulation for accuracy)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)G[i, t] * (double)SigmaTrue[t] * (double)V[j, t];
                    A[i, j] = (fProxy)acc;
                }

            G.Dispose(); R.Dispose(); V.Dispose();
        }
    }

    // ---- timing job: thin (full Golub-Kahan) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpThinJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN U;
        public fProxyN S;
        public fProxyMxN V;
        public void Execute() => SVD.thin(in A, ref U, ref S, ref V);
    }

    // ---- timing job: truncated (GKL Lanczos + full reorthogonalization) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpTruncJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Uk;
        public fProxyN Sk;
        public fProxyMxN Vk;
        public int k;
        public fProxySVDTruncatedCache ws;
        public void Execute() => SVD.truncated(in A, ref Uk, ref Sk, ref Vk, k, ref ws);
    }

    // ---- timing job: randomized (Halko-Martinsson-Tropp) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpRandJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Uk;
        public fProxyN Sk;
        public fProxyMxN Vk;
        public int k;
        public fProxySVDRandomizedCache ws;
        // oversample=10, powerIters=2, seed=0x9E3779B1 (library defaults).
        public void Execute() => SVD.randomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9E3779B1u, 75, ref ws);
    }

    public static partial class SvdComparisonBenchmark
    {
        // ---- full k-sweep for one size ----
        static void BenchSizeFProxy(StringBuilder sb, int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A         = arena.fProxyMat(m, n);
            var sigmaTrue = arena.fProxyVec(n);

            new SvdCmpBuildJobFProxy { A = A, SigmaTrue = sigmaTrue, seed = SvdCmpFmt.BuildSeed }.Run();
            double normA = FNormFProxy(A);

            // thin — k = n (full decomposition)
            {
                var U = arena.fProxyMat(m, n);
                var S = arena.fProxyVec(n);
                var V = arena.fProxyMat(n, n);
                var job  = new SvdCmpThinJobFProxy { A = A, U = U, S = S, V = V };
                var stat = Bench.Time(() => job.Run());
                double sigErr   = SigErrFProxy(S, sigmaTrue, n);
                double reconErr = ReconErrFProxy(A, U, S, V, n, normA);
                double eyOpt    = EYOptFProxy(sigmaTrue, n, normA);
                sb.AppendLine(SvdCmpFmt.CmpRow("fProxy", "thin", m, n, n, stat, sigErr, reconErr, eyOpt));
            }

            foreach (int k in SvdCmpFmt.KVals(n))
            {
                var Uk = arena.fProxyMat(m, k);
                var Sk = arena.fProxyVec(k);
                var Vk = arena.fProxyMat(n, k);

                // truncated (GKL)
                {
                    var ws   = arena.fProxySVDTruncatedCache(m, n, k);
                    var job  = new SvdCmpTruncJobFProxy { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
                    var stat = Bench.Time(() => job.Run());
                    double sigErr   = SigErrFProxy(Sk, sigmaTrue, k);
                    double reconErr = ReconErrFProxy(A, Uk, Sk, Vk, k, normA);
                    double eyOpt    = EYOptFProxy(sigmaTrue, k, normA);
                    sb.AppendLine(SvdCmpFmt.CmpRow("fProxy", "svdTrunc", m, n, k, stat, sigErr, reconErr, eyOpt));
                }

                // randomized (HMT, oversample=10 matches workspace default)
                {
                    var ws   = arena.fProxySVDRandomizedCache(m, n, k);
                    var job  = new SvdCmpRandJobFProxy { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
                    var stat = Bench.Time(() => job.Run());
                    double sigErr   = SigErrFProxy(Sk, sigmaTrue, k);
                    double reconErr = ReconErrFProxy(A, Uk, Sk, Vk, k, normA);
                    double eyOpt    = EYOptFProxy(sigmaTrue, k, normA);
                    sb.AppendLine(SvdCmpFmt.CmpRow("fProxy", "svdRand", m, n, k, stat, sigErr, reconErr, eyOpt));
                }
            }

            arena.Dispose();
        }

        static void BenchThinDedicatedFProxy(StringBuilder sb, int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A         = arena.fProxyMat(m, n);
            var sigmaTrue = arena.fProxyVec(n);
            new SvdCmpBuildJobFProxy { A = A, SigmaTrue = sigmaTrue, seed = SvdCmpFmt.BuildSeed }.Run();
            double normA = FNormFProxy(A);

            var U = arena.fProxyMat(m, n);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            var job  = new SvdCmpThinJobFProxy { A = A, U = U, S = S, V = V };
            var stat = Bench.Time(() => job.Run());
            double sigErr   = SigErrFProxy(S, sigmaTrue, n);
            double reconErr = ReconErrFProxy(A, U, S, V, n, normA);
            double eyOpt    = EYOptFProxy(sigmaTrue, n, normA);
            sb.AppendLine(SvdCmpFmt.CmpRow("fProxy", "svdThin", m, n, n, stat, sigErr, reconErr, eyOpt));

            arena.Dispose();
        }

        static void BenchRandDedicatedFProxy(StringBuilder sb, int m, int n, int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A         = arena.fProxyMat(m, n);
            var sigmaTrue = arena.fProxyVec(n);
            new SvdCmpBuildJobFProxy { A = A, SigmaTrue = sigmaTrue, seed = SvdCmpFmt.BuildSeed }.Run();
            double normA = FNormFProxy(A);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            var ws   = arena.fProxySVDRandomizedCache(m, n, k);
            var job  = new SvdCmpRandJobFProxy { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
            var stat = Bench.Time(() => job.Run());
            double sigErr   = SigErrFProxy(Sk, sigmaTrue, k);
            double reconErr = ReconErrFProxy(A, Uk, Sk, Vk, k, normA);
            double eyOpt    = EYOptFProxy(sigmaTrue, k, normA);
            sb.AppendLine(SvdCmpFmt.CmpRow("fProxy", "svdRand", m, n, k, stat, sigErr, reconErr, eyOpt));

            arena.Dispose();
        }

        static void BenchTrunc1024FProxy(StringBuilder sb, int m, int n, int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A         = arena.fProxyMat(m, n);
            var sigmaTrue = arena.fProxyVec(n);
            new SvdCmpBuildJobFProxy { A = A, SigmaTrue = sigmaTrue, seed = SvdCmpFmt.BuildSeed }.Run();
            double normA = FNormFProxy(A);

            var Uk = arena.fProxyMat(m, k);
            var Sk = arena.fProxyVec(k);
            var Vk = arena.fProxyMat(n, k);
            var ws   = arena.fProxySVDTruncatedCache(m, n, k);
            var job  = new SvdCmpTruncJobFProxy { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
            var stat = Bench.Time(() => job.Run());
            double sigErr   = SigErrFProxy(Sk, sigmaTrue, k);
            double reconErr = ReconErrFProxy(A, Uk, Sk, Vk, k, normA);
            double eyOpt    = EYOptFProxy(sigmaTrue, k, normA);
            sb.AppendLine(SvdCmpFmt.CmpRow("fProxy", "svdTrunc", m, n, k, stat, sigErr, reconErr, eyOpt));

            arena.Dispose();
        }

        // ---- accuracy helpers (managed, not Burst) ----

        static double FNormFProxy(fProxyMxN A)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double s = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) { double v = A[i, j]; s += v * v; }
            return Math.Sqrt(s);
        }

        // max_{i<k} |S[i] - SigmaTrue[i]| / SigmaTrue[0]
        static double SigErrFProxy(fProxyN S, fProxyN sigmaTrue, int k)
        {
            double maxErr = 0, s0 = sigmaTrue[0];
            for (int i = 0; i < k; i++)
            {
                double err = Math.Abs((double)S[i] - (double)sigmaTrue[i]) / s0;
                if (err > maxErr) maxErr = err;
            }
            return maxErr;
        }

        // ||A - Uk·diag(Sk)·Vkᵀ||_F / ||A||_F  (uses first k columns of Uk/Vk)
        static double ReconErrFProxy(fProxyMxN A, fProxyMxN Uk, fProxyN Sk, fProxyMxN Vk, int k, double normA)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double err2 = 0;
            double[] usk = new double[k]; // u[:,t]*s[t] for current row i
            for (int i = 0; i < m; i++)
            {
                for (int t = 0; t < k; t++) usk[t] = (double)Uk[i, t] * (double)Sk[t];
                for (int j = 0; j < n; j++)
                {
                    double approx = 0;
                    for (int t = 0; t < k; t++) approx += usk[t] * (double)Vk[j, t];
                    double diff = (double)A[i, j] - approx;
                    err2 += diff * diff;
                }
            }
            return (normA > 0) ? Math.Sqrt(err2) / normA : Math.Sqrt(err2);
        }

        // sqrt(sum_{i>=k} Sigma[i]^2) / ||A||_F  (Eckart-Young optimal truncation error)
        static double EYOptFProxy(fProxyN sigmaTrue, int k, double normA)
        {
            int n = sigmaTrue.N;
            double tail2 = 0;
            for (int i = k; i < n; i++) { double s = sigmaTrue[i]; tail2 += s * s; }
            return (normA > 0) ? Math.Sqrt(tail2) / normA : Math.Sqrt(tail2);
        }
    }
}
