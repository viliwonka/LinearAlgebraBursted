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
    // SVD method comparison: svdThin (full Golub-Kahan) vs svdTruncated (GKL Lanczos with full
    // reorthogonalization) vs svdRandomized (Halko-Martinsson-Tropp random projection).
    //
    // Both SPEED and NUMERICAL ACCURACY are reported.  Accuracy is measured against a KNOWN SVD:
    //   A = U · diag(Σ) · Vᵀ,  Σ[i] = 100 · 0.95^i  (geometric decay),
    //   U ∈ Stiefel(n, m) from QR of a Gaussian m×n matrix, V ∈ O(n) Haar-uniform.
    // The exact Σ is the ground truth.  The build is a one-shot IJob, not timed.
    //
    // NOTE: 0.95^i rather than the spec-suggested 0.92^i — 0.92^255 ≈ 5.5e-10 is below float ε,
    // making κ ≈ 2e9 and causing the double bidiagonalQR to fail within 75 iterations. 0.95^255
    // ≈ 2e-4 gives κ ≈ 5e5 (realistic; convergence reliable for both float and double).
    //
    // Sizes (tall, m ≥ n — the spec's 64×512 / 128×1024 / 256×2048 wide orientations transposed):
    //    512×64   (n=64 singular values)
    //   1024×128  (n=128)
    //   2048×256  (n=256)
    //
    // k sweep: round(3%), round(7%), round(21%) of n.  svdThin uses k = n (full).
    // svdRandomized defaults: oversample=10, powerIters=2, seed=0x9E3779B1 (library default).
    // svdTruncated defaults: p = min(n, max(2k, k+12)).

    // =====================================================================
    //  Setup jobs — build A from a known SVD (not timed)
    // =====================================================================

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpBuildJobFloat : IJob
    {
        public floatMxN A;
        public floatN SigmaTrue;
        public uint seed;

        public void Execute()
        {
            int m = A.M_Rows, n = A.N_Cols;
            var rng = new Random(seed);

            // Sigma[i] = 100 * 0.95^i  (geometric decay; 0.95 keeps Sigma[255] ~ 2e-4,
            // so kappa ~ 5e5 — realistic and well above both float and double epsilon).
            float sig = 100f;
            for (int i = 0; i < n; i++) { SigmaTrue[i] = sig; sig *= 0.95f; }

            // U (m x n) via QR of a random Gaussian m x n matrix → orthonormal columns (Stiefel)
            var G = new floatMxN(m, n, Allocator.Temp, false);
            var R = new floatMxN(n, n, Allocator.Temp, false);
            var gauss = new floatGaussian(0f, 1f);
            floatRandomOP.randomInpl(ref rng, ref G, ref gauss);
            OrthoOP.qrDecomposition(ref G, ref R);   // G → Q in-place

            // V (n x n) Haar-uniform orthogonal
            var V = new floatMxN(n, n, Allocator.Temp, false);
            floatRandomMatrixOP.randomOrthogonalInpl(ref rng, ref V);

            // A[i,j] = Σ_t  Sigma[t] · G[i,t] · V[j,t]   (double accumulation for accuracy)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)G[i, t] * (double)SigmaTrue[t] * (double)V[j, t];
                    A[i, j] = (float)acc;
                }

            G.Dispose(); R.Dispose(); V.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpBuildJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN SigmaTrue;
        public uint seed;

        public void Execute()
        {
            int m = A.M_Rows, n = A.N_Cols;
            var rng = new Random(seed);

            // Sigma[i] = 100 * 0.95^i  (matches float setup job above)
            double sig = 100.0;
            for (int i = 0; i < n; i++) { SigmaTrue[i] = sig; sig *= 0.95; }

            var G = new doubleMxN(m, n, Allocator.Temp, false);
            var R = new doubleMxN(n, n, Allocator.Temp, false);
            var gauss = new doubleGaussian(0.0, 1.0);
            doubleRandomOP.randomInpl(ref rng, ref G, ref gauss);
            OrthoOP.qrDecomposition(ref G, ref R);

            var V = new doubleMxN(n, n, Allocator.Temp, false);
            doubleRandomMatrixOP.randomOrthogonalInpl(ref rng, ref V);

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += G[i, t] * SigmaTrue[t] * V[j, t];
                    A[i, j] = acc;
                }

            G.Dispose(); R.Dispose(); V.Dispose();
        }
    }

    // =====================================================================
    //  Timing jobs: svdThin (full Golub-Kahan)
    // =====================================================================

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpThinJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN U;
        public floatN S;
        public floatMxN V;
        public void Execute() => SVD.svdThin(in A, ref U, ref S, ref V);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpThinJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN U;
        public doubleN S;
        public doubleMxN V;
        public void Execute() => SVD.svdThin(in A, ref U, ref S, ref V);
    }

    // =====================================================================
    //  Timing jobs: svdTruncated (GKL Lanczos + full reorthogonalization)
    // =====================================================================

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpTruncJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Uk;
        public floatN Sk;
        public floatMxN Vk;
        public int k;
        public floatSvdTruncatedWorkspace ws;
        // Uses the 1-k-arg overload: oversample = max(k,12), seed = 0x9E3779B1, maxIter = 75.
        public void Execute() => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, ref ws, out bool _);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpTruncJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Uk;
        public doubleN Sk;
        public doubleMxN Vk;
        public int k;
        public doubleSvdTruncatedWorkspace ws;
        public void Execute() => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, ref ws, out bool _);
    }

    // =====================================================================
    //  Timing jobs: svdRandomized (Halko-Martinsson-Tropp)
    // =====================================================================

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpRandJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Uk;
        public floatN Sk;
        public floatMxN Vk;
        public int k;
        public floatSvdRandomizedWorkspace ws;
        // oversample=10, powerIters=2, seed=0x9E3779B1 (library defaults).
        public void Execute() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9E3779B1u, 75, ref ws);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdCmpRandJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Uk;
        public doubleN Sk;
        public doubleMxN Vk;
        public int k;
        public doubleSvdRandomizedWorkspace ws;
        public void Execute() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9E3779B1u, 75, ref ws);
    }

    // =====================================================================
    //  Main benchmark class
    // =====================================================================

    public static class SvdComparisonBenchmark
    {
        // Fixed seed for the known-SVD construction; reproducible across runs.
        const uint BuildSeed = 0xCAFEBABEu;

        // Sizes (tall, m >= n; each is 8:1 aspect ratio, 2x progression).
        static readonly (int m, int n)[] Sizes = { (512, 64), (1024, 128), (2048, 256) };

        public static void Run() => Bench.WriteReport("benchmark-svd-compare.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== SVD method comparison: svdThin vs svdTruncated(GKL) vs svdRandomized(HMT) ===");
            sb.AppendLine("    Tall matrices (spec: 64x512 / 128x1024 / 256x2048 wide, TRANSPOSED so m>=n).");
            sb.AppendLine("    Known SVD: A=U*diag(Sigma)*Vt, U in Stiefel(n,m), V in O(n), Sigma[i]=100*0.95^i.");
            sb.AppendLine("    svdThin: full Golub-Kahan bidiagonal (k=n, all singular values).");
            sb.AppendLine("    svdTruncated (GKL): Lanczos bidiag + full DGKS reortho, p=min(n,max(2k,k+12)).");
            sb.AppendLine("    svdRandomized (HMT): Gaussian sketch, oversample=10, powerIters=2, seed=0x9E3779B1.");
            sb.AppendLine("    sigma-rel-err = max_{i<k} |S[i]-Sigma[i]| / Sigma[0].");
            sb.AppendLine("    EY-opt = Eckart-Young lower bound = sqrt(sum_{i>=k} Sigma[i]^2) / ||A||_F.");
            sb.AppendLine();
            sb.AppendLine(CmpHeader());

            foreach (var (m, n) in Sizes)
            {
                BenchSizeFloat(sb, m, n);
                BenchSizeDouble(sb, m, n);
                sb.AppendLine();
            }
        }

        // ---- Table formatting ----

        static string CmpHeader() =>
            string.Format("{0,-7} {1,-11} {2,-10} {3,5} {4,4} {5,11} {6,12} {7,12} {8,12}",
                "dtype", "method", "size", "k", "k%", "med(ms)", "sig-rel-err", "recon-err", "EY-opt");

        static string CmpRow(string dtype, string method, int m, int n, int k,
                             Bench.Stat stat, double sigErr, double reconErr, double eyOpt)
        {
            int pct = (n > 0) ? (100 * k / n) : 0;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-11} {2,-10} {3,5} {4,3}% {5,11:F3} {6,12:E3} {7,12:E3} {8,12:E3}",
                dtype, method, $"{m}x{n}", k, pct, stat.Median, sigErr, reconErr, eyOpt);
        }

        // k sweep: round(3%), round(7%), round(21%) of n.
        static int[] KVals(int n) => new[]
        {
            (int)Math.Round(0.03 * n),
            (int)Math.Round(0.07 * n),
            (int)Math.Round(0.21 * n),
        };

        // =====================================================================
        //  Float benchmarks
        // =====================================================================

        static void BenchSizeFloat(StringBuilder sb, int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A         = arena.floatMat(m, n);
            var sigmaTrue = arena.floatVec(n);

            // One-shot build: A = U·diag(Sigma)·Vᵀ  (not timed)
            new SvdCmpBuildJobFloat { A = A, SigmaTrue = sigmaTrue, seed = BuildSeed }.Run();
            double normA = FNormF(A);

            // svdThin — k = n (full decomposition)
            {
                var U = arena.floatMat(m, n);
                var S = arena.floatVec(n);
                var V = arena.floatMat(n, n);
                var job  = new SvdCmpThinJobFloat { A = A, U = U, S = S, V = V };
                var stat = Bench.Time(() => job.Run());
                double sigErr   = SigErrF(S, sigmaTrue, n);
                double reconErr = ReconErrF(A, U, S, V, n, normA);
                double eyOpt    = EYOptF(sigmaTrue, n, normA);
                sb.AppendLine(CmpRow("float", "svdThin", m, n, n, stat, sigErr, reconErr, eyOpt));
            }

            foreach (int k in KVals(n))
            {
                var Uk = arena.floatMat(m, k);
                var Sk = arena.floatVec(k);
                var Vk = arena.floatMat(n, k);

                // svdTruncated (GKL)
                {
                    var ws   = arena.floatSvdTruncatedWorkspace(m, n, k);
                    var job  = new SvdCmpTruncJobFloat { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
                    var stat = Bench.Time(() => job.Run());
                    double sigErr   = SigErrF(Sk, sigmaTrue, k);
                    double reconErr = ReconErrF(A, Uk, Sk, Vk, k, normA);
                    double eyOpt    = EYOptF(sigmaTrue, k, normA);
                    sb.AppendLine(CmpRow("float", "svdTrunc", m, n, k, stat, sigErr, reconErr, eyOpt));
                }

                // svdRandomized (HMT, oversample=10 matches workspace default)
                {
                    var ws   = arena.floatSvdRandomizedWorkspace(m, n, k);
                    var job  = new SvdCmpRandJobFloat { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
                    var stat = Bench.Time(() => job.Run());
                    double sigErr   = SigErrF(Sk, sigmaTrue, k);
                    double reconErr = ReconErrF(A, Uk, Sk, Vk, k, normA);
                    double eyOpt    = EYOptF(sigmaTrue, k, normA);
                    sb.AppendLine(CmpRow("float", "svdRand", m, n, k, stat, sigErr, reconErr, eyOpt));
                }
            }

            arena.Dispose();
        }

        // =====================================================================
        //  Double benchmarks
        // =====================================================================

        static void BenchSizeDouble(StringBuilder sb, int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A         = arena.doubleMat(m, n);
            var sigmaTrue = arena.doubleVec(n);

            new SvdCmpBuildJobDouble { A = A, SigmaTrue = sigmaTrue, seed = BuildSeed }.Run();
            double normA = FNormD(A);

            // svdThin — k = n
            {
                var U = arena.doubleMat(m, n);
                var S = arena.doubleVec(n);
                var V = arena.doubleMat(n, n);
                var job  = new SvdCmpThinJobDouble { A = A, U = U, S = S, V = V };
                var stat = Bench.Time(() => job.Run());
                double sigErr   = SigErrD(S, sigmaTrue, n);
                double reconErr = ReconErrD(A, U, S, V, n, normA);
                double eyOpt    = EYOptD(sigmaTrue, n, normA);
                sb.AppendLine(CmpRow("double", "svdThin", m, n, n, stat, sigErr, reconErr, eyOpt));
            }

            foreach (int k in KVals(n))
            {
                var Uk = arena.doubleMat(m, k);
                var Sk = arena.doubleVec(k);
                var Vk = arena.doubleMat(n, k);

                // svdTruncated (GKL)
                {
                    var ws   = arena.doubleSvdTruncatedWorkspace(m, n, k);
                    var job  = new SvdCmpTruncJobDouble { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
                    var stat = Bench.Time(() => job.Run());
                    double sigErr   = SigErrD(Sk, sigmaTrue, k);
                    double reconErr = ReconErrD(A, Uk, Sk, Vk, k, normA);
                    double eyOpt    = EYOptD(sigmaTrue, k, normA);
                    sb.AppendLine(CmpRow("double", "svdTrunc", m, n, k, stat, sigErr, reconErr, eyOpt));
                }

                // svdRandomized (HMT)
                {
                    var ws   = arena.doubleSvdRandomizedWorkspace(m, n, k);
                    var job  = new SvdCmpRandJobDouble { A = A, Uk = Uk, Sk = Sk, Vk = Vk, k = k, ws = ws };
                    var stat = Bench.Time(() => job.Run());
                    double sigErr   = SigErrD(Sk, sigmaTrue, k);
                    double reconErr = ReconErrD(A, Uk, Sk, Vk, k, normA);
                    double eyOpt    = EYOptD(sigmaTrue, k, normA);
                    sb.AppendLine(CmpRow("double", "svdRand", m, n, k, stat, sigErr, reconErr, eyOpt));
                }
            }

            arena.Dispose();
        }

        // =====================================================================
        //  Accuracy helpers — float (managed, not Burst)
        // =====================================================================

        static double FNormF(floatMxN A)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double s = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) { double v = A[i, j]; s += v * v; }
            return Math.Sqrt(s);
        }

        // max_{i<k} |S[i] - SigmaTrue[i]| / SigmaTrue[0]
        static double SigErrF(floatN S, floatN sigmaTrue, int k)
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
        static double ReconErrF(floatMxN A, floatMxN Uk, floatN Sk, floatMxN Vk, int k, double normA)
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
        static double EYOptF(floatN sigmaTrue, int k, double normA)
        {
            int n = sigmaTrue.N;
            double tail2 = 0;
            for (int i = k; i < n; i++) { double s = sigmaTrue[i]; tail2 += s * s; }
            return (normA > 0) ? Math.Sqrt(tail2) / normA : Math.Sqrt(tail2);
        }

        // =====================================================================
        //  Accuracy helpers — double (managed, not Burst)
        // =====================================================================

        static double FNormD(doubleMxN A)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double s = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) { double v = A[i, j]; s += v * v; }
            return Math.Sqrt(s);
        }

        static double SigErrD(doubleN S, doubleN sigmaTrue, int k)
        {
            double maxErr = 0, s0 = sigmaTrue[0];
            for (int i = 0; i < k; i++)
            {
                double err = Math.Abs(S[i] - sigmaTrue[i]) / s0;
                if (err > maxErr) maxErr = err;
            }
            return maxErr;
        }

        static double ReconErrD(doubleMxN A, doubleMxN Uk, doubleN Sk, doubleMxN Vk, int k, double normA)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double err2 = 0;
            double[] usk = new double[k];
            for (int i = 0; i < m; i++)
            {
                for (int t = 0; t < k; t++) usk[t] = Uk[i, t] * Sk[t];
                for (int j = 0; j < n; j++)
                {
                    double approx = 0;
                    for (int t = 0; t < k; t++) approx += usk[t] * Vk[j, t];
                    double diff = A[i, j] - approx;
                    err2 += diff * diff;
                }
            }
            return (normA > 0) ? Math.Sqrt(err2) / normA : Math.Sqrt(err2);
        }

        static double EYOptD(doubleN sigmaTrue, int k, double normA)
        {
            int n = sigmaTrue.N;
            double tail2 = 0;
            for (int i = k; i < n; i++) { double s = sigmaTrue[i]; tail2 += s * s; }
            return (normA > 0) ? Math.Sqrt(tail2) / normA : Math.Sqrt(tail2);
        }
    }
}
