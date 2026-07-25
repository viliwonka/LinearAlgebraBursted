using System;
using System.Globalization;
using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic table formatting + config for SvdComparisonBenchmark. Public so the
    // code-generated per-dtype measure methods (in a separate template assembly) can reach them.
    public static class SvdCmpFmt
    {
        // Fixed seed for the known-SVD construction; reproducible across runs.
        public const uint BuildSeed = 0xCAFEBABEu;

        public static string CmpHeader() =>
            string.Format("{0,-7} {1,-11} {2,-10} {3,5} {4,4} {5,11} {6,12} {7,12} {8,12}",
                "dtype", "method", "size", "k", "k%", "med(ms)", "sig-rel-err", "recon-err", "EY-opt");

        public static string CmpRow(string dtype, string method, int m, int n, int k,
                             Bench.Stat stat, double sigErr, double reconErr, double eyOpt)
        {
            int pct = (n > 0) ? (100 * k / n) : 0;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-11} {2,-10} {3,5} {4,3}% {5,11:F3} {6,12:E3} {7,12:E3} {8,12:E3}",
                dtype, method, $"{m}x{n}", k, pct, stat.Median, sigErr, reconErr, eyOpt);
        }

        // k sweep: round(3%), round(7%), round(21%) of n.
        public static int[] KVals(int n) => new[]
        {
            (int)Math.Round(0.03 * n),
            (int)Math.Round(0.07 * n),
            (int)Math.Round(0.21 * n),
        };
    }

    // SVD method comparison: thin (full Golub-Kahan) vs truncated (GKL Lanczos with full
    // reorthogonalization) vs randomized (Halko-Martinsson-Tropp random projection).
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
    // Sizes (tall, m ≥ n): 512×64, 1024×128, 2048×256. k sweep: round(3/7/21%) of n; thin uses k=n.
    //
    // Hand-written harness half. The setup/timing IJobs, per-size measure methods, and accuracy
    // helpers are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SvdComparisonBenchmark.fProxy.cs.
    public static partial class SvdComparisonBenchmark
    {
        // Sizes (tall, m >= n; each is 8:1 aspect ratio, 2x progression).
        static readonly (int m, int n)[] Sizes = { (512, 64), (1024, 128), (2048, 256) };

        public static void Run() => Bench.WriteReport("benchmark-svd-compare.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== SVD method comparison: thin vs truncated(GKL) vs randomized(HMT) ===");
            sb.AppendLine("    Tall matrices (spec: 64x512 / 128x1024 / 256x2048 wide, TRANSPOSED so m>=n).");
            sb.AppendLine("    Known SVD: A=U*diag(Sigma)*Vt, U in Stiefel(n,m), V in O(n), Sigma[i]=100*0.95^i.");
            sb.AppendLine("    thin: full Golub-Kahan bidiagonal (k=n, all singular values).");
            sb.AppendLine("    truncated (GKL): Lanczos bidiag + full DGKS reortho, p=min(n,max(2k,k+12)).");
            sb.AppendLine("    randomized (HMT): Gaussian sketch, oversample=10, powerIters=2, seed=0x9E3779B1.");
            sb.AppendLine("    sigma-rel-err = max_{i<k} |S[i]-Sigma[i]| / Sigma[0].");
            sb.AppendLine("    EY-opt = Eckart-Young lower bound = sqrt(sum_{i>=k} Sigma[i]^2) / ||A||_F.");
            sb.AppendLine();
            sb.AppendLine(SvdCmpFmt.CmpHeader());

            foreach (var (m, n) in Sizes)
            {
                BenchSizeFloat(sb, m, n);
                BenchSizeDouble(sb, m, n);
                sb.AppendLine();
            }

            Section1024Square(sb);
            SectionTall2048x512(sb);
        }

        // ---- Dedicated tall 2048x512 (m > n, the LS-benchmark shape): thin vs truncated vs randomized, k=21 ----
        static void SectionTall2048x512(StringBuilder sb)
        {
            const int m = 2048, n = 512, k = 21;
            sb.AppendLine("--- Dedicated: SVD.thin vs SVD.truncated vs SVD.randomized at 2048x512 (tall), k=21 ---");
            sb.AppendLine(SvdCmpFmt.CmpHeader());

            BenchThinDedicatedFloat(sb, m, n);
            BenchThinDedicatedDouble(sb, m, n);
            BenchTrunc1024Float(sb, m, n, k);
            BenchTrunc1024Double(sb, m, n, k);
            BenchRandDedicatedFloat(sb, m, n, k);
            BenchRandDedicatedDouble(sb, m, n, k);
        }

        // ---- Dedicated square 1024x1024: truncated ONLY, k=54 (matches the 2048x256 k=54 row) ----
        static void Section1024Square(StringBuilder sb)
        {
            const int m = 1024, n = 1024, k = 54;
            sb.AppendLine("--- Dedicated: SVD.truncated at 1024x1024 (square), k=54 (matches the 2048x256 k=54 row) ---");
            sb.AppendLine(SvdCmpFmt.CmpHeader());

            BenchTrunc1024Float(sb, m, n, k);
            BenchTrunc1024Double(sb, m, n, k);
        }
    }
}
