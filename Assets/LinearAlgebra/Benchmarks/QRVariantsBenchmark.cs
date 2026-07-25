using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic row formatters for QRVariantsBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can call them.
    public static class QRVariantsFmt
    {
        public static string RowTall(string dtype, string kernel, int m, int n, Bench.Stat st, double flops)
        {
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-24} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4} {6,12:F2}",
                dtype, kernel + " " + m + "x" + n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }

        public static string RowKernel(string dtype, string kernel, int n, Bench.Stat st, double flops)
        {
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-24} {2,-6} {3,11:F4} {4,11:F4} {5,11:F4} {6,11:F4} {7,12:F2}",
                dtype, kernel, n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }
    }

    // The other Householder paths that share QR's reflector-apply hot loop: column-pivoted QR (QRCP,
    // rank-revealing), the direct least-squares solve, and the COD min-norm solve. Each Execute copies
    // a pristine source into the working matrix so every timed sample does identical work.
    //
    // Hand-written harness half. The timed IJobs (QRCP/QRSolve/QRCPSolve/QRCPMinNorm Job {Float,Double})
    // and build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QRVariantsBenchmark.fProxy.cs.
    public static partial class QRVariantsBenchmark
    {
        // (4/3) N^3 leading term (approximate). QRCP adds an O(N^3) exact pivot-norm recompute on top,
        // and QR.solveInPlace skips the Q reconstruction, so GFLOP/s here is only a rough comparator —
        // the time columns and the A/B speedup are the honest signal.
        static double Flops(int n) => (4.0 / 3.0) * n * (double)n * n;

        // Tall least-squares shapes. The reflector sweep's leading term is 2 n^2 (m - n/3).
        static readonly int[][] TallSizes = { new[] { 2048, 512 }, new[] { 2048, 1024 } };
        static double TallFlops(int m, int n) => 2.0 * n * (double)n * (m - n / 3.0);

        public static void Run() => Bench.WriteReport("benchmark-qrvariants.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== QRCP (column-pivoted, rank-revealing QR; forms Q) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== QR.solveInPlace (Householder least-squares solve; no Q reconstruction) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(SolveFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(SolveDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== QRCP.solveInPlace (QRCP rank-safe LS solve; zero-alloc primitive) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPSolveFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPSolveDouble(n, Flops(n)));
            sb.AppendLine();

            // COD overhead: each size emits the basic and the COD row adjacently on the SAME rank-deficient
            // matrix (rank = 3n/4), so the extra second-sweep cost reads straight off the pair.
            sb.AppendLine("=== QRCP rank-deficient (n x n, rank = 3n/4): basic solveInPlace vs COD minNormSolveInPlace ===");
            sb.AppendLine(HeaderKernel());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPRankDefFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPRankDefDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== TALL overdetermined least squares (m x n, m > n): QR.solveInPlace vs QRCP.solveInPlace ===");
            sb.AppendLine(HeaderTall());
            foreach (var s in TallSizes) sb.AppendLine(SolveTallFloat(s[0], s[1], TallFlops(s[0], s[1])));
            foreach (var s in TallSizes) sb.AppendLine(SolveTallDouble(s[0], s[1], TallFlops(s[0], s[1])));
            foreach (var s in TallSizes) sb.AppendLine(QRCPSolveTallFloat(s[0], s[1], TallFlops(s[0], s[1])));
            foreach (var s in TallSizes) sb.AppendLine(QRCPSolveTallDouble(s[0], s[1], TallFlops(s[0], s[1])));
            sb.AppendLine();
        }

        static string HeaderTall()
        {
            return string.Format("{0,-7} {1,-24} {2,11} {3,11} {4,11} {5,11} {6,12}",
                "dtype", "kernel m x n", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~");
        }

        static string HeaderKernel()
        {
            return string.Format("{0,-7} {1,-24} {2,-6} {3,11} {4,11} {5,11} {6,11} {7,12}",
                "dtype", "kernel", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~");
        }
    }
}
