using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic row formatter for TallWideSolveBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can call it.
    public static class TallWideFmt
    {
        public static string RowKernel(string dtype, string kernel, int n, Bench.Stat st, double flops)
        {
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-24} {2,-6} {3,11:F4} {4,11:F4} {5,11:F4} {6,11:F4} {7,12:F2}",
                dtype, kernel, n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }
    }

    // Non-square solve paths: the rectangular problems the square LU/Cholesky benchmarks never touch.
    //
    //   TALL  (m = 2n): Householder QR -- decompInPlace forms the thin Q, solveInPlace does the direct
    //         no-Q solve.
    //   WIDE  (n = 2m): LQ decomp + minNormSolve, plus the row-pivoted rank-revealing LQRP.
    //
    // Sized by the SMALLER dimension k (the N column), fixed 2:1 aspect. All share the same leading-term
    // flop count — QrFlops(2k, k) = (10/3) k^3 — so GFLOP/s is directly comparable across sections.
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per dtype
    // from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/TallWideSolveBenchmark.fProxy.cs.
    public static partial class TallWideSolveBenchmark
    {
        // Householder QR reflector-sweep leading term for a rows x cols panel (rows >= cols):
        // 2 cols^2 (rows - cols/3). For every section here the controlling panel is 2k x k.
        static double QrFlops(int rows, int cols) => 2.0 * cols * (double)cols * (rows - cols / 3.0);

        public static void Run() => Bench.WriteReport("benchmark-tallwide.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Tall QR factorization (decompInPlace, A is 2k x k; forms thin Q); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(TallQRFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(TallQRDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();

            sb.AppendLine("=== Overdetermined least squares (solveInPlace, A is 2k x k; no Q reconstruction); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(TallLSFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(TallLSDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();

            sb.AppendLine("=== Wide LQ factorization (decomp, A is k x 2k); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();

            sb.AppendLine("=== Underdetermined minimum-norm (minNormSolve, A is k x 2k); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideMinNormFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideMinNormDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();

            sb.AppendLine("=== Wide LQRP row-pivoted factorization (decomp, A is k x 2k; forms L,Q,P; UNBLOCKED); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPDecompFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPDecompDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();

            sb.AppendLine("=== Underdetermined rank-safe basic solve (LQRP.solveInPlace, A is k x 2k; no Q); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPSolveFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPSolveDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();

            // README wide showcase: heavily underdetermined 512 x 2048 (4:1). Both compute the
            // minimum-norm solution in place; LQ needs full row rank, LQRP is rank-revealing (COD).
            sb.AppendLine("=== Underdetermined min-norm wide showcase (A is 512 x 2048): LQ.minNormSolveInPlace vs LQRP.minNormSolveInPlace ===");
            sb.AppendLine(HeaderKernel());
            sb.AppendLine(WideMinNormMNFloat(512, 2048));
            sb.AppendLine(WideMinNormMNDouble(512, 2048));
            sb.AppendLine(WideLQRPMinNormMNFloat(512, 2048));
            sb.AppendLine(WideLQRPMinNormMNDouble(512, 2048));
            sb.AppendLine();

            // COD overhead on a rank-deficient wide system: basic and COD row adjacent on the SAME matrix.
            sb.AppendLine("=== LQRP rank-deficient (k x 2k, rank = 3k/4): basic solveInPlace vs COD minNormSolveInPlace; N column = k ===");
            sb.AppendLine(HeaderKernel());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPRankDefFloat(k, QrFlops(2 * k, k)));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPRankDefDouble(k, QrFlops(2 * k, k)));
            sb.AppendLine();
        }

        static string HeaderKernel()
        {
            return string.Format("{0,-7} {1,-24} {2,-6} {3,11} {4,11} {5,11} {6,11} {7,12}",
                "dtype", "kernel", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~");
        }
    }
}
