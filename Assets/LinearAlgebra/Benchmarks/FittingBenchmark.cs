using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic row formatter. Public so the code-generated per-dtype build methods
    // (in a separate template assembly) can call it — same pattern as TallWideFmt.
    public static class FittingFmt
    {
        public static string Fmt(string dtype, string method, int m, int n, Bench.Stat st)
        {
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-22} {2,-6} {3,-5} {4,11:F4} {5,11:F4} {6,11:F4} {7,11:F4}",
                dtype, method, m, n, st.Min, st.Median, st.Mean, st.Max);
        }
    }

    // Linear regression fit comparison: L2 (QR least squares) vs exact L1 (LP.lad) vs approximate L1
    // (Optimize.ladIRLS), same design matrix + 5%-gross-outlier response. Response-residual fitting,
    // NOT total least squares / total least deviation (those minimize orthogonal distance).
    //
    // Hand-written harness half. The timed IJobs and build+measure method (FitFloat/FitDouble) are
    // code-generated per dtype from CodeGen/TemplateSourceBenchmarks/FittingBenchmark.fProxy.cs.
    public static partial class FittingBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-fitting.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Regression fitting (m obs x n coeffs, 5% gross outliers): L2 QR vs exact L1 LP.lad vs approx L1 ladIRLS ===");
            sb.AppendLine("    Response-residual fitting (ordinary regression), NOT orthogonal distance (total least squares/deviation).");
            sb.AppendLine(string.Format("{0,-7} {1,-22} {2,-6} {3,-5} {4,11} {5,11} {6,11} {7,11}",
                "dtype", "method", "m", "n", "min(ms)", "med(ms)", "mean(ms)", "max(ms)"));
            sb.AppendLine(FitFloat(2048, 4));
            sb.AppendLine(FitDouble(2048, 4));
            sb.AppendLine(FitFloat(2048, 64));
            sb.AppendLine(FitDouble(2048, 64));
            sb.AppendLine();
        }
    }
}
