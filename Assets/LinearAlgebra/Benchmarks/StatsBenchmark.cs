using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Row-major matrix-stats reduction/transform family on N x N matrices. Exists to give the
    // raw-pointer hoist pass an A/B measurement (these methods were on the struct indexer).
    //
    // Hand-written harness half. The timed IJobs and build+measure methods (RowSum{Float,Double},
    // etc.) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/StatsBenchmark.fProxy.cs.
    public static partial class StatsBenchmark
    {
        // One pass over N*N elements is the leading term for every kernel here (variance/standardize
        // are two-pass but still O(N^2)); report elements-touched so the column reads as ~G-elem/s.
        static double Flops(int n) => (double)n * n;

        public static void Run() => Bench.WriteReport("benchmark-stats.txt", Section);

        public static void Section(StringBuilder sb)
        {
            Sub(sb, "Stats.rowSum (row reduction)",       RowSumFloat,      RowSumDouble);
            Sub(sb, "Stats.colSum (col accumulate)",      ColSumFloat,      ColSumDouble);
            Sub(sb, "Stats.rowVariance (two-pass)",       RowVarFloat,      RowVarDouble);
            Sub(sb, "Stats.standardizeRows (in-place)",   StdRowsFloat,     StdRowsDouble);
            Sub(sb, "Stats.softmaxRows (in-place, exp)",  SoftmaxRowsFloat, SoftmaxRowsDouble);
        }

        delegate string Measure(int n, double flops);

        static void Sub(StringBuilder sb, string title, Measure f, Measure d)
        {
            sb.AppendLine($"=== {title} ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(f(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(d(n, Flops(n)));
            sb.AppendLine();
        }
    }
}
