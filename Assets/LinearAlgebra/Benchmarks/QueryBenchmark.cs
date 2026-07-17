using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // A few common Query ops on N x N matrices. Row-inner ops (rowArgMin, argMaxRowNorm) are
    // unit-stride and vectorise; argMaxColNorm is column-inner (strided) and stays scalar -- the
    // row/column asymmetry, benched side by side.
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per
    // dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QueryBenchmark.fProxy.cs.
    public static partial class QueryBenchmark
    {
        // One pass over N*N elements is the leading term; report elements-touched (~G-elem/s).
        static double Flops(int n) => (double)n * n;

        public static void Run() => Bench.WriteReport("benchmark-query.txt", Section);

        public static void Section(StringBuilder sb)
        {
            Sub(sb, "Query.rowArgMin (per-row argmin)",         RowArgMinFloat,     RowArgMinDouble);
            Sub(sb, "Query.argMaxRowNorm L2 (row-inner)",       ArgMaxRowNormFloat, ArgMaxRowNormDouble);
            Sub(sb, "Query.argMaxColNorm L2 (column-inner)",    ArgMaxColNormFloat, ArgMaxColNormDouble);
            Sub(sb, "Query.nearestRow (Euclidean scan)",        NearestRowFloat,    NearestRowDouble);
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
