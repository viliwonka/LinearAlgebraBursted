using System.Text;

namespace BULA.Benchmarks
{
    // LU with partial pivoting. Each Execute factors a pristine Src into U/L via the safe LU.decomp
    // (which copies Src into U internally) and allocates a fresh Pivot in Temp (O(N), negligible vs
    // the O(N^3) factorization), so every timed sample does identical work.
    //
    // Hand-written harness half. The timed IJob (LUJob{Float,Double}) and build+measure method
    // (Bench{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LUBenchmark.fProxy.cs.
    public static partial class LUBenchmark
    {
        // (2/3) N^3 is the standard leading term for LU factorization.
        static double Flops(int n) => (2.0 / 3.0) * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-lu.txt.
        public static void Run() => Bench.WriteReport("benchmark-lu.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== LU factorization with partial pivoting (time = LU.decomp, copies Src internally) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== LU factorization, no pivoting (time = LU.decompNoPivot, copies Src internally) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchNoPivotFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchNoPivotDouble(n, Flops(n)));
            sb.AppendLine();
        }
    }
}
