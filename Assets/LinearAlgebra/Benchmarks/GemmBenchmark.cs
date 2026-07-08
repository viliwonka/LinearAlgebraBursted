using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Dense matrix-matrix product C = A * B (the GEMM that most higher-level ops bottom out on, and
    // the cleanest signal for whether the inner kernel vectorises: a SIMD float path should run
    // roughly 2x a double path of the same N). C is preallocated and overwritten each run, so every
    // timed sample does identical work and the inputs are never mutated.
    //
    // Hand-written harness half. The timed IJob (GemmJob{Float,Double}) and build+measure method
    // (Bench{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/GemmBenchmark.fProxy.cs.
    public static partial class GemmBenchmark
    {
        // 2 N^3: one multiply + one add per inner-product term over N^3 terms.
        static double Flops(int n) => 2.0 * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-gemm.txt.
        public static void Run() => Bench.WriteReport("benchmark-gemm.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== GEMM: dense C = A * B ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n, Flops(n)));
            sb.AppendLine();
        }
    }
}
