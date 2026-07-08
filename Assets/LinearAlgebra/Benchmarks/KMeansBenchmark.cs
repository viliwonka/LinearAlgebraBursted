using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Lloyd k-means clustering with GEMM-accelerated assignment.
    // Fixed D = 64 features, k = 16 clusters, maxIter = 10, Uniform init, seed = 12345u.
    // N varies over Bench.Sizes (number of points). X is N×D, built once (in — not modified).
    // time-only: iteration count per call is data-dependent (at most maxIter), so GFLOP/s misleads.
    //
    // Hand-written harness half. The timed IJob (KMeansJob{Float,Double}) and build+measure method
    // (Bench{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KMeansBenchmark.fProxy.cs.
    public static partial class KMeansBenchmark
    {
        const int D = 64;
        const int K = 16;

        public static void Run() => Bench.WriteReport("benchmark-fit.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== k-means (N points, D=64 features, k=16 clusters, maxIter=10, Uniform init; ms) ===");
            sb.AppendLine("    GEMM-accelerated assignment: O(N·D·k) per iteration via X·Cᵀ GEMM.");
            sb.AppendLine("    N column = number of points; D and k are fixed.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n, D, K));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n, D, K));
            sb.AppendLine();
        }
    }
}
