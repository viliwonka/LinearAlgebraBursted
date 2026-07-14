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

            sb.AppendLine("=== GEMM-scalar-tile: matMatDotUnpacked direct (same-run control for the wide tile) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchScalarTileFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchScalarTileDouble(n, Flops(n)));
            sb.AppendLine();

            // Above Bench.Sizes' top: where the packed (cache-blocked) route earns its keep.
            int[] largeSizes = { 1536, 2048 };
            sb.AppendLine("=== GEMM-large: dense C = A * B, beyond-L2 sizes (packed route) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in largeSizes) sb.AppendLine(BenchFloat(n, Flops(n)));
            foreach (var n in largeSizes) sb.AppendLine(BenchDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== GEMM-TransA: dense C = A^T * B (covariance / compact-WY shape) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchTransAFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchTransADouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== GEMM-AtA: dense C = A^T * A (matAtA single-input kernel) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchAtAFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchAtADouble(n, Flops(n)));
            sb.AppendLine();

            // TransB sections add small sizes: the Kalman/KMeans shapes this kernel serves live
            // at small n, and the trans+dot comparison needs the crossover visible.
            int[] tbSizes = { 16, 32, 64, 128, 256, 512, 1024 };

            sb.AppendLine("=== GEMM-TransB: dense C = A * B^T (Kalman P*H^T / KMeans X*C^T shape) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in tbSizes) sb.AppendLine(BenchTransBFloat(n, Flops(n)));
            foreach (var n in tbSizes) sb.AppendLine(BenchTransBDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== GEMM-TransB-viaTrans: C = A * trans(B) (the Blas.trans + dot route TransB replaces) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in tbSizes) sb.AppendLine(BenchTransBViaTransFloat(n, Flops(n)));
            foreach (var n in tbSizes) sb.AppendLine(BenchTransBViaTransDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== GEMM-AAt: dense C = A * A^T (matAAt single-input kernel) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in tbSizes) sb.AppendLine(BenchAAtFloat(n, Flops(n)));
            foreach (var n in tbSizes) sb.AppendLine(BenchAAtDouble(n, Flops(n)));
            sb.AppendLine();

            // Transpose moves N^2 elements and does zero flops; the flop column is fed N^2 so it
            // reads as Gelem/s (element throughput), not GFLOP/s.
            sb.AppendLine("=== Trans: T = A^T (element throughput; last column is Gelem/s) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in tbSizes) sb.AppendLine(BenchTransFloat(n, 1.0 * n * n));
            foreach (var n in tbSizes) sb.AppendLine(BenchTransDouble(n, 1.0 * n * n));
            sb.AppendLine();
        }
    }
}
