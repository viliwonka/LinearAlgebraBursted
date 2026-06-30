using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Dense matrix-matrix product C = A * B (the GEMM that most higher-level ops bottom out on, and
    // the cleanest signal for whether the inner kernel vectorises: a SIMD float path should run
    // roughly 2x a double path of the same N). C is preallocated and overwritten each run, so every
    // timed sample does identical work and the inputs are never mutated.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN B;
        public floatMxN C;

        public void Execute() => float_OP.dot(in A, in B, ref C);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN B;
        public doubleMxN C;

        public void Execute() => double_OP.dot(in A, in B, ref C);
    }

    public static class GemmBenchmark
    {
        // 2 N^3: one multiply + one add per inner-product term over N^3 terms.
        static double Flops(int n) => 2.0 * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-gemm.txt.
        public static void Run() => Bench.WriteReport("benchmark-gemm.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== GEMM: dense C = A * B ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();
        }

        static string BenchFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var B = arena.floatMat(n, n);
            var C = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    A[i, j] = rng.NextFloat(-1f, 1f);
                    B[i, j] = rng.NextFloat(-1f, 1f);
                }

            var job = new GemmJobFloat { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var B = arena.doubleMat(n, n);
            var C = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    A[i, j] = rng.NextDouble(-1.0, 1.0);
                    B[i, j] = rng.NextDouble(-1.0, 1.0);
                }

            var job = new GemmJobDouble { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
