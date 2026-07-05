using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;
using LinearAlgebra.ML;

namespace LinearAlgebra.Benchmarks
{
    // Lloyd k-means clustering with GEMM-accelerated assignment.
    // Fixed D = 64 features, k = 16 clusters, maxIter = 10, Uniform init, seed = 12345u.
    // N varies over Bench.Sizes (number of points). X is N×D, built once (in — not modified).
    // time-only: iteration count per call is data-dependent (at most maxIter), so GFLOP/s misleads.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct KMeansJobFloat : IJob
    {
        public floatMxN X;                   // N x D, NOT modified
        public floatMxN centroids;           // k x D
        public Indices assignment;           // length N
        public floatKMeansCache ws;

        public void Execute() =>
            KMeans.fit(in X, 16, 12345u, 10, KMeansInit.Uniform,
                                 ref centroids, ref assignment, out float _, out int _, ref ws);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct KMeansJobDouble : IJob
    {
        public doubleMxN X;
        public doubleMxN centroids;
        public Indices assignment;
        public doubleKMeansCache ws;

        public void Execute() =>
            KMeans.fit(in X, 16, 12345u, 10, KMeansInit.Uniform,
                                  ref centroids, ref assignment, out double _, out int _, ref ws);
    }

    public static class KMeansBenchmark
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
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();
        }

        static string BenchFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var X = arena.floatMat(n, D);
            var centroids = arena.floatMat(K, D);
            var assignment = arena.Indices(n);
            var ws = arena.floatKMeansCache(n, D, K);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < D; c++)
                    X[r, c] = rng.NextFloat(-1f, 1f);

            var job = new KMeansJobFloat { X = X, centroids = centroids, assignment = assignment, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var X = arena.doubleMat(n, D);
            var centroids = arena.doubleMat(K, D);
            var assignment = arena.Indices(n);
            var ws = arena.doubleKMeansCache(n, D, K);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < D; c++)
                    X[r, c] = rng.NextDouble(-1.0, 1.0);

            var job = new KMeansJobDouble { X = X, centroids = centroids, assignment = assignment, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
