using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.ML;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of KMeansBenchmark (timed IJob + build+measure method). The
    // dtype-agnostic harness (D/K constants, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/KMeansBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct KMeansJobFloat : IJob
    {
        public floatMxN X;                   // N x D, NOT modified
        public floatMxN centroids;           // k x D
        public Indices assignment;            // length N
        public floatKMeansCache ws;

        public void Execute() =>
            KMeans.fit(in X, 16, 12345u, 10, KMeansInit.Uniform,
                                 ref centroids, ref assignment, out float _, out int _, ref ws);
    }

    public static partial class KMeansBenchmark
    {
        static string BenchFloat(int n, int D, int K)
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
    }
}
