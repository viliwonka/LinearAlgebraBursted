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
    public struct KMeansJobFProxy : IJob
    {
        public fProxyMxN X;                   // N x D, NOT modified
        public fProxyMxN centroids;           // k x D
        public Indices assignment;            // length N
        public fProxyKMeansCache ws;

        public void Execute() =>
            KMeans.fit(in X, 16, 12345u, 10, KMeansInit.Uniform,
                                 ref centroids, ref assignment, out fProxy _, out int _, ref ws);
    }

    public static partial class KMeansBenchmark
    {
        static string BenchFProxy(int n, int D, int K)
        {
            var arena = new Arena(Allocator.Persistent);
            var X = arena.fProxyMat(n, D);
            var centroids = arena.fProxyMat(K, D);
            var assignment = arena.Indices(n);
            var ws = arena.fProxyKMeansCache(n, D, K);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < D; c++)
                    X[r, c] = rng.NextFProxy(-1f, 1f);

            var job = new KMeansJobFProxy { X = X, centroids = centroids, assignment = assignment, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }
    }
}
