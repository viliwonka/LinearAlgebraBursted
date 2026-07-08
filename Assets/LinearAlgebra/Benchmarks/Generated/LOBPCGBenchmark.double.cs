using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LOBPCGBenchmark: the timed IJob plus the build+measure method.
    // The dtype-agnostic harness (constants, Run, Section) lives in the hand-written partial in
    // Assets/LinearAlgebra/Benchmarks/LOBPCGBenchmark.cs. See that file for what this benchmark measures.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgJobDouble : IJob
    {
        public doubleMxN A;      // SPD, not modified (dense forwarder copies internally as needed)
        public doubleLOBPCGCache ws;
        public int k, maxIter;
        public double tol;
        public NativeArray<LOBPCGInfo> infoOut; // length 1

        public void Execute() => infoOut[0] = Eigen.lobpcg(in A, ref ws, k, tol, maxIter);
    }

    public static partial class LOBPCGBenchmark
    {
        static string BenchDouble(int N, int K, int maxIter)
        {
            var arena = new Arena(Allocator.Persistent);
            var M = arena.doubleMat(N, N);
            var A = arena.doubleMat(N, N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    M[r, c] = rng.NextDouble(-1f, 1f);
            Blas.dot(in M, in M, ref A, true);
            for (int d = 0; d < N; d++) A[d, d] += (double)1;

            var ws = arena.doubleLOBPCGCache(N, K);
            var infoOut = new NativeArray<LOBPCGInfo>(1, Allocator.Persistent);
            var job = new LobpcgJobDouble { A = A, ws = ws, k = K, maxIter = maxIter, tol = (double)1e-20, infoOut = infoOut };
            var stat = Bench.Time(() => job.Run());

            var info = infoOut[0];
            string row = string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,10} {5,10} {6,14:E3}",
                "double", N, stat.Min, stat.Median, info.iterations, info.converged, info.maxResidual);

            infoOut.Dispose();
            arena.Dispose();
            return row;
        }
    }
}
