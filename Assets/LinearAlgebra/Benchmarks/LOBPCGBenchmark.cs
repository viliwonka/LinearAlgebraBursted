using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // LOBPCG.lobpcg (k smallest eigenpairs of a symmetric operator) — not covered by
    // EigenSvdBenchmark.cs (that file predates LOBPCG's generalization). Dense SPD input
    // A = MᵀM + I (same recipe as IterativeBenchmark.cs — guarantees SPD, well-conditioned smallest
    // eigenvalues clustered near 1). tol is set near machine-epsilon so every timed sample runs the
    // full maxIter budget (deterministic timing, mirroring the fixed-K convention used by the other
    // iterative-solver benchmarks); iterations/converged/maxResidual are reported alongside timing to
    // show how far that fixed budget actually gets, not just how fast.

    // info is written into a length-1 NativeArray, not a plain struct field: IJob.Run() executes on
    // an internal copy of the job struct, so a plain value-type field written inside Execute() is NOT
    // visible on the caller's job variable afterwards (only pointer-backed data — NativeArray/Arena
    // buffers — survives the copy). Every other benchmark in this folder sidesteps this by only ever
    // reading back Arena-backed buffers (floatN/floatMxN); LOBPCGInfo has no such buffer to piggyback
    // on, so it gets its own one-element NativeArray output.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgJobFloat : IJob
    {
        public floatMxN A;      // SPD, not modified (dense forwarder copies internally as needed)
        public floatLOBPCGCache ws;
        public int k, maxIter;
        public float tol;
        public NativeArray<LOBPCGInfo> infoOut; // length 1

        public void Execute() => infoOut[0] = LOBPCG.lobpcg(in A, ref ws, k, tol, maxIter);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgJobDouble : IJob
    {
        public doubleMxN A;
        public doubleLOBPCGCache ws;
        public int k, maxIter;
        public double tol;
        public NativeArray<LOBPCGInfo> infoOut; // length 1

        public void Execute() => infoOut[0] = LOBPCG.lobpcg(in A, ref ws, k, tol, maxIter);
    }

    public static class LOBPCGBenchmark
    {
        const int N = 512;
        const int K = 4;         // smallest eigenpairs requested
        const int MaxIter = 50;  // fixed budget for deterministic timing

        public static void Run() => Bench.WriteReport("benchmark-lobpcg.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine(string.Format("=== LOBPCG.lobpcg, dense SPD (A = M^T M + I), k={0} smallest, maxIter={1} (ms) ===", K, MaxIter));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,11} {3,11} {4,10} {5,10} {6,14}",
                "dtype", "N", "min(ms)", "med(ms)", "iters", "converged", "maxResidual"));
            sb.AppendLine(BenchFloat());
            sb.AppendLine(BenchDouble());
            sb.AppendLine();
        }

        static string BenchFloat()
        {
            var arena = new Arena(Allocator.Persistent);
            var M = arena.floatMat(N, N);
            var A = arena.floatMat(N, N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    M[r, c] = rng.NextFloat(-1f, 1f);
            Blas.dot(in M, in M, ref A, true);
            for (int d = 0; d < N; d++) A[d, d] += 1f;

            var ws = arena.floatLOBPCGCache(N, K);
            var infoOut = new NativeArray<LOBPCGInfo>(1, Allocator.Persistent);
            var job = new LobpcgJobFloat { A = A, ws = ws, k = K, maxIter = MaxIter, tol = 1e-20f, infoOut = infoOut };
            var stat = Bench.Time(() => job.Run());

            var info = infoOut[0];
            string row = string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,10} {5,10} {6,14:E3}",
                "float", N, stat.Min, stat.Median, info.iterations, info.converged, info.maxResidual);

            infoOut.Dispose();
            arena.Dispose();
            return row;
        }

        static string BenchDouble()
        {
            var arena = new Arena(Allocator.Persistent);
            var M = arena.doubleMat(N, N);
            var A = arena.doubleMat(N, N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    M[r, c] = rng.NextDouble(-1.0, 1.0);
            Blas.dot(in M, in M, ref A, true);
            for (int d = 0; d < N; d++) A[d, d] += 1.0;

            var ws = arena.doubleLOBPCGCache(N, K);
            var infoOut = new NativeArray<LOBPCGInfo>(1, Allocator.Persistent);
            var job = new LobpcgJobDouble { A = A, ws = ws, k = K, maxIter = MaxIter, tol = 1e-20, infoOut = infoOut };
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
