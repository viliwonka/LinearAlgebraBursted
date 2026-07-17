using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of QueryBenchmark. A few common Query ops on big N x N matrices:
    // rowArgMin (per-row argmin, the k-means assignment primitive), argMaxRowNorm (row reduction ->
    // SIMD kernels) and argMaxColNorm (column reduction restructured into a row-major per-column
    // accumulate -- the colSum trick, so it now matches the row op), and nearestRow (a linear scan).
    // Hand-written harness: Assets/LinearAlgebra/Benchmarks/QueryBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QueryRowArgMinJobFProxy : IJob
    {
        public fProxyMxN A;
        public Indices Idx;
        public void Execute() => Query.rowArgMin(in A, ref Idx);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QueryArgMaxRowNormJobFProxy : IJob
    {
        public fProxyMxN A;
        public NativeArray<int> Out;
        public void Execute() => Out[0] = Query.argMaxRowNorm(in A, Norm.L2);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QueryArgMaxColNormJobFProxy : IJob
    {
        public fProxyMxN A;
        public NativeArray<int> Out;
        public void Execute() => Out[0] = Query.argMaxColNorm(in A, Norm.L2);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QueryNearestRowJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN Q;
        public NativeArray<int> Out;
        public void Execute()
        {
            Query.nearestRow(in A, in Q, Metric.Euclidean, out int idx, out fProxy _);
            Out[0] = idx;
        }
    }

    public static partial class QueryBenchmark
    {
        static fProxyMxN FillFProxy(Arena arena, int n)
        {
            var A = arena.fProxyMat(n, n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            return A;
        }

        static string RowArgMinFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = FillFProxy(arena, n);
            var idx = arena.Indices(n);
            var job = new QueryRowArgMinJobFProxy { A = A, Idx = idx };
            var stat = Bench.Time(() => job.Run());
            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string ArgMaxRowNormFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = FillFProxy(arena, n);
            var outv = new NativeArray<int>(1, Allocator.Persistent);
            var job = new QueryArgMaxRowNormJobFProxy { A = A, Out = outv };
            var stat = Bench.Time(() => job.Run());
            outv.Dispose(); arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string ArgMaxColNormFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = FillFProxy(arena, n);
            var outv = new NativeArray<int>(1, Allocator.Persistent);
            var job = new QueryArgMaxColNormJobFProxy { A = A, Out = outv };
            var stat = Bench.Time(() => job.Run());
            outv.Dispose(); arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string NearestRowFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = FillFProxy(arena, n);
            var q = arena.fProxyVec(n);
            var rng = new Unity.Mathematics.Random(0x9E3779B9u ^ (uint)n);
            for (int c = 0; c < n; c++) q[c] = rng.NextFProxy(-1f, 1f);
            var outv = new NativeArray<int>(1, Allocator.Persistent);
            var job = new QueryNearestRowJobFProxy { A = A, Q = q, Out = outv };
            var stat = Bench.Time(() => job.Run());
            outv.Dispose(); arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }
    }
}
