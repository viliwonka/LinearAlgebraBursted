using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of GemmBenchmark (timed IJob + build+measure method). The
    // dtype-agnostic harness (Flops, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/GemmBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN C;

        public void Execute() => Blas.dot(in A, in B, ref C);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmTransAJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN C;

        public void Execute() => Blas.dot(in A, in B, ref C, transposeA: true);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmAtAJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN C;

        public void Execute() => Blas.dot(in A, in A, ref C, transposeA: true);
    }

    public static partial class GemmBenchmark
    {
        static string BenchAtAFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(n, n);
            var C = arena.fProxyMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = rng.NextFProxy(-1f, 1f);

            var job = new GemmAtAJobFProxy { A = A, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(n, n);
            var B = arena.fProxyMat(n, n);
            var C = arena.fProxyMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    A[i, j] = rng.NextFProxy(-1f, 1f);
                    B[i, j] = rng.NextFProxy(-1f, 1f);
                }

            var job = new GemmJobFProxy { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchTransAFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(n, n);
            var B = arena.fProxyMat(n, n);
            var C = arena.fProxyMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    A[i, j] = rng.NextFProxy(-1f, 1f);
                    B[i, j] = rng.NextFProxy(-1f, 1f);
                }

            var job = new GemmTransAJobFProxy { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }
    }
}
