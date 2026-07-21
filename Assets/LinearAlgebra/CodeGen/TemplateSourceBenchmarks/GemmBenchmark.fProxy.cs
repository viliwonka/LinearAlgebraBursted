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

        public void Execute()
        {
            BurstProbe.RequireBursted();
            Blas.dot(in A, in B, ref C);
        }
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

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmTransBJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN C;

        public void Execute() => Blas.dot(in A, in B, ref C, transposeA: false, transposeB: true);
    }

    // Baseline route the TransB kernel replaces: materialize Bᵀ (Temp alloc + strided transpose),
    // then plain GEMM.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmTransBViaTransJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN C;

        public void Execute()
        {
            var Bt = new fProxyMxN(B.N_Cols, B.M_Rows, Allocator.Temp, true);
            Blas.trans(in B, ref Bt);
            Blas.dot(in A, in Bt, ref C);
            Bt.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GemmAAtJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN C;

        public void Execute() => Blas.dot(in A, in A, ref C, transposeA: false, transposeB: true);
    }

    // Same-run control for the wide-tile A/B: calls the scalar register tile directly,
    // bypassing matMatDot's routing.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public unsafe struct GemmScalarTileJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN C;

        public void Execute()
        {
            Unity.Collections.LowLevel.Unsafe.UnsafeUtility.MemClear(
                C.Data.Ptr, (long)C.Data.Length * Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SizeOf<fProxy>());
            LinearAlgebra.Internal.UnsafeOP.matMatDotUnpacked(A.Data.Ptr, B.Data.Ptr, C.Data.Ptr, A.M_Rows, A.N_Cols, B.N_Cols);
        }
    }

    // Same-run control for the packed route: the packed driver called directly, bypassing
    // matMatDot's working-set gate, so the pack-copy overhead below the gate stays visible.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public unsafe struct GemmPackedJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN C;

        public void Execute()
        {
            Unity.Collections.LowLevel.Unsafe.UnsafeUtility.MemClear(
                C.Data.Ptr, (long)C.Data.Length * Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SizeOf<fProxy>());
            LinearAlgebra.Internal.UnsafeOP.matMatDotPacked(A.Data.Ptr, B.Data.Ptr, C.Data.Ptr, A.M_Rows, A.N_Cols, B.N_Cols);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TransJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN T;

        public void Execute() => Blas.trans(in A, ref T);
    }

    public static partial class GemmBenchmark
    {
        static string BenchTransBFProxy(int n, double flops)
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

            var job = new GemmTransBJobFProxy { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchTransBViaTransFProxy(int n, double flops)
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

            var job = new GemmTransBViaTransJobFProxy { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchScalarTileFProxy(int n, double flops)
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

            var job = new GemmScalarTileJobFProxy { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchPackedFProxy(int n, double flops)
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

            var job = new GemmPackedJobFProxy { A = A, B = B, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchTransFProxy(int n, double elems)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(n, n);
            var T = arena.fProxyMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = rng.NextFProxy(-1f, 1f);

            var job = new TransJobFProxy { A = A, T = T };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, elems);
        }

        static string BenchAAtFProxy(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(n, n);
            var C = arena.fProxyMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = rng.NextFProxy(-1f, 1f);

            var job = new GemmAAtJobFProxy { A = A, C = C };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

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
