using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of StatsBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Flops, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/StatsBenchmark.cs.
    //
    // Covers the row-major matrix-stats reduction/transform family (rowSum, colSum, rowVariance,
    // standardizeRows, softmaxRows) so the raw-pointer hoist pass has an A/B measurement.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct StatsRowSumJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN Dest;
        public void Execute() => Stats.rowSum(in A, ref Dest);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct StatsColSumJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN Dest;
        public void Execute() => Stats.colSum(in A, ref Dest);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct StatsRowVarJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN Dest;
        public void Execute() => Stats.rowVariance(in A, ref Dest);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct StatsStdRowsJobFProxy : IJob
    {
        public fProxyMxN A;
        public void Execute() => Stats.standardizeRows(ref A);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct StatsSoftmaxRowsJobFProxy : IJob
    {
        public fProxyMxN A;
        public void Execute() => Stats.softmaxRows(ref A);
    }

    public static partial class StatsBenchmark
    {
        static fProxyMxN FillFProxy(Allocator allocator, int n)
        {
            var A = new fProxyMxN(n, n, allocator);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            return A;
        }

        static string RowSumFProxy(int n, double flops)
        {
            var A = FillFProxy(Allocator.Persistent, n);
            var dest = new fProxyN(n, Allocator.Persistent);
            var job = new StatsRowSumJobFProxy { A = A, Dest = dest };
            var stat = Bench.Time(() => job.Run());
            A.Dispose(); dest.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string ColSumFProxy(int n, double flops)
        {
            var A = FillFProxy(Allocator.Persistent, n);
            var dest = new fProxyN(n, Allocator.Persistent);
            var job = new StatsColSumJobFProxy { A = A, Dest = dest };
            var stat = Bench.Time(() => job.Run());
            A.Dispose(); dest.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string RowVarFProxy(int n, double flops)
        {
            var A = FillFProxy(Allocator.Persistent, n);
            var dest = new fProxyN(n, Allocator.Persistent);
            var job = new StatsRowVarJobFProxy { A = A, Dest = dest };
            var stat = Bench.Time(() => job.Run());
            A.Dispose(); dest.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string StdRowsFProxy(int n, double flops)
        {
            var A = FillFProxy(Allocator.Persistent, n);
            var job = new StatsStdRowsJobFProxy { A = A };
            var stat = Bench.Time(() => job.Run());
            A.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string SoftmaxRowsFProxy(int n, double flops)
        {
            var A = FillFProxy(Allocator.Persistent, n);
            var job = new StatsSoftmaxRowsJobFProxy { A = A };
            var stat = Bench.Time(() => job.Run());
            A.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }
    }
}
