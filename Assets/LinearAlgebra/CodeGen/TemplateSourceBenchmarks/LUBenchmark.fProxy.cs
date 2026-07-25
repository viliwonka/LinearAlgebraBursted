using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of LUBenchmark (timed IJob + build+measure method). The
    // dtype-agnostic harness (Flops, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LUBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LUJobFProxy : IJob
    {
        public fProxyMxN U;
        public fProxyMxN L;
        public fProxyMxN Src;

        public void Execute()
        {
            int rows = Src.M_Rows;
            var P = new Pivot(rows, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LUNoPivotJobFProxy : IJob
    {
        public fProxyMxN U;
        public fProxyMxN L;
        public fProxyMxN Src;

        public void Execute()
        {
            LU.decompNoPivot(in Src, ref L, ref U);
        }
    }

    public static partial class LUBenchmark
    {
        static string BenchFProxy(int n, double flops)
        {
            var U = new fProxyMxN(n, n, Allocator.Persistent);
            var L = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                 // diagonal dominance => well-conditioned, full rank

            var job = new LUJobFProxy { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            U.Dispose();
            L.Dispose();
            Src.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchNoPivotFProxy(int n, double flops)
        {
            var U = new fProxyMxN(n, n, Allocator.Persistent);
            var L = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                 // diagonal dominance => no pivoting needed, full rank

            var job = new LUNoPivotJobFProxy { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            U.Dispose();
            L.Dispose();
            Src.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }
    }
}
