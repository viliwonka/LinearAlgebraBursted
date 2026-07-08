using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LUBenchmark (timed IJob + build+measure method). The
    // dtype-agnostic harness (Flops, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LUBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LUJobFloat : IJob
    {
        public floatMxN U;
        public floatMxN L;
        public floatMxN Src;

        public void Execute()
        {
            int rows = Src.M_Rows;
            var P = new Pivot(rows, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            P.Dispose();
        }
    }

    public static partial class LUBenchmark
    {
        static string BenchFloat(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.floatMat(n, n);
            var L = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                 // diagonal dominance => well-conditioned, full rank

            var job = new LUJobFloat { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, flops);
        }
    }
}
