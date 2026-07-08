using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of CholeskyBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Flops, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/CholeskyBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN L;

        public void Execute() => CHO.decomp(in A, ref L);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholPivotJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN L;
        public floatCHOPCache ws;

        public void Execute()
        {
            var P = new Pivot(A.M_Rows, Allocator.Temp);
            CHOP.decomp(in A, ref L, ref P, ref ws);
            P.Dispose();
        }
    }

    public static partial class CholeskyBenchmark
    {
        static string BenchFloat(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var L = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;                // symmetric
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;                   // diagonal dominance => SPD

            var job = new CholJobFloat { A = A, L = L };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, flops);
        }

        static string PivotFloat(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var L = arena.floatMat(n, n);
            var ws = arena.floatCHOPCache(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;                // symmetric
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;                   // diagonal dominance => full-rank SPD

            var job = new CholPivotJobFloat { A = A, L = L, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, flops);
        }
    }
}
