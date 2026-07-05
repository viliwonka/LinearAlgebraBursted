using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // LU with partial pivoting. Each Execute factors a pristine Src into U/L via the safe LU.decomp
    // (which copies Src into U internally) and allocates a fresh Pivot in Temp (O(N), negligible vs
    // the O(N^3) factorization), so every timed sample does identical work.

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

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LUJobDouble : IJob
    {
        public doubleMxN U;
        public doubleMxN L;
        public doubleMxN Src;

        public void Execute()
        {
            int rows = Src.M_Rows;
            var P = new Pivot(rows, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            P.Dispose();
        }
    }

    public static class LUBenchmark
    {
        // (2/3) N^3 is the standard leading term for LU factorization.
        static double Flops(int n) => (2.0 / 3.0) * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-lu.txt.
        public static void Run() => Bench.WriteReport("benchmark-lu.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== LU factorization with partial pivoting (time = LU.decomp, copies Src internally) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();
        }

        static string BenchFloat(int n)
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
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.doubleMat(n, n);
            var L = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new LUJobDouble { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
