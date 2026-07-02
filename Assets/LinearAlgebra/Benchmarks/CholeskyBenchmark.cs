using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Cholesky factorization A = L L^T of a symmetric positive-definite matrix. A is taken `in`
    // (never mutated) and L is overwritten each run, so the SPD input is built once and every timed
    // sample does identical work. The input is symmetric + diagonally dominant, which guarantees SPD.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN L;

        public void Execute() => Cholesky.choleskyDecomposition(in A, ref L);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;

        public void Execute() => Cholesky.choleskyDecomposition(in A, ref L);
    }

    // Pivoted (rank-revealing) Cholesky P^T A P = L L^T. Same (1/3)N^3 factor work plus the
    // rank-revealing bookkeeping (largest-diagonal pivot search, symmetric row/col swaps, and the
    // Schur update's symmetric mirror), so its GFLOP/s is below the plain factorization by design.
    // The n x n working copy is a pre-built workspace (so the timing is the kernel, not a per-call
    // alloc); the O(n) Pivot is the only per-Execute allocation, as in the QRCP benchmark.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholPivotJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN L;
        public floatCholeskyPivot_WS ws;

        public void Execute()
        {
            var P = new Pivot(A.M_Rows, Allocator.Temp);
            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, ref ws);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholPivotJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;
        public doubleCholeskyPivot_WS ws;

        public void Execute()
        {
            var P = new Pivot(A.M_Rows, Allocator.Temp);
            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, ref ws);
            P.Dispose();
        }
    }

    public static class CholeskyBenchmark
    {
        // (1/3) N^3 is the standard leading term for Cholesky (half the work of LU).
        static double Flops(int n) => (1.0 / 3.0) * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-chol.txt.
        public static void Run() => Bench.WriteReport("benchmark-chol.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Cholesky factorization A = L L^T (SPD input) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Pivoted (rank-revealing) Cholesky P^T A P = L L^T (full-rank SPD input) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(PivotFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(PivotDouble(n));
            sb.AppendLine();
        }

        static string BenchFloat(int n)
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
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var L = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    A[i, j] = v;
                    A[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new CholJobDouble { A = A, L = L };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }

        static string PivotFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var L = arena.floatMat(n, n);
            var ws = arena.floatCholeskyPivot_WS(n);

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
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string PivotDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var L = arena.doubleMat(n, n);
            var ws = arena.doubleCholeskyPivot_WS(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    A[i, j] = v;
                    A[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new CholPivotJobDouble { A = A, L = L, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
