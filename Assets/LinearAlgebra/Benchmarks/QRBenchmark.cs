using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // QR Householder factorization (also forms Q explicitly). Each Execute copies a pristine source
    // into the working matrix and factors it, so every timed sample does identical work
    // (decompInPlace overwrites its input). The O(N^2) copy against an O(N^3) factorization is
    // < 1% for N >= 128 and is included in the reported time.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRJobFloat : IJob
    {
        public floatMxN Q;
        public floatMxN R;
        public floatMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            QR.decompInPlace(ref Q, ref R);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRJobDouble : IJob
    {
        public doubleMxN Q;
        public doubleMxN R;
        public doubleMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            QR.decompInPlace(ref Q, ref R);
        }
    }

    public static class QRBenchmark
    {
        // (4/3) N^3 is the standard leading term for square Householder QR (approximate; forming Q
        // explicitly adds more, so GFLOP/s here is a lower bound on real work).
        static double Flops(int n) => (4.0 / 3.0) * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-qr.txt.
        public static void Run() => Bench.WriteReport("benchmark-qr.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== QR Householder factorization (time = copy-in + decompInPlace, forms Q) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();
        }

        static string BenchFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.floatMat(n, n);
            var R = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                 // diagonal dominance => full rank, no zero-column early-out

            var job = new QRJobFloat { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.doubleMat(n, n);
            var R = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRJobDouble { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
