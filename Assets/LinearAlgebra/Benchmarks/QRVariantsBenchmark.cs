using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // The other Householder paths that share QR's reflector-apply hot loop: column-pivoted QR (QRCP,
    // rank-revealing) and the direct least-squares solve. Each Execute copies a pristine source into
    // the working matrix so every timed sample does identical work.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPJobFloat : IJob
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

            var P = new Pivot(Q.N_Cols, Allocator.Temp);
            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPJobDouble : IJob
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

            var P = new Pivot(Q.N_Cols, Allocator.Temp);
            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRSolveJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN b;
        public floatN bSrc;
        public floatN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            OrthoOP.qrDirectSolve(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;
        public doubleN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            OrthoOP.qrDirectSolve(ref A, ref b, ref x);
        }
    }

    public static class QRVariantsBenchmark
    {
        // (4/3) N^3 leading term (approximate). QRCP adds an O(N^3) exact pivot-norm recompute on top,
        // and qrDirectSolve skips the Q reconstruction, so GFLOP/s here is only a rough comparator —
        // the time columns and the A/B speedup are the honest signal.
        static double Flops(int n) => (4.0 / 3.0) * n * (double)n * n;

        public static void Run() => Bench.WriteReport("benchmark-qrvariants.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== QRCP (column-pivoted, rank-revealing QR; forms Q) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== qrDirectSolve (Householder least-squares solve; no Q reconstruction) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(SolveFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(SolveDouble(n));
            sb.AppendLine();
        }

        static string QRCPFloat(int n)
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
                Src[d, d] += n;

            var job = new QRCPJobFloat { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string QRCPDouble(int n)
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

            var job = new QRCPJobDouble { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }

        static string SolveFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var b = arena.floatVec(n);
            var bSrc = arena.floatVec(n);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string SolveDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var bSrc = arena.doubleVec(n);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
