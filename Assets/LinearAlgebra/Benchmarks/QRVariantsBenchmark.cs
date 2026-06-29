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

    // qrcpDirectSolve: QRCP-based rank-safe LS solve using the zero-alloc primitive.
    // A is copied into Q internally (A is NOT modified); b is read via dot(in b, in Q, ref x)
    // (b is NOT modified). Q, R, u are pre-allocated arena scratch. Pivot is per-Execute Temp alloc.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPSolveJobFloat : IJob
    {
        public floatMxN A;     // n x n input, NOT modified
        public floatN b;       // n, NOT modified
        public floatN x;       // n, solution output
        public floatMxN Q;     // n x n scratch (receives copy of A)
        public floatMxN R;     // n x n scratch
        public floatN u;       // n scratch

        public void Execute()
        {
            var P = new Pivot(A.N_Cols, Allocator.Temp);
            OrthoOP.qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, out int _);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b;
        public doubleN x;
        public doubleMxN Q;
        public doubleMxN R;
        public doubleN u;

        public void Execute()
        {
            var P = new Pivot(A.N_Cols, Allocator.Temp);
            OrthoOP.qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, out int _);
            P.Dispose();
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

            sb.AppendLine("=== qrcpDirectSolve (QRCP rank-safe LS solve; zero-alloc primitive) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPSolveFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPSolveDouble(n));
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

        static string QRCPSolveFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var b = arena.floatVec(n);
            var x = arena.floatVec(n);
            var Q = arena.floatMat(n, n);
            var R = arena.floatMat(n, n);
            var u = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                b[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new QRCPSolveJobFloat { A = A, b = b, x = x, Q = Q, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string QRCPSolveDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var x = arena.doubleVec(n);
            var Q = arena.doubleMat(n, n);
            var R = arena.doubleMat(n, n);
            var u = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                b[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new QRCPSolveJobDouble { A = A, b = b, x = x, Q = Q, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
