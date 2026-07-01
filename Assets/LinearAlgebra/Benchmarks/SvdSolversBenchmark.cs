using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Randomized SVD (top-k, k=16) and the two pseudo-inverse solvers that ride the Golub-Kahan SVD.
    // All three are time-only: the cost is dominated by iterative convergence, so GFLOP/s would be
    // misleading. A is never modified; workspaces are built once outside the timing loop.

    // ---- svdRandomized ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdRandomizedJobFloat : IJob
    {
        public floatMxN A;          // n x n input, NOT modified
        public floatMxN Uk;         // n x k
        public floatN Sk;           // length k
        public floatMxN Vk;         // n x k
        public floatSVDRandomized_WS ws;

        public void Execute() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, 16, ref ws);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdRandomizedJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Uk;
        public doubleN Sk;
        public doubleMxN Vk;
        public doubleSVDRandomized_WS ws;

        public void Execute() => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, 16, ref ws);
    }

    // ---- pinvSolve ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PinvSolveJobFloat : IJob
    {
        public floatMxN A;          // n x n, NOT modified by Golub-Kahan path
        public floatN b;
        public floatN x;
        public floatSVD_WS ws;

        public void Execute() => SVD.pinvSolve(ref A, in b, ref x, out bool _, ref ws);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PinvSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b;
        public doubleN x;
        public doubleSVD_WS ws;

        public void Execute() => SVD.pinvSolve(ref A, in b, ref x, out bool _, ref ws);
    }

    // ---- pseudoInverse ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PseudoInverseJobFloat : IJob
    {
        public floatMxN A;          // n x n, NOT modified by Golub-Kahan path
        public floatMxN Aplus;      // n x n
        public floatSVD_WS ws;

        public void Execute() => SVD.pseudoInverse(ref A, ref Aplus, out bool _, ref ws);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PseudoInverseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Aplus;
        public doubleSVD_WS ws;

        public void Execute() => SVD.pseudoInverse(ref A, ref Aplus, out bool _, ref ws);
    }

    public static class SvdSolversBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-svdsolvers.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== svdRandomized (Halko-Martinsson-Tropp, top-k singular triplets, k=16; ms) ===");
            sb.AppendLine("    Randomized low-rank path: GEMM sketch -> QR range-finder -> small exact SVD.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdRandFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdRandDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== pinvSolve (Moore-Penrose minimum-norm LS solve via Golub-Kahan SVD; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(PinvFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(PinvDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== pseudoInverse (Moore-Penrose pseudo-inverse matrix via Golub-Kahan SVD; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(PseudoInvFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(PseudoInvDouble(n));
            sb.AppendLine();
        }

        // ---- svdRandomized ----

        static string SvdRandFloat(int n)
        {
            const int k = 16;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Uk = arena.floatMat(n, k);
            var Sk = arena.floatVec(k);
            var Vk = arena.floatMat(n, k);
            var ws = arena.floatSVDRandomized_WS(n, n, k);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);

            var job = new SvdRandomizedJobFloat { A = A, Uk = Uk, Sk = Sk, Vk = Vk, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string SvdRandDouble(int n)
        {
            const int k = 16;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Uk = arena.doubleMat(n, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            var ws = arena.doubleSVDRandomized_WS(n, n, k);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);

            var job = new SvdRandomizedJobDouble { A = A, Uk = Uk, Sk = Sk, Vk = Vk, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- pinvSolve ----

        static string PinvFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var b = arena.floatVec(n);
            var x = arena.floatVec(n);
            var ws = arena.floatSVD_WS(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                A[d, d] += n;
            for (int i = 0; i < n; i++)
                b[i] = rng.NextFloat(-1f, 1f);

            var job = new PinvSolveJobFloat { A = A, b = b, x = x, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string PinvDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var x = arena.doubleVec(n);
            var ws = arena.doubleSVD_WS(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                A[d, d] += n;
            for (int i = 0; i < n; i++)
                b[i] = rng.NextDouble(-1.0, 1.0);

            var job = new PinvSolveJobDouble { A = A, b = b, x = x, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- pseudoInverse ----

        static string PseudoInvFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Aplus = arena.floatMat(n, n);
            var ws = arena.floatSVD_WS(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new PseudoInverseJobFloat { A = A, Aplus = Aplus, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string PseudoInvDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Aplus = arena.doubleMat(n, n);
            var ws = arena.doubleSVD_WS(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new PseudoInverseJobDouble { A = A, Aplus = Aplus, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
