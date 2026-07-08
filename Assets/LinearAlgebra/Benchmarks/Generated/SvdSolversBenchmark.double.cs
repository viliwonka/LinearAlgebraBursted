using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of SvdSolversBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/SvdSolversBenchmark.cs.

    // ---- randomized ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdRandomizedJobDouble : IJob
    {
        public doubleMxN A;          // n x n input, NOT modified
        public doubleMxN Uk;         // n x k
        public doubleN Sk;           // length k
        public doubleMxN Vk;         // n x k
        public doubleSVDRandomizedCache ws;

        public void Execute() => SVD.randomized(in A, ref Uk, ref Sk, ref Vk, 16, ref ws);
    }

    // ---- pinvSolve ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PinvSolveJobDouble : IJob
    {
        public doubleMxN A;          // n x n, NOT modified by Golub-Kahan path
        public doubleN b;
        public doubleN x;
        public doubleSVDCache ws;

        public void Execute() => SVD.pinvSolve(ref A, in b, ref x, ref ws);
    }

    // ---- pseudoInverse ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PseudoInverseJobDouble : IJob
    {
        public doubleMxN A;          // n x n, NOT modified by Golub-Kahan path
        public doubleMxN Aplus;      // n x n
        public doubleSVDCache ws;

        public void Execute() => SVD.pseudoInverse(ref A, ref Aplus, ref ws);
    }

    public static partial class SvdSolversBenchmark
    {
        // ---- randomized ----
        static string SvdRandDouble(int n)
        {
            const int k = 16;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Uk = arena.doubleMat(n, k);
            var Sk = arena.doubleVec(k);
            var Vk = arena.doubleMat(n, k);
            var ws = arena.doubleSVDRandomizedCache(n, n, k);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1f, 1f);

            var job = new SvdRandomizedJobDouble { A = A, Uk = Uk, Sk = Sk, Vk = Vk, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- pinvSolve ----
        static string PinvDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var x = arena.doubleVec(n);
            var ws = arena.doubleSVDCache(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1f, 1f);
            for (int d = 0; d < n; d++)
                A[d, d] += n;
            for (int i = 0; i < n; i++)
                b[i] = rng.NextDouble(-1f, 1f);

            var job = new PinvSolveJobDouble { A = A, b = b, x = x, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- pseudoInverse ----
        static string PseudoInvDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Aplus = arena.doubleMat(n, n);
            var ws = arena.doubleSVDCache(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1f, 1f);
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new PseudoInverseJobDouble { A = A, Aplus = Aplus, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
