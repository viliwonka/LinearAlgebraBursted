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

    // ---- face-off: CHO vs CHOP vs LU, all decompInPlace (destructive), SPD input ----
    // Each Execute() re-copies a pristine Src into the working buffer before the timed destructive
    // call (Src -> A copy included in the timed sample, same convention as
    // DirectSolveBenchmark's LuSolveTransAJobFloat) -- decompInPlace overwrites its argument, so
    // without the re-copy every run after the first would be re-factoring an already-triangular
    // matrix instead of the intended SPD input.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholInPlaceJobFloat : IJob
    {
        public floatMxN A;      // receives Src each Execute; destroyed by decompInPlace
        public floatMxN Src;

        public void Execute()
        {
            int n = Src.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            CHO.decompInPlace(ref A);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholPivotInPlaceJobFloat : IJob
    {
        public floatMxN A;      // receives Src each Execute; destroyed by decomp (L aliases A)
        public floatMxN Src;
        public floatCHOPCache ws;

        public void Execute()
        {
            int n = Src.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];

            var P = new Pivot(n, Allocator.Temp);
            // in-place: L aliases A's own storage, same pattern CHOP.solveInPlace uses internally.
            CHOP.decomp(in A, ref A, ref P, ref ws);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LUFaceOffInPlaceJobFloat : IJob
    {
        public floatMxN A;      // receives Src each Execute; destroyed by decompInPlace
        public floatMxN Src;

        public void Execute()
        {
            int n = Src.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];

            var P = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref A, ref P);
            P.Dispose();
        }
    }

    public static partial class CholeskyBenchmark
    {
        // Face-off SPD build (symmetric random fill + diagonal dominance, same recipe BenchFloat/
        // PivotFloat below already use) is inlined into each of the three methods below rather than
        // shared via a private helper: a helper returning floatMxN would collide across the
        // generated float.cs/double.cs halves of this partial class (CS0111 -- same name and
        // parameter types, C# does not overload on return type alone).

        static string FaceOffCholFloat(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(n, n);
            var A = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new CholInPlaceJobFloat { A = A, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, flops);
        }

        static string FaceOffCholPivotFloat(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(n, n);
            var A = arena.floatMat(n, n);
            var ws = arena.floatCHOPCache(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new CholPivotInPlaceJobFloat { A = A, Src = Src, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, flops);
        }

        static string FaceOffLUFloat(int n, double flops)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(n, n);
            var A = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new LUFaceOffInPlaceJobFloat { A = A, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, flops);
        }

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
