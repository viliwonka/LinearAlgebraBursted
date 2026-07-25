using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of CholeskyBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Flops, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/CholeskyBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN L;

        public void Execute() => CHO.decomp(in A, ref L);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholPivotJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN L;
        public fProxyCHOPCache ws;

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
    // DirectSolveBenchmark's LuSolveTransAJobFProxy) -- decompInPlace overwrites its argument, so
    // without the re-copy every run after the first would be re-factoring an already-triangular
    // matrix instead of the intended SPD input.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholInPlaceJobFProxy : IJob
    {
        public fProxyMxN A;      // receives Src each Execute; destroyed by decompInPlace
        public fProxyMxN Src;

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
    public struct CholPivotInPlaceJobFProxy : IJob
    {
        public fProxyMxN A;      // receives Src each Execute; destroyed by decomp (L aliases A)
        public fProxyMxN Src;
        public fProxyCHOPCache ws;

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
    public struct LUFaceOffInPlaceJobFProxy : IJob
    {
        public fProxyMxN A;      // receives Src each Execute; destroyed by decompInPlace
        public fProxyMxN Src;

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
        // Face-off SPD build (symmetric random fill + diagonal dominance, same recipe BenchFProxy/
        // PivotFProxy below already use) is inlined into each of the three methods below rather than
        // shared via a private helper: a helper returning fProxyMxN would collide across the
        // generated float.cs/double.cs halves of this partial class (CS0111 -- same name and
        // parameter types, C# does not overload on return type alone).

        static string FaceOffCholFProxy(int n, double flops)
        {
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var A = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new CholInPlaceJobFProxy { A = A, Src = Src };
            var stat = Bench.Time(() => job.Run());

            Src.Dispose(); A.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string FaceOffCholPivotFProxy(int n, double flops)
        {
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var ws = new fProxyCHOPCache(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new CholPivotInPlaceJobFProxy { A = A, Src = Src, ws = ws };
            var stat = Bench.Time(() => job.Run());

            Src.Dispose(); A.Dispose(); ws.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string FaceOffLUFProxy(int n, double flops)
        {
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var A = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new LUFaceOffInPlaceJobFProxy { A = A, Src = Src };
            var stat = Bench.Time(() => job.Run());

            Src.Dispose(); A.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string BenchFProxy(int n, double flops)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var L = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;                // symmetric
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;                   // diagonal dominance => SPD

            var job = new CholJobFProxy { A = A, L = L };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); L.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string PivotFProxy(int n, double flops)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var L = new fProxyMxN(n, n, Allocator.Persistent);
            var ws = new fProxyCHOPCache(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;                // symmetric
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;                   // diagonal dominance => full-rank SPD

            var job = new CholPivotJobFProxy { A = A, L = L, ws = ws };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); L.Dispose(); ws.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }
    }
}
