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
    public struct SvdRandomizedJobFProxy : IJob
    {
        public fProxyMxN A;          // n x n input, NOT modified
        public fProxyMxN Uk;         // n x k
        public fProxyN Sk;           // length k
        public fProxyMxN Vk;         // n x k
        public fProxySVDRandomizedCache ws;

        public void Execute() => SVD.randomized(in A, ref Uk, ref Sk, ref Vk, 16, ref ws);
    }

    // ---- pinvSolve ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PinvSolveJobFProxy : IJob
    {
        public fProxyMxN A;          // n x n, NOT modified by Golub-Kahan path
        public fProxyN b;
        public fProxyN x;
        public fProxySVDCache ws;

        public void Execute() => SVD.pinvSolve(ref A, in b, ref x, ref ws);
    }

    // ---- pseudoInverse ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PseudoInverseJobFProxy : IJob
    {
        public fProxyMxN A;          // n x n, NOT modified by Golub-Kahan path
        public fProxyMxN Aplus;      // n x n
        public fProxySVDCache ws;

        public void Execute() => SVD.pseudoInverse(ref A, ref Aplus, ref ws);
    }

    public static partial class SvdSolversBenchmark
    {
        // ---- randomized ----
        static string SvdRandFProxy(int n)
        {
            const int k = 16;
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Uk = new fProxyMxN(n, k, Allocator.Persistent);
            var Sk = new fProxyN(k, Allocator.Persistent);
            var Vk = new fProxyMxN(n, k, Allocator.Persistent);
            var ws = new fProxySVDRandomizedCache(n, n, k, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);

            var job = new SvdRandomizedJobFProxy { A = A, Uk = Uk, Sk = Sk, Vk = Vk, ws = ws };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Uk.Dispose(); Sk.Dispose(); Vk.Dispose(); ws.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }

        // ---- pinvSolve ----
        static string PinvFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var b = new fProxyN(n, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);
            var ws = new fProxySVDCache(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                A[d, d] += n;
            for (int i = 0; i < n; i++)
                b[i] = rng.NextFProxy(-1f, 1f);

            var job = new PinvSolveJobFProxy { A = A, b = b, x = x, ws = ws };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); b.Dispose(); x.Dispose(); ws.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }

        // ---- pseudoInverse ----
        static string PseudoInvFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Aplus = new fProxyMxN(n, n, Allocator.Persistent);
            var ws = new fProxySVDCache(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new PseudoInverseJobFProxy { A = A, Aplus = Aplus, ws = ws };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Aplus.Dispose(); ws.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }
    }
}
