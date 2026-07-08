using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of DirectSolveBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (N, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/DirectSolveBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuSolveJobFProxy : IJob
    {
        public fProxyMxN U;     // receives Src via LU.decomp (copies internally)
        public fProxyMxN L;
        public fProxyMxN Src;
        public fProxyN b;       // receives bSrc, overwritten with the solution
        public fProxyN bSrc;

        public void Execute()
        {
            int n = Src.M_Rows;
            for (int i = 0; i < n; i++) b[i] = bSrc[i];

            var P = new Pivot(n, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            LU.decompSolve(ref L, ref U, in P, ref b);
            P.Dispose();
        }
    }

    // CHO.decomp + CHO.decompSolve factor-then-solve, as the explicit two-call composition.
    // A and L are distinct buffers so A is NOT destroyed (only b, re-copied from bSrc each Execute).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CholSolveJobFProxy : IJob
    {
        public fProxyMxN A;     // SPD input, not modified (distinct from L)
        public fProxyMxN L;
        public fProxyN b;
        public fProxyN bSrc;

        public void Execute()
        {
            for (int i = 0; i < bSrc.N; i++) b[i] = bSrc[i];
            var info = CHO.decomp(in A, ref L);
            if (info.Solved) CHO.decompSolve(ref L, ref b);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrSquareSolveJobFProxy : IJob
    {
        public fProxyMxN A;     // receives Src, destroyed by solveInPlace
        public fProxyMxN Src;
        public fProxyN b;
        public fProxyN bSrc;
        public fProxyN x;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < n; i++) b[i] = bSrc[i];

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    // QR.solveInPlace via the caller-owned fProxyQRCache: identical fused, never-forms-Q kernel as
    // QrSquareSolveJobFProxy above (bit-identical results), but the reflector-apply accumulators are
    // cache-owned instead of a fresh Allocator.Temp alloc per Execute.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrSquareSolveCacheJobFProxy : IJob
    {
        public fProxyMxN A;     // receives Src, destroyed by solveInPlace
        public fProxyMxN Src;
        public fProxyN b;
        public fProxyN bSrc;
        public fProxyN x;
        public fProxyQRCache Cache;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < n; i++) b[i] = bSrc[i];

            QR.solveInPlace(ref A, ref b, ref x, ref Cache);
        }
    }

    public static partial class DirectSolveBenchmark
    {
        static string LuSolveFProxy(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.fProxyMat(N, N);
            var L = arena.fProxyMat(N, N);
            var Src = arena.fProxyMat(N, N);
            var b = arena.fProxyVec(N);
            var bSrc = arena.fProxyVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < N; d++)
                Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFProxy(-1f, 1f);

            var job = new LuSolveJobFProxy { U = U, L = L, Src = Src, b = b, bSrc = bSrc };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("LU fProxy", N, stat);
        }

        static string CholSolveFProxy(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(N, N);
            var L = arena.fProxyMat(N, N);
            var b = arena.fProxyVec(N);
            var bSrc = arena.fProxyVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int i = 0; i < N; i++)
                for (int j = i; j < N; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;
                }
            for (int d = 0; d < N; d++) A[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFProxy(-1f, 1f);

            var job = new CholSolveJobFProxy { A = A, L = L, b = b, bSrc = bSrc };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("Cholesky fProxy", N, stat);
        }

        static string QrSolveFProxy(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(N, N);
            var Src = arena.fProxyMat(N, N);
            var b = arena.fProxyVec(N);
            var bSrc = arena.fProxyVec(N);
            var x = arena.fProxyVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < N; d++) Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFProxy(-1f, 1f);

            var job = new QrSquareSolveJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("QR fProxy", N, stat);
        }

        static string QrSolveCacheFProxy(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(N, N);
            var Src = arena.fProxyMat(N, N);
            var b = arena.fProxyVec(N);
            var bSrc = arena.fProxyVec(N);
            var x = arena.fProxyVec(N);
            var cache = arena.fProxyQRCache(N, N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < N; d++) Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFProxy(-1f, 1f);

            var job = new QrSquareSolveCacheJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x, Cache = cache };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("QR fProxy (cache)", N, stat);
        }
    }
}
