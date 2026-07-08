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
    public struct LuSolveJobFloat : IJob
    {
        public floatMxN U;     // receives Src via LU.decomp (copies internally)
        public floatMxN L;
        public floatMxN Src;
        public floatN b;       // receives bSrc, overwritten with the solution
        public floatN bSrc;

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
    public struct CholSolveJobFloat : IJob
    {
        public floatMxN A;     // SPD input, not modified (distinct from L)
        public floatMxN L;
        public floatN b;
        public floatN bSrc;

        public void Execute()
        {
            for (int i = 0; i < bSrc.N; i++) b[i] = bSrc[i];
            var info = CHO.decomp(in A, ref L);
            if (info.Solved) CHO.decompSolve(ref L, ref b);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrSquareSolveJobFloat : IJob
    {
        public floatMxN A;     // receives Src, destroyed by solveInPlace
        public floatMxN Src;
        public floatN b;
        public floatN bSrc;
        public floatN x;

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

    // QR.solveInPlace via the caller-owned floatQRCache: identical fused, never-forms-Q kernel as
    // QrSquareSolveJobFloat above (bit-identical results), but the reflector-apply accumulators are
    // cache-owned instead of a fresh Allocator.Temp alloc per Execute.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrSquareSolveCacheJobFloat : IJob
    {
        public floatMxN A;     // receives Src, destroyed by solveInPlace
        public floatMxN Src;
        public floatN b;
        public floatN bSrc;
        public floatN x;
        public floatQRCache Cache;

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
        static string LuSolveFloat(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.floatMat(N, N);
            var L = arena.floatMat(N, N);
            var Src = arena.floatMat(N, N);
            var b = arena.floatVec(N);
            var bSrc = arena.floatVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < N; d++)
                Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFloat(-1f, 1f);

            var job = new LuSolveJobFloat { U = U, L = L, Src = Src, b = b, bSrc = bSrc };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("LU float", N, stat);
        }

        static string CholSolveFloat(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(N, N);
            var L = arena.floatMat(N, N);
            var b = arena.floatVec(N);
            var bSrc = arena.floatVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int i = 0; i < N; i++)
                for (int j = i; j < N; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;
                }
            for (int d = 0; d < N; d++) A[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFloat(-1f, 1f);

            var job = new CholSolveJobFloat { A = A, L = L, b = b, bSrc = bSrc };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("Cholesky float", N, stat);
        }

        static string QrSolveFloat(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(N, N);
            var Src = arena.floatMat(N, N);
            var b = arena.floatVec(N);
            var bSrc = arena.floatVec(N);
            var x = arena.floatVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < N; d++) Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFloat(-1f, 1f);

            var job = new QrSquareSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("QR float", N, stat);
        }

        static string QrSolveCacheFloat(int N)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(N, N);
            var Src = arena.floatMat(N, N);
            var b = arena.floatVec(N);
            var bSrc = arena.floatVec(N);
            var x = arena.floatVec(N);
            var cache = arena.floatQRCache(N, N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < N; d++) Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextFloat(-1f, 1f);

            var job = new QrSquareSolveCacheJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x, Cache = cache };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("QR float (cache)", N, stat);
        }
    }
}
