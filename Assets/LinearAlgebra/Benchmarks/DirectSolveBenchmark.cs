using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // The end-to-end "solve Ax=b" entry points, as opposed to decompositions.md's factorization-only
    // benchmarks: LU.decompSolve, CHO.decomp+decompSolve, QR.solveInPlace (square). Each Execute copies a
    // pristine source into the working buffers (factorization/solve destroys them), so every timed
    // sample does identical work. solvers.md notes the triangular-solve step itself is O(n^2), dominated
    // by the O(n^3) factorization in every case here — these numbers are effectively the factorization
    // cost plus a negligible solve.

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

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuSolveJobDouble : IJob
    {
        public doubleMxN U;
        public doubleMxN L;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;

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

    // CHO.decomp + CHO.decompSolve factor-then-solve, as the explicit two-call composition
    // (choleskySolve(in A, ref L, ref b) was deleted — it was this exact composition in disguise).
    // A and L are distinct buffers so A is NOT destroyed (only b, which is re-copied from bSrc each
    // Execute).
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
    public struct CholSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;
        public doubleN b;
        public doubleN bSrc;

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

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrSquareSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;
        public doubleN x;

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

    // QR.solveInPlace via the caller-owned floatQRCache/doubleQRCache (commit-3 addition): identical fused,
    // never-forms-Q kernel as QrSquareSolveJobFloat/Double above (bit-identical results), but the
    // reflector-apply accumulator w (and u) are cache-owned instead of a fresh Allocator.Temp alloc
    // per Execute — isolates the Temp-alloc-elimination win at a size where it's otherwise buried in
    // the O(n^3) factorization cost.
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

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrSquareSolveCacheJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;
        public doubleN x;
        public doubleQRCache Cache;

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

    public static class DirectSolveBenchmark
    {
        // One representative square size — the O(n^2) triangular solve is negligible next to the
        // O(n^3) factorization at this scale (see decompositions.md / solvers.md).
        const int N = 1024;

        public static void Run() => Bench.WriteReport("benchmark-directsolve.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Direct solve Ax=b, square N=" + N + " (factor + triangular solve, ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            sb.AppendLine(LuSolveFloat());
            sb.AppendLine(LuSolveDouble());
            sb.AppendLine(CholSolveFloat());
            sb.AppendLine(CholSolveDouble());
            sb.AppendLine(QrSolveFloat());
            sb.AppendLine(QrSolveDouble());
            sb.AppendLine(QrSolveCacheFloat());
            sb.AppendLine(QrSolveCacheDouble());
            sb.AppendLine();
        }

        static string LuSolveFloat()
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

        static string LuSolveDouble()
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.doubleMat(N, N);
            var L = arena.doubleMat(N, N);
            var Src = arena.doubleMat(N, N);
            var b = arena.doubleVec(N);
            var bSrc = arena.doubleVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < N; d++)
                Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextDouble(-1.0, 1.0);

            var job = new LuSolveJobDouble { U = U, L = L, Src = Src, b = b, bSrc = bSrc };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("LU double", N, stat);
        }

        static string CholSolveFloat()
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

        static string CholSolveDouble()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(N, N);
            var L = arena.doubleMat(N, N);
            var b = arena.doubleVec(N);
            var bSrc = arena.doubleVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int i = 0; i < N; i++)
                for (int j = i; j < N; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    A[i, j] = v;
                    A[j, i] = v;
                }
            for (int d = 0; d < N; d++) A[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextDouble(-1.0, 1.0);

            var job = new CholSolveJobDouble { A = A, L = L, b = b, bSrc = bSrc };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("Cholesky double", N, stat);
        }

        static string QrSolveFloat()
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

        static string QrSolveDouble()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(N, N);
            var Src = arena.doubleMat(N, N);
            var b = arena.doubleVec(N);
            var bSrc = arena.doubleVec(N);
            var x = arena.doubleVec(N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < N; d++) Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextDouble(-1.0, 1.0);

            var job = new QrSquareSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("QR double", N, stat);
        }

        // Cache-overload variant (commit-3 fProxyQRCache): same fused kernel as QrSolveFloat/Double,
        // but u/w come from a pre-built Arena.floatQRCache(N, N)/doubleQRCache(N, N) instead of a
        // fresh Allocator.Temp alloc every Execute — isolates the Temp-alloc-elimination win.
        static string QrSolveCacheFloat()
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

        static string QrSolveCacheDouble()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(N, N);
            var Src = arena.doubleMat(N, N);
            var b = arena.doubleVec(N);
            var bSrc = arena.doubleVec(N);
            var x = arena.doubleVec(N);
            var cache = arena.doubleQRCache(N, N);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < N; d++) Src[d, d] += N;
            for (int i = 0; i < N; i++) bSrc[i] = rng.NextDouble(-1.0, 1.0);

            var job = new QrSquareSolveCacheJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x, Cache = cache };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("QR double (cache)", N, stat);
        }
    }
}
