using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of TallWideSolveBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (QrFlops, header formatter, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/TallWideSolveBenchmark.cs; the shared RowKernel formatter lives
    // in the public TallWideFmt helper there.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TallQRJobFloat : IJob
    {
        public floatMxN Q;     // m x n; receives A, overwritten with the orthonormal factor
        public floatMxN R;     // n x n
        public floatMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            QR.decompInPlace(ref Q, ref R);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TallLSJobFloat : IJob
    {
        public floatMxN A;     // m x n, destroyed (becomes R)
        public floatMxN Src;
        public floatN b;       // length m, destroyed (becomes Qᵀb)
        public floatN bSrc;
        public floatN x;       // length n, solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
            {
                b[r] = bSrc[r];
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            }

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQJobFloat : IJob
    {
        public floatMxN A;     // m x n (n >= m); not modified by LQ.decomp
        public floatMxN L;     // m x m
        public floatMxN Q;     // m x n

        public void Execute() => LQ.decomp(in A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideMinNormJobFloat : IJob
    {
        public floatMxN A;     // m x n (n >= m); not modified
        public floatN b;       // length m; not modified (copied internally)
        public floatN x;       // length n, min-norm solution

        public void Execute() => LQ.minNormSolve(ref A, ref b, ref x);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPJobFloat : IJob
    {
        public floatMxN A;     // m x n (n >= m); not modified by LQRP.decomp
        public floatMxN L;     // m x m
        public floatMxN Q;     // m x n
        public Pivot P;         // size m

        public void Execute() => LQRP.decomp(in A, ref L, ref Q, ref P);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPSolveJobFloat : IJob
    {
        public floatMxN A;     // m x n; DESTROYED by solveInPlace (restored from Src each sample)
        public floatMxN Src;
        public floatN b;       // length m; NOT modified by LQRP.solveInPlace (read-only)
        public floatN x;       // length n, basic solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQRP.solveInPlace(ref A, ref b, ref x);
        }
    }

    // LQRP.minNormSolveInPlace: COD min-norm solve; on a rank-deficient wide system it runs a second
    // orthogonal stage the basic solve skips (at full row rank the two coincide).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPMinNormJobFloat : IJob
    {
        public floatMxN A;     // m x n; DESTROYED by minNormSolveInPlace (restored from Src each sample)
        public floatMxN Src;
        public floatN b;       // length m; NOT modified (b is preserved)
        public floatN x;       // length n, min-norm solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQRP.minNormSolveInPlace(ref A, ref b, ref x);
        }
    }

    public static partial class TallWideSolveBenchmark
    {
        // ---- Tall QR factorization (overdetermined: 2k x k) ----
        static string TallQRFloat(int k, double flops)
        {
            int m = 2 * k, n = k;
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.floatMat(m, n);
            var R = arena.floatMat(n, n);
            var Src = arena.floatMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;             // full column rank, no zero-column early-out

            var job = new TallQRJobFloat { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, flops);
        }

        // ---- Overdetermined least squares (2k x k) ----
        static string TallLSFloat(int k, double flops)
        {
            int m = 2 * k, n = k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var bSrc = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;

            var job = new TallLSJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, flops);
        }

        // ---- Wide LQ factorization (underdetermined: k x 2k) ----
        static string WideLQFloat(int k, double flops)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var L = arena.floatMat(m, m);
            var Q = arena.floatMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new WideLQJobFloat { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, flops);
        }

        // ---- Underdetermined minimum-norm (k x 2k) ----
        static string WideMinNormFloat(int k, double flops)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new WideMinNormJobFloat { A = A, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, flops);
        }

        // ---- Wide LQRP row-pivoted factorization (k x 2k) ----
        static string WideLQRPDecompFloat(int k, double flops)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var L = arena.floatMat(m, m);
            var Q = arena.floatMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new WideLQRPJobFloat { A = A, L = L, Q = Q, P = P };
            var stat = Bench.Time(() => job.Run());

            P.Dispose();
            arena.Dispose();
            return Bench.Row("float", k, stat, flops);
        }

        // ---- Underdetermined rank-safe basic solve (k x 2k) ----
        static string WideLQRPSolveFloat(int k, double flops)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                Src[d, d] += m + n;

            var job = new WideLQRPSolveJobFloat { A = A, Src = Src, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, flops);
        }

        // Rank-deficient k x 2k input of exact rank r: fill the first r rows at random, then set each
        // trailing row i>=r to a copy of row i-r. Runs basic solveInPlace then COD adjacently.
        static string WideLQRPRankDefFloat(int k, double flops)
        {
            int m = k, n = 2 * k, rank = (3 * k) / 4;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < rank; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int r = rank; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = Src[r - rank, c];
            for (int r = 0; r < m; r++)
                b[r] = rng.NextFloat(-1f, 1f);

            var basic = new WideLQRPSolveJobFloat { A = A, Src = Src, b = b, x = x };
            var sB = Bench.Time(() => basic.Run());
            var cod = new WideLQRPMinNormJobFloat { A = A, Src = Src, b = b, x = x };
            var sC = Bench.Time(() => cod.Run());

            arena.Dispose();
            return TallWideFmt.RowKernel("float", "basic solveInPlace", k, sB, flops)
                 + "\n" + TallWideFmt.RowKernel("float", "COD minNormSolveInPlace", k, sC, flops);
        }
    }
}
