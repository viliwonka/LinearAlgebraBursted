using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of TallWideSolveBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (QrFlops, header formatter, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/TallWideSolveBenchmark.cs; the shared RowKernel formatter lives
    // in the public TallWideFmt helper there.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TallQRJobFProxy : IJob
    {
        public fProxyMxN Q;     // m x n; receives A, overwritten with the orthonormal factor
        public fProxyMxN R;     // n x n
        public fProxyMxN Src;

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
    public struct TallLSJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n, destroyed (becomes R)
        public fProxyMxN Src;
        public fProxyN b;       // length m, destroyed (becomes Qᵀb)
        public fProxyN bSrc;
        public fProxyN x;       // length n, solution

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
    public struct WideLQJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n (n >= m); not modified by LQ.decomp
        public fProxyMxN L;     // m x m
        public fProxyMxN Q;     // m x n

        public void Execute() => LQ.decomp(in A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideMinNormJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n (n >= m); not modified
        public fProxyN b;       // length m; not modified (copied internally)
        public fProxyN x;       // length n, min-norm solution

        public void Execute() => LQ.minNormSolve(in A, in b, ref x);
    }

    // LQ.minNormSolveInPlace: full-row-rank min-norm solve that factors A in place (no working copy).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideMinNormInPlaceJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n; DESTROYED by minNormSolveInPlace (restored from Src each sample)
        public fProxyMxN Src;
        public fProxyN b;       // length m; not modified
        public fProxyN x;       // length n, min-norm solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQ.minNormSolveInPlace(ref A, in b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n (n >= m); not modified by LQRP.decomp
        public fProxyMxN L;     // m x m
        public fProxyMxN Q;     // m x n
        public Pivot P;         // size m

        public void Execute() => LQRP.decomp(in A, ref L, ref Q, ref P);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPSolveJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n; DESTROYED by solveInPlace (restored from Src each sample)
        public fProxyMxN Src;
        public fProxyN b;       // length m; NOT modified by LQRP.solveInPlace (read-only)
        public fProxyN x;       // length n, basic solution

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
    public struct WideLQRPMinNormJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n; DESTROYED by minNormSolveInPlace (restored from Src each sample)
        public fProxyMxN Src;
        public fProxyN b;       // length m; NOT modified (b is preserved)
        public fProxyN x;       // length n, min-norm solution

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
        static string TallQRFProxy(int k, double flops)
        {
            int m = 2 * k, n = k;
            var Q = new fProxyMxN(m, n, Allocator.Persistent);
            var R = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;             // full column rank, no zero-column early-out

            var job = new TallQRJobFProxy { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            Q.Dispose(); R.Dispose(); Src.Dispose();
            return Bench.Row("fProxy", k, stat, flops);
        }

        // ---- Overdetermined least squares (2k x k) ----
        static string TallLSFProxy(int k, double flops)
        {
            int m = 2 * k, n = k;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var bSrc = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;

            var job = new TallLSJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); x.Dispose();
            return Bench.Row("fProxy", k, stat, flops);
        }

        // ---- Wide LQ factorization (underdetermined: k x 2k) ----
        static string WideLQFProxy(int k, double flops)
        {
            int m = k, n = 2 * k;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var L = new fProxyMxN(m, m, Allocator.Persistent);
            var Q = new fProxyMxN(m, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new WideLQJobFProxy { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); L.Dispose(); Q.Dispose();
            return Bench.Row("fProxy", k, stat, flops);
        }

        // ---- Underdetermined minimum-norm (k x 2k) ----
        static string WideMinNormFProxy(int k, double flops)
        {
            int m = k, n = 2 * k;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new WideMinNormJobFProxy { A = A, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); b.Dispose(); x.Dispose();
            return Bench.Row("fProxy", k, stat, flops);
        }

        // ---- Min-norm InPlace at an explicit m x n aspect (README wide showcase): LQ (full-rank) vs
        //      LQRP (rank-revealing COD); both DESTROY A and return the minimum-2-norm solution ----
        static string WideMinNormMNFProxy(int m, int n)
        {
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                Src[d, d] += m + n;

            var job = new WideMinNormInPlaceJobFProxy { A = A, Src = Src, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); x.Dispose();
            return TallWideFmt.RowKernel("fProxy", "LQ.minNormSolveInPlace", m, stat, 0);
        }

        static string WideLQRPMinNormMNFProxy(int m, int n)
        {
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                Src[d, d] += m + n;

            var job = new WideLQRPMinNormJobFProxy { A = A, Src = Src, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); x.Dispose();
            return TallWideFmt.RowKernel("fProxy", "LQRP.minNormSolveInPlace", m, stat, 0);
        }

        // ---- Wide LQRP row-pivoted factorization (k x 2k) ----
        static string WideLQRPDecompFProxy(int k, double flops)
        {
            int m = k, n = 2 * k;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var L = new fProxyMxN(m, m, Allocator.Persistent);
            var Q = new fProxyMxN(m, n, Allocator.Persistent);
            var P = new Pivot(m, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new WideLQRPJobFProxy { A = A, L = L, Q = Q, P = P };
            var stat = Bench.Time(() => job.Run());

            P.Dispose();
            A.Dispose(); L.Dispose(); Q.Dispose();
            return Bench.Row("fProxy", k, stat, flops);
        }

        // ---- Underdetermined rank-safe basic solve (k x 2k) ----
        static string WideLQRPSolveFProxy(int k, double flops)
        {
            int m = k, n = 2 * k;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                Src[d, d] += m + n;

            var job = new WideLQRPSolveJobFProxy { A = A, Src = Src, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); x.Dispose();
            return Bench.Row("fProxy", k, stat, flops);
        }

        // Rank-deficient k x 2k input of exact rank r: fill the first r rows at random, then set each
        // trailing row i>=r to a copy of row i-r. Runs basic solveInPlace then COD adjacently.
        static string WideLQRPRankDefFProxy(int k, double flops)
        {
            int m = k, n = 2 * k, rank = (3 * k) / 4;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < rank; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int r = rank; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = Src[r - rank, c];
            for (int r = 0; r < m; r++)
                b[r] = rng.NextFProxy(-1f, 1f);

            var basic = new WideLQRPSolveJobFProxy { A = A, Src = Src, b = b, x = x };
            var sB = Bench.Time(() => basic.Run());
            var cod = new WideLQRPMinNormJobFProxy { A = A, Src = Src, b = b, x = x };
            var sC = Bench.Time(() => cod.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); x.Dispose();
            return TallWideFmt.RowKernel("fProxy", "basic solveInPlace", k, sB, flops)
                 + "\n" + TallWideFmt.RowKernel("fProxy", "COD minNormSolveInPlace", k, sC, flops);
        }
    }
}
