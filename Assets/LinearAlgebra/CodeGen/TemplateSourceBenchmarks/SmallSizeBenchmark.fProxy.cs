using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of SmallSizeBenchmark (timed IJobs + build+measure methods returning
    // Bench.Stat). The dtype-agnostic harness (size lists, shape formatters, Run, Section) is
    // hand-written in Assets/LinearAlgebra/Benchmarks/SmallSizeBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallQRJobFProxy : IJob
    {
        public fProxyMxN Q;     // m x n (m >= n); receives Src, overwritten with the orthonormal factor
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
    public struct SmallLQJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n (m <= n); not modified by LQ.decomp
        public fProxyMxN L;     // m x m
        public fProxyMxN Q;     // m x n

        public void Execute() => LQ.decomp(in A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallCholJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN L;

        public void Execute() => CHO.decomp(in A, ref L);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallLUJobFProxy : IJob
    {
        public fProxyMxN U;
        public fProxyMxN L;
        public fProxyMxN Src;

        public void Execute()
        {
            int rows = Src.M_Rows;
            var P = new Pivot(rows, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            P.Dispose();
        }
    }

    public static partial class SmallSizeBenchmark
    {
        // ---- QR (square + tall share one path; sized by m x n, m >= n) ----
        static Bench.Stat QRFProxy(int m, int n)
        {
            var Q = new fProxyMxN(m, n, Allocator.Persistent);
            var R = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 1000003 + n));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;             // diagonal dominance => full rank, no zero-column early-out

            var job = new SmallQRJobFProxy { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            Q.Dispose(); R.Dispose(); Src.Dispose();
            return stat;
        }

        // ---- LQ (square + wide share one path; sized by m x n, m <= n) ----
        static Bench.Stat LQFProxy(int m, int n)
        {
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var L = new fProxyMxN(m, m, Allocator.Persistent);
            var Q = new fProxyMxN(m, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 1000003 + n));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new SmallLQJobFProxy { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); L.Dispose(); Q.Dispose();
            return stat;
        }

        // ---- Cholesky (square SPD only) ----
        static Bench.Stat CholFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var L = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;                 // symmetric
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;                    // diagonal dominance => SPD

            var job = new SmallCholJobFProxy { A = A, L = L };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); L.Dispose();
            return stat;
        }

        // ---- LU (square, partial pivoting) ----
        static Bench.Stat LUFProxy(int n)
        {
            var U = new fProxyMxN(n, n, Allocator.Persistent);
            var L = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                  // diagonal dominance => well-conditioned, full rank

            var job = new SmallLUJobFProxy { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            U.Dispose(); L.Dispose(); Src.Dispose();
            return stat;
        }
    }
}
