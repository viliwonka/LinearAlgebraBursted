using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of QRVariantsBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Flops/TallFlops, size lists, header formatters, Run, Section) is
    // hand-written in Assets/LinearAlgebra/Benchmarks/QRVariantsBenchmark.cs; the shared row
    // formatters live in the public QRVariantsFmt helper there.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPJobFProxy : IJob
    {
        public fProxyMxN Q;
        public fProxyMxN R;
        public fProxyMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            var P = new Pivot(Q.N_Cols, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRSolveJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Src;
        public fProxyN b;
        public fProxyN bSrc;
        public fProxyN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    // QRCP.solveInPlace: QRCP-based rank-safe LS solve using the zero-alloc, no-copy primitive — the
    // fused destructive fast path (applies Qᵀ to b during factorization, never reconstructs Q).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPSolveJobFProxy : IJob
    {
        public fProxyMxN A;     // n x n scratch; receives Src, destroyed by solveInPlace
        public fProxyMxN Src;   // pristine source, re-copied into A each Execute
        public fProxyN b;       // m, destroyed (becomes Qᵀb); reset from bSrc each Execute
        public fProxyN bSrc;    // pristine RHS
        public fProxyN x;       // n, solution output
        public fProxyMxN R;     // n x n scratch
        public fProxyN u;       // m scratch

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            var P = new Pivot(A.N_Cols, Allocator.Temp);
            QRCP.solveInPlace(ref A, ref b, ref x, ref R, ref P, ref u);
            P.Dispose();
        }
    }

    // QRCP.minNormSolveInPlace: the complete-orthogonal-decomposition (COD / xGELSY) min-norm solve.
    // Structurally identical to QRCPSolveJob; on a rank-deficient input it runs a SECOND orthogonal
    // sweep basic solveInPlace skips (at full rank the two coincide).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPMinNormJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Src;
        public fProxyN b;
        public fProxyN bSrc;
        public fProxyN x;
        public fProxyMxN R;
        public fProxyN u;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            var P = new Pivot(A.N_Cols, Allocator.Temp);
            QRCP.minNormSolveInPlace(ref A, ref b, ref x, ref R, ref P, ref u);
            P.Dispose();
        }
    }

    public static partial class QRVariantsBenchmark
    {
        static string QRCPFProxy(int n, double flops)
        {
            var Q = new fProxyMxN(n, n, Allocator.Persistent);
            var R = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPJobFProxy { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            Q.Dispose(); R.Dispose(); Src.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string SolveFProxy(int n, double flops)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var b = new fProxyN(n, Allocator.Persistent);
            var bSrc = new fProxyN(n, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); x.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        static string QRCPSolveFProxy(int n, double flops)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var b = new fProxyN(n, Allocator.Persistent);
            var bSrc = new fProxyN(n, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);
            var R = new fProxyMxN(n, n, Allocator.Persistent);
            var u = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPSolveJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); x.Dispose(); R.Dispose(); u.Dispose();
            return Bench.Row("fProxy", n, stat, flops);
        }

        // Rank-deficient n x n input of exact rank r: fill the first r columns at random, then set each
        // trailing column j>=r to a copy of column j-r. Runs basic solveInPlace then COD adjacently.
        static string QRCPRankDefFProxy(int n, double flops)
        {
            int rank = (3 * n) / 4;
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var b = new fProxyN(n, Allocator.Persistent);
            var bSrc = new fProxyN(n, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);
            var R = new fProxyMxN(n, n, Allocator.Persistent);
            var u = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < rank; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int r = 0; r < n; r++)
                for (int c = rank; c < n; c++)
                    Src[r, c] = Src[r, c - rank];

            var basic = new QRCPSolveJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var sB = Bench.Time(() => basic.Run());
            var cod = new QRCPMinNormJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var sC = Bench.Time(() => cod.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); x.Dispose(); R.Dispose(); u.Dispose();
            return QRVariantsFmt.RowKernel("fProxy", "basic solveInPlace", n, sB, flops)
                 + "\n" + QRVariantsFmt.RowKernel("fProxy", "COD minNormSolveInPlace", n, sC, flops);
        }

        static string SolveTallFProxy(int m, int n, double flops)
        {
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var bSrc = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 31 + n));
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); x.Dispose();
            return QRVariantsFmt.RowTall("fProxy", "QR.solveInPlace", m, n, stat, flops);
        }

        static string QRCPSolveTallFProxy(int m, int n, double flops)
        {
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var bSrc = new fProxyN(m, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);
            var R = new fProxyMxN(n, n, Allocator.Persistent);
            var u = new fProxyN(m, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 31 + n));
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFProxy(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPSolveJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); x.Dispose(); R.Dispose(); u.Dispose();
            return QRVariantsFmt.RowTall("fProxy", "QRCP.solveInPlace", m, n, stat, flops);
        }
    }
}
