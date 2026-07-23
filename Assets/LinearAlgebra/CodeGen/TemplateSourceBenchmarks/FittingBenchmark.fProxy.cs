using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of FittingBenchmark (timed IJobs + build+measure method). The
    // dtype-agnostic harness (Run, Section, formatter) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/FittingBenchmark.cs.
    //
    // Linear regression fit of m observations onto n coefficients, on ONE design matrix + an
    // outlier-contaminated response: L2 (QR least squares), exact L1 (LP.lad = Frisch-Newton at these
    // m), approximate L1 (Optimize.ladIRLS). Response residuals, not orthogonal distance (NOT TLS).

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FitQRJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n; DESTROYED by solveInPlace (restored from Src each sample)
        public fProxyMxN Src;
        public fProxyN b;       // length m; DESTROYED (restored from bSrc each sample)
        public fProxyN bSrc;
        public fProxyN x;       // length n, coefficients

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
    public struct FitLadJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n; NOT modified
        public fProxyN b;       // length m; NOT modified
        public fProxyN x;       // length n, coefficients

        public void Execute() => LP.lad(in A, in b, ref x, out double _);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FitIRLSJobFProxy : IJob
    {
        public fProxyMxN A;     // m x n; NOT modified
        public fProxyN b;       // length m; NOT modified
        public fProxyN x;       // length n, coefficients

        public void Execute() => Optimize.ladIRLS(in A, in b, ref x);
    }

    public static partial class FittingBenchmark
    {
        static string FitFProxy(int m, int n)
        {
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            var Src = new fProxyMxN(m, n, Allocator.Persistent);
            var b = new fProxyN(m, Allocator.Persistent);
            var bSrc = new fProxyN(m, Allocator.Persistent);
            var xTrue = new fProxyN(n, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)m ^ ((uint)n << 16));
            for (int j = 0; j < n; j++) xTrue[j] = rng.NextFProxy(-1f, 1f);
            for (int r = 0; r < m; r++)
            {
                fProxy dot = (fProxy)0;
                for (int c = 0; c < n; c++)
                {
                    fProxy a = rng.NextFProxy(-1f, 1f);
                    Src[r, c] = a;
                    dot += a * xTrue[c];
                }
                fProxy noise = rng.NextFProxy(-0.05f, 0.05f);
                fProxy outlier = rng.NextFProxy(0f, 1f) < (fProxy)0.05 ? rng.NextFProxy(-20f, 20f) : (fProxy)0;
                bSrc[r] = dot + noise + outlier;
            }

            var qr = new FitQRJobFProxy { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var sQR = Bench.Time(() => qr.Run());

            // LAD / IRLS read A and b directly (not destroyed) -> load Src/bSrc once.
            for (int r = 0; r < m; r++)
            {
                b[r] = bSrc[r];
                for (int c = 0; c < n; c++) A[r, c] = Src[r, c];
            }
            var lad = new FitLadJobFProxy { A = A, b = b, x = x };
            var sLad = Bench.Time(() => lad.Run());
            var irls = new FitIRLSJobFProxy { A = A, b = b, x = x };
            var sIRLS = Bench.Time(() => irls.Run());

            A.Dispose(); Src.Dispose(); b.Dispose(); bSrc.Dispose(); xTrue.Dispose(); x.Dispose();
            var sb = new StringBuilder();
            sb.AppendLine(FittingFmt.Fmt("fProxy", "QR.solveInPlace L2", m, n, sQR));
            sb.AppendLine(FittingFmt.Fmt("fProxy", "LP.lad exact L1", m, n, sLad));
            sb.Append(FittingFmt.Fmt("fProxy", "Optimize.ladIRLS L1", m, n, sIRLS));
            return sb.ToString();
        }
    }
}
