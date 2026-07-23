using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of BlockCGSparseBenchmark: block-CG over a BSR 2D-Poisson operator vs the
    // scalar loop of s independent cg solves, plus a matvec-only probe (block spMM vs s x scalar spMV)
    // that isolates the s x n multivector layout cost from the solver bookkeeping. SPARSE is the real
    // block-CG use case (dense should use the direct multi-RHS solver). Dtype-agnostic harness (Run,
    // Section) is hand-written in Assets/LinearAlgebra/Benchmarks/BlockCGSparseBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BlockCgSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyMxN B, X, R, P, Q;
        public int K; public fProxy Tol;
        public Indices Out;   // [0] = block iters, [1] = minActive

        public void Execute()
        {
            int s = B.M_Rows, n = B.N_Cols;
            for (int i = 0; i < s; i++) for (int c = 0; c < n; c++) X[i, c] = (fProxy)0;
            var info = Krylov.bcg(new fProxyBSROperator(in A), in B, ref X, ref R, ref P, ref Q, K, Tol);
            Out[0] = info.iterations;
            Out[1] = info.minActive;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BlockCgrQSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyMxN B, X, R, P, AP, Pa;
        public int K; public fProxy Tol;
        public Indices Out;   // [0] = block iters, [1] = minActive

        public void Execute()
        {
            int s = B.M_Rows, n = B.N_Cols;
            for (int i = 0; i < s; i++) for (int c = 0; c < n; c++) X[i, c] = (fProxy)0;
            var info = Krylov.bcgrq(new fProxyBSROperator(in A), in B, ref X, ref R, ref P, ref AP, ref Pa, K, Tol);
            Out[0] = info.iterations;
            Out[1] = info.minActive;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BlockBfbcgSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyMxN B, X, R, P, AP, Pa;
        public int K; public fProxy Tol;
        public Indices Out;   // [0] = block iters, [1] = minActive

        public void Execute()
        {
            int s = B.M_Rows, n = B.N_Cols;
            for (int i = 0; i < s; i++) for (int c = 0; c < n; c++) X[i, c] = (fProxy)0;
            var info = Krylov.bfbcg(new fProxyBSROperator(in A), in B, ref X, ref R, ref P, ref AP, ref Pa, K, Tol);
            Out[0] = info.iterations;
            Out[1] = info.minActive;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ScalarLoopSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyMxN B;
        public fProxyN x, r, p, Ap, bcol;
        public int S, K; public fProxy Tol;
        public Indices Out;   // [0] = total iters over the s columns

        public void Execute()
        {
            int n = B.N_Cols;
            var op = new fProxyBSROperator(in A);
            int total = 0;
            for (int j = 0; j < S; j++)
            {
                for (int c = 0; c < n; c++) { bcol[c] = B[j, c]; x[c] = (fProxy)0; }
                var info = Krylov.cg(in op, in bcol, ref x, ref r, ref p, ref Ap, K, Tol);
                total += info.iterations;
            }
            Out[0] = total;
        }
    }

    // Matvec-only layout probe: Reps block spMM(s) calls vs Reps*s single-vector spMV calls -- same total
    // matvec work, so the wall-clock ratio is the pure s x n multivector SIMD/layout efficiency.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpMMProbeJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyMxN V, AV;
        public int S, Reps;
        public void Execute() { for (int r = 0; r < Reps; r++) BSR.spMM(in A, in V, ref AV, S); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpMVLoopProbeJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyMxN V;
        public fProxyN x, y;
        public int S, Reps;
        public void Execute()
        {
            int n = V.N_Cols;
            var op = new fProxyBSROperator(in A);
            for (int r = 0; r < Reps; r++)
                for (int j = 0; j < S; j++)
                {
                    for (int c = 0; c < n; c++) x[c] = V[j, c];
                    op.Apply(in x, ref y);
                }
        }
    }

    public static partial class BlockCGSparseBenchmark
    {
        static string BenchFProxy(int grid, int s)
        {
            const string fmt = "{0,-7}{1,-7}{2,-4}{3,-14}{4,10:F4}{5,12:F4}{6,8}{7,8}";

            var A = fProxyGallery.fProxyLaplacian2D(grid, grid, Allocator.Persistent);       // n = grid*grid, ~5 nonzeros/row
            int n = A.M_Rows;
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(grid * 131 + s));

            var B = new fProxyMxN(s, n, Allocator.Persistent);                     // independent random RHS
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) B[i, c] = rng.NextFProxy(-1f, 1f);

            fProxy tol = Consts.fProxySqrtEps;
            int cap = 4 * n;
            var outv = new Indices(2, Allocator.Persistent);
            var sb = new StringBuilder();

            // block-CG
            var X = new fProxyMxN(s, n, Allocator.Persistent); var R = new fProxyMxN(s, n, Allocator.Persistent);
            var P = new fProxyMxN(s, n, Allocator.Persistent); var Q = new fProxyMxN(s, n, Allocator.Persistent);
            var blockJob = new BlockCgSparseJobFProxy { A = A, B = B, X = X, R = R, P = P, Q = Q, K = cap, Tol = tol, Out = outv };
            var blockStat = Bench.Time(() => blockJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "block-CG", blockStat.Median, blockStat.Min, outv[0], outv[1]));

            // bcgrq
            var Xrq = new fProxyMxN(s, n, Allocator.Persistent); var Rrq = new fProxyMxN(s, n, Allocator.Persistent);
            var Prq = new fProxyMxN(s, n, Allocator.Persistent); var APrq = new fProxyMxN(s, n, Allocator.Persistent); var Parq = new fProxyMxN(s, n, Allocator.Persistent);
            var rqJob = new BlockCgrQSparseJobFProxy { A = A, B = B, X = Xrq, R = Rrq, P = Prq, AP = APrq, Pa = Parq, K = cap, Tol = tol, Out = outv };
            var rqStat = Bench.Time(() => rqJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "bcgrq", rqStat.Median, rqStat.Min, outv[0], outv[1]));

            // bfbcg
            var Xbf = new fProxyMxN(s, n, Allocator.Persistent); var Rbf = new fProxyMxN(s, n, Allocator.Persistent);
            var Pbf = new fProxyMxN(s, n, Allocator.Persistent); var APbf = new fProxyMxN(s, n, Allocator.Persistent); var Pabf = new fProxyMxN(s, n, Allocator.Persistent);
            var bfJob = new BlockBfbcgSparseJobFProxy { A = A, B = B, X = Xbf, R = Rbf, P = Pbf, AP = APbf, Pa = Pabf, K = cap, Tol = tol, Out = outv };
            var bfStat = Bench.Time(() => bfJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "bfbcg", bfStat.Median, bfStat.Min, outv[0], outv[1]));

            // scalar loop
            var x = new fProxyN(n, Allocator.Persistent); var r = new fProxyN(n, Allocator.Persistent); var p = new fProxyN(n, Allocator.Persistent);
            var Ap = new fProxyN(n, Allocator.Persistent); var bcol = new fProxyN(n, Allocator.Persistent);
            var loopJob = new ScalarLoopSparseJobFProxy { A = A, B = B, x = x, r = r, p = p, Ap = Ap, bcol = bcol, S = s, K = cap, Tol = tol, Out = outv };
            var loopStat = Bench.Time(() => loopJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "scalar x s", loopStat.Median, loopStat.Min, outv[0], 0));

            // matvec-only layout probe (Reps = 50 fixed)
            int reps = 50;
            var AV = new fProxyMxN(s, n, Allocator.Persistent);
            var mmJob = new SpMMProbeJobFProxy { A = A, V = B, AV = AV, S = s, Reps = reps };
            var mmStat = Bench.Time(() => mmJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "spMM x50", mmStat.Median, mmStat.Min, reps, 0));

            var yv = new fProxyN(n, Allocator.Persistent);
            var mvJob = new SpMVLoopProbeJobFProxy { A = A, V = B, x = x, y = yv, S = s, Reps = reps };
            var mvStat = Bench.Time(() => mvJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "spMVx s x50", mvStat.Median, mvStat.Min, reps * s, 0));

            A.Dispose(); B.Dispose(); outv.Dispose();
            X.Dispose(); R.Dispose(); P.Dispose(); Q.Dispose();
            Xrq.Dispose(); Rrq.Dispose(); Prq.Dispose(); APrq.Dispose(); Parq.Dispose();
            Xbf.Dispose(); Rbf.Dispose(); Pbf.Dispose(); APbf.Dispose(); Pabf.Dispose();
            x.Dispose(); r.Dispose(); p.Dispose(); Ap.Dispose(); bcol.Dispose();
            AV.Dispose(); yv.Dispose();
            return sb.ToString();
        }
    }
}
