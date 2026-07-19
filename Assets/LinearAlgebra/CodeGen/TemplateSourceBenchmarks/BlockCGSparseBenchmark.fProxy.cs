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
            var info = Krylov.cg(new fProxyBSROperator(in A), in B, ref X, ref R, ref P, ref Q, K, Tol);
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
            var arena = new Arena(Allocator.Persistent);
            const string fmt = "{0,-7}{1,-7}{2,-4}{3,-14}{4,10:F4}{5,12:F4}{6,8}{7,8}";

            var A = arena.fProxyLaplacian2D(grid, grid);       // n = grid*grid, ~5 nonzeros/row
            int n = A.M_Rows;
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(grid * 131 + s));

            var B = arena.fProxyMat(s, n);                     // independent random RHS
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) B[i, c] = rng.NextFProxy(-1f, 1f);

            fProxy tol = Consts.fProxySqrtEps;
            int cap = 4 * n;
            var outv = arena.Indices(2);
            var sb = new StringBuilder();

            // block-CG
            var X = arena.fProxyMat(s, n); var R = arena.fProxyMat(s, n);
            var P = arena.fProxyMat(s, n); var Q = arena.fProxyMat(s, n);
            var blockJob = new BlockCgSparseJobFProxy { A = A, B = B, X = X, R = R, P = P, Q = Q, K = cap, Tol = tol, Out = outv };
            var blockStat = Bench.Time(() => blockJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "block-CG", blockStat.Median, blockStat.Min, outv[0], outv[1]));

            // scalar loop
            var x = arena.fProxyVec(n); var r = arena.fProxyVec(n); var p = arena.fProxyVec(n);
            var Ap = arena.fProxyVec(n); var bcol = arena.fProxyVec(n);
            var loopJob = new ScalarLoopSparseJobFProxy { A = A, B = B, x = x, r = r, p = p, Ap = Ap, bcol = bcol, S = s, K = cap, Tol = tol, Out = outv };
            var loopStat = Bench.Time(() => loopJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "scalar x s", loopStat.Median, loopStat.Min, outv[0], 0));

            // matvec-only layout probe (Reps = 50 fixed)
            int reps = 50;
            var AV = arena.fProxyMat(s, n);
            var mmJob = new SpMMProbeJobFProxy { A = A, V = B, AV = AV, S = s, Reps = reps };
            var mmStat = Bench.Time(() => mmJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "spMM x50", mmStat.Median, mmStat.Min, reps, 0));

            var yv = arena.fProxyVec(n);
            var mvJob = new SpMVLoopProbeJobFProxy { A = A, V = B, x = x, y = yv, S = s, Reps = reps };
            var mvStat = Bench.Time(() => mvJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "spMVx s x50", mvStat.Median, mvStat.Min, reps * s, 0));

            arena.Dispose();
            return sb.ToString();
        }
    }
}
