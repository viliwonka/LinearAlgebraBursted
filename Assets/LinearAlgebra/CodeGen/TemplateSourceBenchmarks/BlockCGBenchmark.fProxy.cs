using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of BlockCGBenchmark: two timed IJobs (block-CG over s RHS at once, vs
    // the scalar loop of s independent cg solves) + the build+measure method. The dtype-agnostic
    // harness (Run, Section) is hand-written in Assets/LinearAlgebra/Benchmarks/BlockCGBenchmark.cs.
    //
    // Both jobs solve the SAME s systems (SPD A, s x n block B) to the SAME tolerance, so the wall-clock
    // ratio is the true block-vs-scalar payoff: block-CG shares one Krylov subspace (fewer iterations)
    // and streams A over the whole block once per iteration via ApplyBlock (one GEMM vs s GEMVs), at the
    // cost of O(s^2 n) block updates + a tiny s x s Cholesky per iteration.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BlockCgTolJobFProxy : IJob
    {
        public fProxyMxN A;               // n x n SPD
        public fProxyMxN B;               // s x n block RHS
        public fProxyMxN X, R, P, Q;      // s x n block scratch (X = solution)
        public int K; public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            int s = B.M_Rows, n = B.N_Cols;
            for (int i = 0; i < s; i++) for (int c = 0; c < n; c++) X[i, c] = (fProxy)0;
            var info = Krylov.bcg(new fProxyDenseOperator(in A), in B, ref X, ref R, ref P, ref Q, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ScalarLoopTolJobFProxy : IJob
    {
        public fProxyMxN A;               // n x n SPD
        public fProxyMxN B;               // s x n block RHS
        public fProxyN x, r, p, Ap, bcol; // reused per column
        public int S, K; public fProxy Tol;
        public Indices Iters;             // total iterations summed over the s columns

        public void Execute()
        {
            int n = B.N_Cols;
            var op = new fProxyDenseOperator(in A);
            int total = 0;
            for (int j = 0; j < S; j++)
            {
                for (int c = 0; c < n; c++) { bcol[c] = B[j, c]; x[c] = (fProxy)0; }
                var info = Krylov.cg(in op, in bcol, ref x, ref r, ref p, ref Ap, K, Tol);
                total += info.iterations;
            }
            Iters[0] = total;
        }
    }

    public static partial class BlockCGBenchmark
    {
        static string BenchFProxy(int n, int s)
        {
            var arena = new Arena(Allocator.Persistent);
            const string fmt = "{0,-7}{1,-7}{2,-4}{3,-14}{4,10:F4}{5,12:F4}{6,8}";

            var M = arena.fProxyMat(n, n);
            var A = arena.fProxyMat(n, n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(n * 131 + s));
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++) M[row, col] = rng.NextFProxy(-1f, 1f);
            Blas.dot(in M, in M, ref A, true);                    // A = M^T M
            for (int d = 0; d < n; d++) A[d, d] += n;             // + n I  -> SPD, cond grows with n

            var B = arena.fProxyMat(s, n);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) B[i, c] = rng.NextFProxy(-1f, 1f);

            fProxy tol = Consts.fProxySqrtEps;
            int cap = 4 * n;
            var iters = arena.Indices(1);
            var sb = new StringBuilder();

            // block-CG
            var X = arena.fProxyMat(s, n); var R = arena.fProxyMat(s, n);
            var P = arena.fProxyMat(s, n); var Q = arena.fProxyMat(s, n);
            var blockJob = new BlockCgTolJobFProxy { A = A, B = B, X = X, R = R, P = P, Q = Q, K = cap, Tol = tol, Iters = iters };
            var blockStat = Bench.Time(() => blockJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "block-CG", blockStat.Median, blockStat.Min, iters[0]));

            // scalar loop of s independent cg solves
            var x = arena.fProxyVec(n); var r = arena.fProxyVec(n); var p = arena.fProxyVec(n);
            var Ap = arena.fProxyVec(n); var bcol = arena.fProxyVec(n);
            var loopJob = new ScalarLoopTolJobFProxy { A = A, B = B, x = x, r = r, p = p, Ap = Ap, bcol = bcol, S = s, K = cap, Tol = tol, Iters = iters };
            var loopStat = Bench.Time(() => loopJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, s, "scalar x s", loopStat.Median, loopStat.Min, iters[0]));

            arena.Dispose();
            return sb.ToString();
        }
    }
}
