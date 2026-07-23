using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of BlockArnoldiBenchmark: bgmres + bgcrodr timed IJobs (task #70 --
    // measures the block-Arnoldi step's row-orthonormalization cost, the sole consumer LQRP.decomp +
    // LQRPRankFloored fed before being replaced by Krylov.RowOrthoRankFloored) + the build+measure
    // method. The dtype-agnostic harness (Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/BlockArnoldiBenchmark.cs.
    //
    // Two RHS blocks per (n, s): "distinct" (full row rank, no deflation) and "dup" (row s-1 forced
    // equal to row 0, mirrors KrylovBlockBatteryTests' checks #8/#9 -- forces the block-Arnoldi basis
    // to deflate every step, exercising the rank-revealing path being optimized).

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BgmresBlockJobFProxy : IJob
    {
        public fProxyMxN A;          // n x n dense general
        public fProxyMxN B;          // s x n block RHS
        public fProxyMxN X;          // s x n block solution
        public int Restart, MaxIter; public fProxy Tol;
        public Indices Out;          // [0]=iterations [1]=minActive [2]=(int)status

        public void Execute()
        {
            int s = B.M_Rows, n = B.N_Cols;
            for (int i = 0; i < s; i++) for (int c = 0; c < n; c++) X[i, c] = (fProxy)0;
            var info = Krylov.bgmres(in A, in B, ref X, Restart, MaxIter, Tol);
            Out[0] = info.iterations;
            Out[1] = info.minActive;
            Out[2] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BgcrodrBlockJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyMxN X;
        public int Restart, Recycle, MaxIter; public fProxy Tol;
        public Indices Out;

        public void Execute()
        {
            int s = B.M_Rows, n = B.N_Cols;
            for (int i = 0; i < s; i++) for (int c = 0; c < n; c++) X[i, c] = (fProxy)0;
            var info = Krylov.bgcrodr(in A, in B, ref X, Restart, Recycle, MaxIter, Tol);
            Out[0] = info.iterations;
            Out[1] = info.minActive;
            Out[2] = (int)info.status;
        }
    }

    public static partial class BlockArnoldiBenchmark
    {
        static string BenchFProxy(int n, int s, int restart, int maxIter, int recycle)
        {
            const string fmt = "{0,-7}{1,-6}{2,-4}{3,-10}{4,-9}{5,10:F4}{6,12:F4}{7,8}{8,8}{9,14}";

            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-1), (fProxy)1, 0x5EED1u ^ (uint)(n * 131 + s), Allocator.Persistent);
            for (int d = 0; d < n; d++) A[d, d] += (fProxy)(2 * n);   // diagonally dominant, nonsymmetric

            fProxy tol = Consts.fProxySqrtEps;
            var sb = new StringBuilder();

            void RunPair(string tag, in fProxyMxN B)
            {
                var Xg = new fProxyMxN(s, n, Allocator.Persistent);
                var outGm = new Indices(3, Allocator.Persistent);
                var jobGm = new BgmresBlockJobFProxy { A = A, B = B, X = Xg, Restart = restart, MaxIter = maxIter, Tol = tol, Out = outGm };
                var stGm = Bench.Time(() => jobGm.Run());
                sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                    "fProxy", n, s, tag, "bgmres", stGm.Median, stGm.Min, outGm[0], outGm[1], (IterativeSolveStatus)outGm[2]));

                var Xc = new fProxyMxN(s, n, Allocator.Persistent);
                var outGc = new Indices(3, Allocator.Persistent);
                var jobGc = new BgcrodrBlockJobFProxy { A = A, B = B, X = Xc, Restart = restart, Recycle = recycle, MaxIter = maxIter, Tol = tol, Out = outGc };
                var stGc = Bench.Time(() => jobGc.Run());
                sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                    "fProxy", n, s, tag, "bgcrodr", stGc.Median, stGc.Min, outGc[0], outGc[1], (IterativeSolveStatus)outGc[2]));

                Xg.Dispose(); outGm.Dispose(); Xc.Dispose(); outGc.Dispose();
            }

            var Bdistinct = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1), (fProxy)1, 0xB10c1u ^ (uint)(n * 131 + s), Allocator.Persistent);
            RunPair("distinct", in Bdistinct);

            var Bdup = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1), (fProxy)1, 0xB10c2u ^ (uint)(n * 131 + s), Allocator.Persistent);
            for (int c = 0; c < n; c++) Bdup[s - 1, c] = Bdup[0, c];
            RunPair("dup", in Bdup);

            A.Dispose(); Bdistinct.Dispose(); Bdup.Dispose();
            return sb.ToString();
        }
    }
}
