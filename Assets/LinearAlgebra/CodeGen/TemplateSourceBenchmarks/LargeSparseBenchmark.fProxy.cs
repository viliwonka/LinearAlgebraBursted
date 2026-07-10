using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Sparse;
using LinearAlgebra.Gallery;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LargeSparseBenchmark (timed IJobs, the LOBPCG report helper, the
    // residual helper, and the per-family build+measure methods). The dtype-agnostic harness
    // (constants, table formatters, Run/RunLobpcg/Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LargeSparseBenchmark.cs; shared formatters live in the public
    // LargeSparseFmt helper there.
    //
    // Every timed Krylov job writes its SolveInfo/LstsqInfo (status, iterations) into `outInfo` so the
    // report can show iterations-executed alongside wall clock -- a fixed K=40, tol=0 timing can exit
    // early on a breakdown guard and look "faster" while doing strictly less work (see benchmark
    // hygiene note in docs/draft-spec-krylov-optimization.md); iters+status makes that visible instead
    // of silently masquerading as speed.

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpmvJobFProxy : IJob { public fProxyBSR A; public fProxyN x, y; public int reps;
        public void Execute() { for (int k = 0; k < reps; k++) { if ((k & 1) == 0) BSR.spMV(in A, in x, ref y); else BSR.spMV(in A, in y, ref x); } } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCgJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, r, p, Ap; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpPcgJobFProxy : IJob { public fProxyBSR A; public fProxyBlockJacobi M; public fProxyN b, x, r, p, Ap, z; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpMinresJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, y, r1, r2, v, w, w1, w2; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpBicgJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, r, rHat0, p, v, t; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCglsJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, r, s, p, q; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLsqrJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, u, v, w, tmpM, tmpN; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLsmrJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, u, v, h, hbar, tmpM, tmpN; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    // Eigen job writes [produced, solved, 0] into outInfo (reference-backed, visible after Run).
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLanczosJobFProxy : IJob { public fProxyBSR A; public fProxyLanczosCache ws; public fProxyN vals; public int steps; public NativeArray<double> outInfo;
        public void Execute() { var info = Eigen.lanczos(in A, ref ws, ref vals, steps); outInfo[0] = info.produced; outInfo[1] = info.Solved ? 1 : 0; outInfo[2] = 0; } }

    // LOBPCG smallest-k eigenpairs. outInfo = [status, iterations, converged, maxResidual, orthoErr].
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLobpcgJobFProxy : IJob { public fProxyBSR A; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { var info = Eigen.lobpcg(in A, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLobpcgPrecJobFProxy : IJob { public fProxyBSR A; public fProxyBlockJacobi M; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { var info = Eigen.lobpcg(in A, in M, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

    // Output-orthonormality check for the LOBPCG rows. float/double emit WriteFloat/WriteDouble into the
    // same partial class (they merge in the Benchmarks assembly).
    static partial class LobpcgReport
    {
        public static void WriteFProxy(in LOBPCGInfo info, in fProxyLOBPCGCache ws, int k, NativeArray<double> o)
        {
            double orth = 0;
            for (int i = 0; i < k; i++)
                for (int j = i; j < k; j++)
                {
                    double d = 0;
                    for (int c = 0; c < ws.X.N_Cols; c++) d += (double)ws.X[i, c] * (double)ws.X[j, c];
                    double e = math.abs(d - (i == j ? 1.0 : 0.0));
                    if (e > orth) orth = e;
                }
            o[0] = (int)info.status; o[1] = info.iterations; o[2] = info.converged; o[3] = info.maxResidual; o[4] = orth;
        }
    }

    public static partial class LargeSparseBenchmark
    {
        static double Res(in fProxyBSR A, in fProxyN x, in fProxyN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { double d = (double)Ax[i] - (double)b[i]; num += d * d; den += (double)b[i] * (double)b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        static void BenchKrylovFProxy(StringBuilder sb, int BR, int[] Ns, fProxy density, int K, int spmvReps)
        {
            var oi = new NativeArray<double>(2, Allocator.Persistent);

            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.fProxyRandomSparseSPD(nb, BR, density, 0x5A17u);
                var M = arena.fProxyBlockJacobi(in A);
                var xKnown = arena.fProxyRandomVec(N, 0.5f, 1.5f, 0xB0Bu);
                var b = arena.fProxyVec(N); BSR.spMV(in A, in xKnown, ref b);
                string sz = N.ToString();

                var sx = arena.fProxyRandomVec(N, -1f, 1f, 7u); var sy = arena.fProxyVec(N);
                var spmvJob = new SpmvJobFProxy { A = A, x = sx, y = sy, reps = spmvReps };
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "spMV x" + spmvReps, Bench.Time(() => spmvJob.Run()), 0.0, 0, 0));

                var x = arena.fProxyVec(N);
                var cgJob = new SpCgJobFProxy { A = A, b = b, x = x, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), K = K, outInfo = oi };
                var cgStat = Bench.Time(() => cgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG", cgStat, Res(in A, in x, in b), (int)oi[1], (int)oi[0]));
                var xp = arena.fProxyVec(N);
                var pcgJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xp, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = K, outInfo = oi };
                var pcgStat = Bench.Time(() => pcgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi", pcgStat, Res(in A, in xp, in b), (int)oi[1], (int)oi[0]));
                var xm = arena.fProxyVec(N);
                var mrJob = new SpMinresJobFProxy { A = A, b = b, x = xm, y = arena.fProxyVec(N), r1 = arena.fProxyVec(N), r2 = arena.fProxyVec(N), v = arena.fProxyVec(N), w = arena.fProxyVec(N), w1 = arena.fProxyVec(N), w2 = arena.fProxyVec(N), K = K, outInfo = oi };
                var mrStat = Bench.Time(() => mrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "MINRES", mrStat, Res(in A, in xm, in b), (int)oi[1], (int)oi[0]));

                var An = arena.fProxyRandomSparse(nb, nb, BR, density, 0x1234u);
                var bn = arena.fProxyVec(N); BSR.spMV(in An, in xKnown, ref bn);
                var xn = arena.fProxyVec(N);
                var bicgJob = new SpBicgJobFProxy { A = An, b = bn, x = xn, r = arena.fProxyVec(N), rHat0 = arena.fProxyVec(N), p = arena.fProxyVec(N), v = arena.fProxyVec(N), t = arena.fProxyVec(N), K = K, outInfo = oi };
                var bicgStat = Bench.Time(() => bicgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "BiCGStab", bicgStat, Res(in An, in xn, in bn), (int)oi[1], (int)oi[0]));

                int mb = 2 * nb, m = mb * BR;
                var At = arena.fProxyRandomSparse(mb, nb, BR, density, 0xC0DEu);
                var bt = arena.fProxyVec(m); BSR.spMV(in At, in xKnown, ref bt);
                string rsz = m + "x" + N;
                var xc = arena.fProxyVec(N);
                var cglsJob = new SpCglsJobFProxy { A = At, b = bt, x = xc, r = arena.fProxyVec(m), s = arena.fProxyVec(N), p = arena.fProxyVec(N), q = arena.fProxyVec(m), K = K, outInfo = oi };
                var cglsStat = Bench.Time(() => cglsJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", rsz, "CGLS", cglsStat, Res(in At, in xc, in bt), (int)oi[1], (int)oi[0]));
                var xl = arena.fProxyVec(N);
                var lsqrJob = new SpLsqrJobFProxy { A = At, b = bt, x = xl, u = arena.fProxyVec(m), v = arena.fProxyVec(N), w = arena.fProxyVec(N), tmpM = arena.fProxyVec(m), tmpN = arena.fProxyVec(N), K = K, outInfo = oi };
                var lsqrStat = Bench.Time(() => lsqrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", rsz, "LSQR", lsqrStat, Res(in At, in xl, in bt), (int)oi[1], (int)oi[0]));
                var xr = arena.fProxyVec(N);
                var lsmrJob = new SpLsmrJobFProxy { A = At, b = bt, x = xr, u = arena.fProxyVec(m), v = arena.fProxyVec(N), h = arena.fProxyVec(N), hbar = arena.fProxyVec(N), tmpM = arena.fProxyVec(m), tmpN = arena.fProxyVec(N), K = K, outInfo = oi };
                var lsmrStat = Bench.Time(() => lsmrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", rsz, "LSMR", lsmrStat, Res(in At, in xr, in bt), (int)oi[1], (int)oi[0]));

                arena.Dispose();
            }

            oi.Dispose();
        }

        // b=1 (scalar BSR) stencil section: fProxyLaplacian2D(1, N) collapses the generator's
        // x-neighbor coupling (grid width 1), leaving a genuine SCALAR tridiagonal SPD system
        // (diag=4, off-diag=-1, nnz ~= 3N) -- the low-fill, b=1 regime where R1's vector-op fusion
        // is the largest fraction of per-iteration traffic (spec: BR=4/1.5% fill spMV moves ~6.7MB
        // vs ~0.5MB of vector sweeps per matvec; a scalar/low-fill stencil inverts that ratio).
        // Only the SPD-compatible solvers run here (CG/PCG-Jacobi/MINRES) -- BiCGStab needs a
        // non-symmetric operator and CGLS/LSQR/LSMR need a rectangular one, neither of which this
        // generator produces.
        static void BenchStencilFProxy(StringBuilder sb, int[] Ns, int K)
        {
            var oi = new NativeArray<double>(2, Allocator.Persistent);

            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyLaplacian2D(1, N);
                var M = arena.fProxyBlockJacobi(in A);
                var xKnown = arena.fProxyRandomVec(N, 0.5f, 1.5f, 0xB0Bu);
                var b = arena.fProxyVec(N); BSR.spMV(in A, in xKnown, ref b);
                string sz = N.ToString();

                var x = arena.fProxyVec(N);
                var cgJob = new SpCgJobFProxy { A = A, b = b, x = x, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), K = K, outInfo = oi };
                var cgStat = Bench.Time(() => cgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG", cgStat, Res(in A, in x, in b), (int)oi[1], (int)oi[0]));
                var xp = arena.fProxyVec(N);
                var pcgJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xp, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = K, outInfo = oi };
                var pcgStat = Bench.Time(() => pcgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi", pcgStat, Res(in A, in xp, in b), (int)oi[1], (int)oi[0]));
                var xm = arena.fProxyVec(N);
                var mrJob = new SpMinresJobFProxy { A = A, b = b, x = xm, y = arena.fProxyVec(N), r1 = arena.fProxyVec(N), r2 = arena.fProxyVec(N), v = arena.fProxyVec(N), w = arena.fProxyVec(N), w1 = arena.fProxyVec(N), w2 = arena.fProxyVec(N), K = K, outInfo = oi };
                var mrStat = Bench.Time(() => mrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "MINRES", mrStat, Res(in A, in xm, in b), (int)oi[1], (int)oi[0]));

                arena.Dispose();
            }

            oi.Dispose();
        }

        static void BenchEigenFProxy(StringBuilder sb, int BR, int[] Ns, fProxy density, int lanczosSteps)
        {
            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.fProxyRandomSparseSPD(nb, BR, density, 0x5A17u);
                string sz = N.ToString();
                var outInfo = new NativeArray<double>(3, Allocator.Persistent);

                var lws = arena.fProxyLanczosCache(N, lanczosSteps);
                var lvals = arena.fProxyVec(lanczosSteps);
                var lanJob = new SpLanczosJobFProxy { A = A, ws = lws, vals = lvals, steps = lanczosSteps, outInfo = outInfo };
                var lanStat = Bench.Time(() => lanJob.Run());
                sb.AppendLine(LargeSparseFmt.EigRow("fProxy", sz, "Lanczos s=" + lanczosSteps, lanStat, new[] { outInfo[0], outInfo[1], outInfo[2] }));

                outInfo.Dispose();
                arena.Dispose();
            }
        }

        static void BenchLobpcgFProxy(StringBuilder sb, int[] eigGrids, int lobpcgK, int lobpcgGuard, int lobpcgMaxIter)
        {
            foreach (int g in eigGrids)
            {
                var arena = new Arena(Allocator.Persistent);
                int n = g * g;
                var A = arena.fProxyLaplacian2D(g, g);
                var M = arena.fProxyBlockJacobi(in A);
                string grid = g + "x" + g + "(" + n + ")";
                var oi = new NativeArray<double>(5, Allocator.Persistent);
                fProxy tol = Consts.fProxySqrtEps;

                new SpLobpcgJobFProxy { A = A, ws = arena.fProxyLOBPCGCache(n, lobpcgK), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi }.Run();
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "none", 0, LargeSparseFmt.Snap(oi));
                new SpLobpcgPrecJobFProxy { A = A, M = M, ws = arena.fProxyLOBPCGCache(n, lobpcgK), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi }.Run();
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "blockJac", 0, LargeSparseFmt.Snap(oi));
                new SpLobpcgJobFProxy { A = A, ws = arena.fProxyLOBPCGCache(n, lobpcgK + lobpcgGuard), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi }.Run();
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "none", lobpcgGuard, LargeSparseFmt.Snap(oi));
                new SpLobpcgPrecJobFProxy { A = A, M = M, ws = arena.fProxyLOBPCGCache(n, lobpcgK + lobpcgGuard), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi }.Run();
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "blockJac", lobpcgGuard, LargeSparseFmt.Snap(oi));

                oi.Dispose();
                arena.Dispose();
            }
        }
    }
}
