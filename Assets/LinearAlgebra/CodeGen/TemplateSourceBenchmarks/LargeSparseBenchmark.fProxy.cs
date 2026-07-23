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
    // early on a breakdown guard and look "faster" while doing strictly less work; iters+status makes
    // that visible instead of silently masquerading as speed.

    // SpCgJobFProxy/SpPcgJobFProxy's `tol` field serves both the fixed-K/tol=0 throughput rows
    // (default 0 runs the full K budget) and the iterations-to-convergence rows, which set a
    // real tol/maxIter.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpCgJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, r, p, Ap; public int K; public fProxy tol; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, tol); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpPcgJobFProxy : IJob { public fProxyBSR A; public fProxyBlockJacobi M; public fProxyN b, x, r, p, Ap, z; public int K; public fProxy tol; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, tol); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    // SSOR twin of SpPcgJobFProxy (same reuse for fixed-K throughput and convergence-comparison
    // rows).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpPcgSSORJobFProxy : IJob { public fProxyBSR A; public fProxySSOR M; public fProxyN b, x, r, p, Ap, z; public int K; public fProxy tol; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, tol); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpMinresJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, y, r1, r2, v, w, w1, w2; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpBicgJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, r, rHat0, p, v, t; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpLsqrJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, u, v, w, tmpM, tmpN; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpLsmrJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, u, v, h, hbar, tmpM, tmpN; public int K; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, K, 0f); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    // Eigen job writes [produced, solved, 0] into outInfo (reference-backed, visible after Run).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpLanczosJobFProxy : IJob { public fProxyBSR A; public fProxyLanczosCache ws; public fProxyN vals; public int steps; public NativeArray<double> outInfo;
        public void Execute() { var info = Eigen.lanczos(in A, ref ws, ref vals, steps); outInfo[0] = info.produced; outInfo[1] = info.Solved ? 1 : 0; outInfo[2] = 0; } }

    // LOBPCG smallest-k eigenpairs. outInfo = [status, iterations, converged, maxResidual, orthoErr].
    // Timed via Bench.Time (1 warmup + 4 timed .Run() calls on the SAME captured `ws`). lobpcg only
    // reseeds ws.X when it is all-zero (its own warm-start contract), so re-running the SAME ws
    // without resetting X would warm-start every timed sample from the PREVIOUS sample's converged
    // eigenvectors and measure ~0 iterations from the 2nd run on. Zeroing X at the top of every
    // Execute() forces the SAME deterministic reseed (lobpcg's own fixed-seed 0x9E3779B1u fill) on
    // every sample -- a fair, reproducible, cold-start measurement each time, mirroring the x[i]=0
    // reset every Krylov job above already does.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpLobpcgJobFProxy : IJob { public fProxyBSR A; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < ws.X.M_Rows; i++) for (int c = 0; c < ws.X.N_Cols; c++) ws.X[i, c] = (fProxy)0; var info = Eigen.lobpcg(in A, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpLobpcgPrecJobFProxy : IJob { public fProxyBSR A; public fProxyBlockJacobi M; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < ws.X.M_Rows; i++) for (int c = 0; c < ws.X.N_Cols; c++) ws.X[i, c] = (fProxy)0; var info = Eigen.lobpcg(in A, in M, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

    // SSOR preconditioner axis for LOBPCG. Only fProxyBlockJacobi has a dedicated
    // `lobpcg(in fProxyBSR, in TPre, ...)` overload -- fProxySSOR goes through the generic
    // `lobpcg<TOp,TPre>` core via fProxyBSROperator.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SpLobpcgSSORJobFProxy : IJob { public fProxyBSR A; public fProxySSOR M; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < ws.X.M_Rows; i++) for (int c = 0; c < ws.X.N_Cols; c++) ws.X[i, c] = (fProxy)0; var info = Eigen.lobpcg(new fProxyBSROperator(in A), in M, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

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

        // The PCG rows carry a preconditioner axis (none/CG, block-Jacobi, SSOR). "none" is CG
        // itself (algebraically PCG with M=I, same recurrence) rather than a redundant literal
        // PCG-identity row. The fixed-K/tol=0 rows below measure per-ITERATION wall-clock cost
        // (every solver runs the full K budget, so iters is uninformative there by design); the
        // SEPARATE "@tol" rows at the end of the largest N run to a REAL tolerance and are where
        // SSOR's iteration-count win is actually visible.
        static void BenchKrylovFProxy(StringBuilder sb, int BR, int[] Ns, fProxy density, int K)
        {
            var oi = new NativeArray<double>(2, Allocator.Persistent);

            foreach (int N in Ns)
            {
                int nb = N / BR;
                var A = fProxyGallery.fProxyRandomSparseSPD(nb, BR, density, 0x5A17u, Allocator.Persistent);
                var M = new fProxyBlockJacobi(in A, Allocator.Persistent);
                var ssor = new fProxySSOR(in A, Allocator.Persistent);
                var xKnown = GenerateOP.fProxyRandomVec(N, 0.5f, 1.5f, 0xB0Bu, Allocator.Persistent);
                var b = new fProxyN(N, Allocator.Persistent); BSR.spMV(in A, in xKnown, ref b);
                string sz = N.ToString();

                var x = new fProxyN(N, Allocator.Persistent);
                var cgR = new fProxyN(N, Allocator.Persistent); var cgP = new fProxyN(N, Allocator.Persistent); var cgAp = new fProxyN(N, Allocator.Persistent);
                var cgJob = new SpCgJobFProxy { A = A, b = b, x = x, r = cgR, p = cgP, Ap = cgAp, K = K, outInfo = oi };
                var cgStat = Bench.Time(() => cgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG", cgStat, Res(in A, in x, in b), (int)oi[1], (int)oi[0]));
                var xp = new fProxyN(N, Allocator.Persistent);
                var pcgR = new fProxyN(N, Allocator.Persistent); var pcgP = new fProxyN(N, Allocator.Persistent); var pcgAp = new fProxyN(N, Allocator.Persistent); var pcgZ = new fProxyN(N, Allocator.Persistent);
                var pcgJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xp, r = pcgR, p = pcgP, Ap = pcgAp, z = pcgZ, K = K, outInfo = oi };
                var pcgStat = Bench.Time(() => pcgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi", pcgStat, Res(in A, in xp, in b), (int)oi[1], (int)oi[0]));
                var xs = new fProxyN(N, Allocator.Persistent);
                var ssorR = new fProxyN(N, Allocator.Persistent); var ssorP = new fProxyN(N, Allocator.Persistent); var ssorAp = new fProxyN(N, Allocator.Persistent); var ssorZ = new fProxyN(N, Allocator.Persistent);
                var ssorJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xs, r = ssorR, p = ssorP, Ap = ssorAp, z = ssorZ, K = K, outInfo = oi };
                var ssorStat = Bench.Time(() => ssorJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR", ssorStat, Res(in A, in xs, in b), (int)oi[1], (int)oi[0]));

                // Iterations-to-CONVERGENCE (real tol, generous maxIter) -- ONE size only
                // (largest N: the trend is visible at one size; budget discipline). This is the
                // row set where SSOR's iteration-count win is visible; the fixed-K/tol=0 rows
                // above cannot show it (every solver there runs the full K by construction).
                fProxyN xc1 = default, cgConvR = default, cgConvP = default, cgConvAp = default;
                fProxyN xc2 = default, pcgConvR = default, pcgConvP = default, pcgConvAp = default, pcgConvZ = default;
                fProxyN xc3 = default, ssorConvR = default, ssorConvP = default, ssorConvAp = default, ssorConvZ = default;
                bool ranConv = N == Ns[Ns.Length - 1];
                if (ranConv)
                {
                    fProxy convTol = Consts.fProxySqrtEps;
                    int convMaxIter = 8 * N;

                    xc1 = new fProxyN(N, Allocator.Persistent);
                    cgConvR = new fProxyN(N, Allocator.Persistent); cgConvP = new fProxyN(N, Allocator.Persistent); cgConvAp = new fProxyN(N, Allocator.Persistent);
                    var cgConvJob = new SpCgJobFProxy { A = A, b = b, x = xc1, r = cgConvR, p = cgConvP, Ap = cgConvAp, K = convMaxIter, tol = convTol, outInfo = oi };
                    var cgConvStat = Bench.Time(() => cgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG@tol", cgConvStat, Res(in A, in xc1, in b), (int)oi[1], (int)oi[0]));

                    xc2 = new fProxyN(N, Allocator.Persistent);
                    pcgConvR = new fProxyN(N, Allocator.Persistent); pcgConvP = new fProxyN(N, Allocator.Persistent); pcgConvAp = new fProxyN(N, Allocator.Persistent); pcgConvZ = new fProxyN(N, Allocator.Persistent);
                    var pcgConvJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xc2, r = pcgConvR, p = pcgConvP, Ap = pcgConvAp, z = pcgConvZ, K = convMaxIter, tol = convTol, outInfo = oi };
                    var pcgConvStat = Bench.Time(() => pcgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi@tol", pcgConvStat, Res(in A, in xc2, in b), (int)oi[1], (int)oi[0]));

                    xc3 = new fProxyN(N, Allocator.Persistent);
                    ssorConvR = new fProxyN(N, Allocator.Persistent); ssorConvP = new fProxyN(N, Allocator.Persistent); ssorConvAp = new fProxyN(N, Allocator.Persistent); ssorConvZ = new fProxyN(N, Allocator.Persistent);
                    var ssorConvJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xc3, r = ssorConvR, p = ssorConvP, Ap = ssorConvAp, z = ssorConvZ, K = convMaxIter, tol = convTol, outInfo = oi };
                    var ssorConvStat = Bench.Time(() => ssorConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR@tol", ssorConvStat, Res(in A, in xc3, in b), (int)oi[1], (int)oi[0]));
                }

                var xm = new fProxyN(N, Allocator.Persistent);
                var mrY = new fProxyN(N, Allocator.Persistent); var mrR1 = new fProxyN(N, Allocator.Persistent); var mrR2 = new fProxyN(N, Allocator.Persistent);
                var mrV = new fProxyN(N, Allocator.Persistent); var mrW = new fProxyN(N, Allocator.Persistent); var mrW1 = new fProxyN(N, Allocator.Persistent); var mrW2 = new fProxyN(N, Allocator.Persistent);
                var mrJob = new SpMinresJobFProxy { A = A, b = b, x = xm, y = mrY, r1 = mrR1, r2 = mrR2, v = mrV, w = mrW, w1 = mrW1, w2 = mrW2, K = K, outInfo = oi };
                var mrStat = Bench.Time(() => mrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "MINRES", mrStat, Res(in A, in xm, in b), (int)oi[1], (int)oi[0]));

                var An = fProxyGallery.fProxyRandomSparse(nb, nb, BR, density, 0x1234u, Allocator.Persistent);
                var bn = new fProxyN(N, Allocator.Persistent); BSR.spMV(in An, in xKnown, ref bn);
                var xn = new fProxyN(N, Allocator.Persistent);
                var bicgR = new fProxyN(N, Allocator.Persistent); var bicgRHat0 = new fProxyN(N, Allocator.Persistent); var bicgP = new fProxyN(N, Allocator.Persistent);
                var bicgV = new fProxyN(N, Allocator.Persistent); var bicgT = new fProxyN(N, Allocator.Persistent);
                var bicgJob = new SpBicgJobFProxy { A = An, b = bn, x = xn, r = bicgR, rHat0 = bicgRHat0, p = bicgP, v = bicgV, t = bicgT, K = K, outInfo = oi };
                var bicgStat = Bench.Time(() => bicgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "BiCGStab", bicgStat, Res(in An, in xn, in bn), (int)oi[1], (int)oi[0]));

                int mb = 2 * nb, m = mb * BR;
                var At = fProxyGallery.fProxyRandomSparse(mb, nb, BR, density, 0xC0DEu, Allocator.Persistent);
                var bt = new fProxyN(m, Allocator.Persistent); BSR.spMV(in At, in xKnown, ref bt);
                string rsz = m + "x" + N;
                var xl = new fProxyN(N, Allocator.Persistent);
                var lsqrU = new fProxyN(m, Allocator.Persistent); var lsqrV = new fProxyN(N, Allocator.Persistent); var lsqrW = new fProxyN(N, Allocator.Persistent);
                var lsqrTmpM = new fProxyN(m, Allocator.Persistent); var lsqrTmpN = new fProxyN(N, Allocator.Persistent);
                var lsqrJob = new SpLsqrJobFProxy { A = At, b = bt, x = xl, u = lsqrU, v = lsqrV, w = lsqrW, tmpM = lsqrTmpM, tmpN = lsqrTmpN, K = K, outInfo = oi };
                var lsqrStat = Bench.Time(() => lsqrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", rsz, "LSQR", lsqrStat, Res(in At, in xl, in bt), (int)oi[1], (int)oi[0]));
                var xr = new fProxyN(N, Allocator.Persistent);
                var lsmrU = new fProxyN(m, Allocator.Persistent); var lsmrV = new fProxyN(N, Allocator.Persistent); var lsmrH = new fProxyN(N, Allocator.Persistent); var lsmrHbar = new fProxyN(N, Allocator.Persistent);
                var lsmrTmpM = new fProxyN(m, Allocator.Persistent); var lsmrTmpN = new fProxyN(N, Allocator.Persistent);
                var lsmrJob = new SpLsmrJobFProxy { A = At, b = bt, x = xr, u = lsmrU, v = lsmrV, h = lsmrH, hbar = lsmrHbar, tmpM = lsmrTmpM, tmpN = lsmrTmpN, K = K, outInfo = oi };
                var lsmrStat = Bench.Time(() => lsmrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", rsz, "LSMR", lsmrStat, Res(in At, in xr, in bt), (int)oi[1], (int)oi[0]));

                A.Dispose(); M.Dispose(); ssor.Dispose(); xKnown.Dispose(); b.Dispose();
                x.Dispose(); cgR.Dispose(); cgP.Dispose(); cgAp.Dispose();
                xp.Dispose(); pcgR.Dispose(); pcgP.Dispose(); pcgAp.Dispose(); pcgZ.Dispose();
                xs.Dispose(); ssorR.Dispose(); ssorP.Dispose(); ssorAp.Dispose(); ssorZ.Dispose();
                if (ranConv)
                {
                    xc1.Dispose(); cgConvR.Dispose(); cgConvP.Dispose(); cgConvAp.Dispose();
                    xc2.Dispose(); pcgConvR.Dispose(); pcgConvP.Dispose(); pcgConvAp.Dispose(); pcgConvZ.Dispose();
                    xc3.Dispose(); ssorConvR.Dispose(); ssorConvP.Dispose(); ssorConvAp.Dispose(); ssorConvZ.Dispose();
                }
                xm.Dispose(); mrY.Dispose(); mrR1.Dispose(); mrR2.Dispose(); mrV.Dispose(); mrW.Dispose(); mrW1.Dispose(); mrW2.Dispose();
                An.Dispose(); bn.Dispose(); xn.Dispose(); bicgR.Dispose(); bicgRHat0.Dispose(); bicgP.Dispose(); bicgV.Dispose(); bicgT.Dispose();
                At.Dispose(); bt.Dispose();
                xl.Dispose(); lsqrU.Dispose(); lsqrV.Dispose(); lsqrW.Dispose(); lsqrTmpM.Dispose(); lsqrTmpN.Dispose();
                xr.Dispose(); lsmrU.Dispose(); lsmrV.Dispose(); lsmrH.Dispose(); lsmrHbar.Dispose(); lsmrTmpM.Dispose(); lsmrTmpN.Dispose();
            }

            oi.Dispose();
        }

        // b=1 (scalar BSR) stencil section: fProxyLaplacian2D(1, N) collapses the generator's
        // x-neighbor coupling (grid width 1), leaving a genuine SCALAR tridiagonal SPD system
        // (diag=4, off-diag=-1, nnz ~= 3N) -- the low-fill, b=1 regime where vector-op fusion is
        // the largest fraction of per-iteration traffic. Only the SPD-compatible solvers run here
        // (CG/PCG-Jacobi/PCG-SSOR/MINRES) -- BiCGStab needs a non-symmetric operator and
        // LSQR/LSMR need a rectangular one, neither of which this generator produces. PCG-SSOR
        // carries both the fixed-K row and the "@tol" convergence-comparison row at the largest N,
        // same reasoning as BenchKrylovFProxy's own comment.
        static void BenchStencilFProxy(StringBuilder sb, int[] Ns, int K)
        {
            var oi = new NativeArray<double>(2, Allocator.Persistent);

            foreach (int N in Ns)
            {
                var A = fProxyGallery.fProxyLaplacian2D(1, N, Allocator.Persistent);
                var M = new fProxyBlockJacobi(in A, Allocator.Persistent);
                var ssor = new fProxySSOR(in A, Allocator.Persistent);
                var xKnown = GenerateOP.fProxyRandomVec(N, 0.5f, 1.5f, 0xB0Bu, Allocator.Persistent);
                var b = new fProxyN(N, Allocator.Persistent); BSR.spMV(in A, in xKnown, ref b);
                string sz = N.ToString();

                var x = new fProxyN(N, Allocator.Persistent);
                var cgR = new fProxyN(N, Allocator.Persistent); var cgP = new fProxyN(N, Allocator.Persistent); var cgAp = new fProxyN(N, Allocator.Persistent);
                var cgJob = new SpCgJobFProxy { A = A, b = b, x = x, r = cgR, p = cgP, Ap = cgAp, K = K, outInfo = oi };
                var cgStat = Bench.Time(() => cgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG", cgStat, Res(in A, in x, in b), (int)oi[1], (int)oi[0]));
                var xp = new fProxyN(N, Allocator.Persistent);
                var pcgR = new fProxyN(N, Allocator.Persistent); var pcgP = new fProxyN(N, Allocator.Persistent); var pcgAp = new fProxyN(N, Allocator.Persistent); var pcgZ = new fProxyN(N, Allocator.Persistent);
                var pcgJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xp, r = pcgR, p = pcgP, Ap = pcgAp, z = pcgZ, K = K, outInfo = oi };
                var pcgStat = Bench.Time(() => pcgJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi", pcgStat, Res(in A, in xp, in b), (int)oi[1], (int)oi[0]));
                var xs = new fProxyN(N, Allocator.Persistent);
                var ssorR = new fProxyN(N, Allocator.Persistent); var ssorP = new fProxyN(N, Allocator.Persistent); var ssorAp = new fProxyN(N, Allocator.Persistent); var ssorZ = new fProxyN(N, Allocator.Persistent);
                var ssorJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xs, r = ssorR, p = ssorP, Ap = ssorAp, z = ssorZ, K = K, outInfo = oi };
                var ssorStat = Bench.Time(() => ssorJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR", ssorStat, Res(in A, in xs, in b), (int)oi[1], (int)oi[0]));
                var xm = new fProxyN(N, Allocator.Persistent);
                var mrY = new fProxyN(N, Allocator.Persistent); var mrR1 = new fProxyN(N, Allocator.Persistent); var mrR2 = new fProxyN(N, Allocator.Persistent);
                var mrV = new fProxyN(N, Allocator.Persistent); var mrW = new fProxyN(N, Allocator.Persistent); var mrW1 = new fProxyN(N, Allocator.Persistent); var mrW2 = new fProxyN(N, Allocator.Persistent);
                var mrJob = new SpMinresJobFProxy { A = A, b = b, x = xm, y = mrY, r1 = mrR1, r2 = mrR2, v = mrV, w = mrW, w1 = mrW1, w2 = mrW2, K = K, outInfo = oi };
                var mrStat = Bench.Time(() => mrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "MINRES", mrStat, Res(in A, in xm, in b), (int)oi[1], (int)oi[0]));

                fProxyN xc1 = default, cgConvR = default, cgConvP = default, cgConvAp = default;
                fProxyN xc2 = default, pcgConvR = default, pcgConvP = default, pcgConvAp = default, pcgConvZ = default;
                fProxyN xc3 = default, ssorConvR = default, ssorConvP = default, ssorConvAp = default, ssorConvZ = default;
                bool ranConv = N == Ns[Ns.Length - 1];
                if (ranConv)
                {
                    fProxy convTol = Consts.fProxySqrtEps;
                    int convMaxIter = 8 * N;

                    xc1 = new fProxyN(N, Allocator.Persistent);
                    cgConvR = new fProxyN(N, Allocator.Persistent); cgConvP = new fProxyN(N, Allocator.Persistent); cgConvAp = new fProxyN(N, Allocator.Persistent);
                    var cgConvJob = new SpCgJobFProxy { A = A, b = b, x = xc1, r = cgConvR, p = cgConvP, Ap = cgConvAp, K = convMaxIter, tol = convTol, outInfo = oi };
                    var cgConvStat = Bench.Time(() => cgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG@tol", cgConvStat, Res(in A, in xc1, in b), (int)oi[1], (int)oi[0]));

                    xc2 = new fProxyN(N, Allocator.Persistent);
                    pcgConvR = new fProxyN(N, Allocator.Persistent); pcgConvP = new fProxyN(N, Allocator.Persistent); pcgConvAp = new fProxyN(N, Allocator.Persistent); pcgConvZ = new fProxyN(N, Allocator.Persistent);
                    var pcgConvJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xc2, r = pcgConvR, p = pcgConvP, Ap = pcgConvAp, z = pcgConvZ, K = convMaxIter, tol = convTol, outInfo = oi };
                    var pcgConvStat = Bench.Time(() => pcgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi@tol", pcgConvStat, Res(in A, in xc2, in b), (int)oi[1], (int)oi[0]));

                    xc3 = new fProxyN(N, Allocator.Persistent);
                    ssorConvR = new fProxyN(N, Allocator.Persistent); ssorConvP = new fProxyN(N, Allocator.Persistent); ssorConvAp = new fProxyN(N, Allocator.Persistent); ssorConvZ = new fProxyN(N, Allocator.Persistent);
                    var ssorConvJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xc3, r = ssorConvR, p = ssorConvP, Ap = ssorConvAp, z = ssorConvZ, K = convMaxIter, tol = convTol, outInfo = oi };
                    var ssorConvStat = Bench.Time(() => ssorConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR@tol", ssorConvStat, Res(in A, in xc3, in b), (int)oi[1], (int)oi[0]));
                }

                A.Dispose(); M.Dispose(); ssor.Dispose(); xKnown.Dispose(); b.Dispose();
                x.Dispose(); cgR.Dispose(); cgP.Dispose(); cgAp.Dispose();
                xp.Dispose(); pcgR.Dispose(); pcgP.Dispose(); pcgAp.Dispose(); pcgZ.Dispose();
                xs.Dispose(); ssorR.Dispose(); ssorP.Dispose(); ssorAp.Dispose(); ssorZ.Dispose();
                xm.Dispose(); mrY.Dispose(); mrR1.Dispose(); mrR2.Dispose(); mrV.Dispose(); mrW.Dispose(); mrW1.Dispose(); mrW2.Dispose();
                if (ranConv)
                {
                    xc1.Dispose(); cgConvR.Dispose(); cgConvP.Dispose(); cgConvAp.Dispose();
                    xc2.Dispose(); pcgConvR.Dispose(); pcgConvP.Dispose(); pcgConvAp.Dispose(); pcgConvZ.Dispose();
                    xc3.Dispose(); ssorConvR.Dispose(); ssorConvP.Dispose(); ssorConvAp.Dispose(); ssorConvZ.Dispose();
                }
            }

            oi.Dispose();
        }

        static void BenchEigenFProxy(StringBuilder sb, int BR, int[] Ns, fProxy density, int lanczosSteps)
        {
            foreach (int N in Ns)
            {
                int nb = N / BR;
                var A = fProxyGallery.fProxyRandomSparseSPD(nb, BR, density, 0x5A17u, Allocator.Persistent);
                string sz = N.ToString();
                var outInfo = new NativeArray<double>(3, Allocator.Persistent);

                var lws = new fProxyLanczosCache(N, lanczosSteps, Allocator.Persistent);
                var lvals = new fProxyN(lanczosSteps, Allocator.Persistent);
                var lanJob = new SpLanczosJobFProxy { A = A, ws = lws, vals = lvals, steps = lanczosSteps, outInfo = outInfo };
                var lanStat = Bench.Time(() => lanJob.Run());
                sb.AppendLine(LargeSparseFmt.EigRow("fProxy", sz, "Lanczos s=" + lanczosSteps, lanStat, new[] { outInfo[0], outInfo[1], outInfo[2] }));

                outInfo.Dispose();
                A.Dispose(); lws.Dispose(); lvals.Dispose();
            }
        }

        // Rows: none/guard0 -> blockJac/guard0 -> SSOR/guard0 -> blockJac/guardG, showing both
        // levers (precond alone, then precond+guard stacking). Every row carries wall-clock
        // (Bench.Time, 1 warmup + 4 timed) alongside iterations -- see SpLobpcgJobFProxy's comment
        // for why ws.X must be re-zeroed every Execute() for that to be a fair repeated measurement.
        static void BenchLobpcgFProxy(StringBuilder sb, int[] eigGrids, int lobpcgK, int lobpcgGuard, int lobpcgMaxIter)
        {
            foreach (int g in eigGrids)
            {
                int n = g * g;
                var A = fProxyGallery.fProxyLaplacian2D(g, g, Allocator.Persistent);
                var M = new fProxyBlockJacobi(in A, Allocator.Persistent);
                var ssor = new fProxySSOR(in A, Allocator.Persistent);
                string grid = g + "x" + g + "(" + n + ")";
                var oi = new NativeArray<double>(5, Allocator.Persistent);
                fProxy tol = Consts.fProxySqrtEps;

                var noneWs = new fProxyLOBPCGCache(n, lobpcgK, Allocator.Persistent);
                var noneJob = new SpLobpcgJobFProxy { A = A, ws = noneWs, k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var noneStat = Bench.Time(() => noneJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "none", 0, noneStat, LargeSparseFmt.Snap(oi));

                var jacWs = new fProxyLOBPCGCache(n, lobpcgK, Allocator.Persistent);
                var jacJob = new SpLobpcgPrecJobFProxy { A = A, M = M, ws = jacWs, k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var jacStat = Bench.Time(() => jacJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "blockJac", 0, jacStat, LargeSparseFmt.Snap(oi));

                var ssorWs = new fProxyLOBPCGCache(n, lobpcgK, Allocator.Persistent);
                var ssorJob = new SpLobpcgSSORJobFProxy { A = A, M = ssor, ws = ssorWs, k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var ssorStat = Bench.Time(() => ssorJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "SSOR", 0, ssorStat, LargeSparseFmt.Snap(oi));

                var jacGuardWs = new fProxyLOBPCGCache(n, lobpcgK + lobpcgGuard, Allocator.Persistent);
                var jacGuardJob = new SpLobpcgPrecJobFProxy { A = A, M = M, ws = jacGuardWs, k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var jacGuardStat = Bench.Time(() => jacGuardJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "blockJac", lobpcgGuard, jacGuardStat, LargeSparseFmt.Snap(oi));

                oi.Dispose();
                A.Dispose(); M.Dispose(); ssor.Dispose();
                noneWs.Dispose(); jacWs.Dispose(); ssorWs.Dispose(); jacGuardWs.Dispose();
            }
        }
    }
}
