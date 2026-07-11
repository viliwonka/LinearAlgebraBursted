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

    // Krylov R3: SpCgJobFProxy/SpPcgJobFProxy grew a `tol` field (replacing the hardcoded `0f`
    // argument) so the SAME job types serve both the
    // fixed-K/tol=0 throughput rows below (every existing call site omits `tol`, leaving it at
    // its struct default 0 -- byte-identical behavior to the old hardcoded literal) and the new
    // iterations-to-CONVERGENCE comparison (BenchPrecondConvergenceFProxy), which sets a real
    // tol/maxIter. No new job type needed for that reuse. SpmvJobFProxy (a standalone spMV
    // throughput row, uninformative once CG/PCG rows already show spMV's ~60-95%-of-cost share)
    // was DELETED, not left unused, to pay for the new PCG-SSOR row this round adds -- see
    // BenchKrylovFProxy's own comment.
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCgJobFProxy : IJob { public fProxyBSR A; public fProxyN b, x, r, p, Ap; public int K; public fProxy tol; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, tol); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpPcgJobFProxy : IJob { public fProxyBSR A; public fProxyBlockJacobi M; public fProxyN b, x, r, p, Ap, z; public int K; public fProxy tol; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, tol); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

    // Krylov R3: SSOR twin of SpPcgJobFProxy (same reuse for fixed-K throughput and
    // convergence-comparison rows).
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpPcgSSORJobFProxy : IJob { public fProxyBSR A; public fProxySSOR M; public fProxyN b, x, r, p, Ap, z; public int K; public fProxy tol; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; var info = Krylov.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, tol); outInfo[0] = (int)info.status; outInfo[1] = info.iterations; } }

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
    // Krylov R3b: these jobs are now wired through
    // Bench.Time (1 warmup + 4 timed .Run() calls on the SAME captured `ws`) to add a wall-clock
    // column to the LOBPCG report -- lobpcg only reseeds ws.X when it is all-zero (its own
    // warm-start contract), so re-running the SAME ws without resetting X would warm-start every
    // timed sample from the PREVIOUS sample's converged eigenvectors and measure ~0 iterations
    // from the 2nd run on. Zeroing X at the top of every Execute() forces the SAME deterministic
    // reseed (lobpcg's own fixed-seed 0x9E3779B1u fill) on every sample -- a fair, reproducible,
    // cold-start measurement each time, mirroring the x[i]=0 reset every Krylov job above already does.
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLobpcgJobFProxy : IJob { public fProxyBSR A; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < ws.X.M_Rows; i++) for (int c = 0; c < ws.X.N_Cols; c++) ws.X[i, c] = (fProxy)0; var info = Eigen.lobpcg(in A, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLobpcgPrecJobFProxy : IJob { public fProxyBSR A; public fProxyBlockJacobi M; public fProxyLOBPCGCache ws; public int k; public fProxy tol; public int maxIter; public NativeArray<double> outInfo;
        public void Execute() { for (int i = 0; i < ws.X.M_Rows; i++) for (int c = 0; c < ws.X.N_Cols; c++) ws.X[i, c] = (fProxy)0; var info = Eigen.lobpcg(in A, in M, ref ws, k, tol, maxIter); LobpcgReport.WriteFProxy(in info, in ws, k, outInfo); } }

    // Krylov R3b: SSOR preconditioner axis for LOBPCG (fProxySSOR drops into TPre unchanged,
    // verified by the LobpcgAcceptsSSORPreconditioner test). Hypothesis under test: LOBPCG's
    // per-iteration cost is dominated by Rayleigh-Ritz work, so SSOR's iteration cut might win
    // wall-clock even though its OWN apply is 2-4x block-Jacobi's -- unlike plain PCG, where that
    // apply-cost multiple was decisive. Only
    // fProxyBlockJacobi got a dedicated `lobpcg(in fProxyBSR, in TPre, ...)` overload -- fProxySSOR
    // goes through the generic `lobpcg<TOp,TPre>` core via fProxyBSROperator, same as the
    // LobpcgAcceptsSSORPreconditioner test does.
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
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

        // Krylov R3: the PCG rows grow a preconditioner
        // axis (none/CG, block-Jacobi, SSOR). "none" is CG itself (algebraically PCG with M=I,
        // same recurrence) rather than a redundant literal PCG-identity row. The fixed-K/tol=0
        // rows below measure per-ITERATION wall-clock cost (every solver runs the full K budget,
        // so iters is uninformative there by design); the SEPARATE "@tol" rows at the end of the
        // largest N (BenchPrecondConvergenceFProxy inline below) run to a REAL tolerance and are
        // where SSOR's iteration-count win is actually visible -- see that block's own comment.
        // spMV x50 (a standalone throughput row, uninformative once CG/PCG already show spMV's
        // ~60-95%-of-cost share -- spec's own number) was DELETED to pay for the new PCG-SSOR row
        // (Q7 budget ruling: cut redundancy, don't grow the report unboundedly).
        static void BenchKrylovFProxy(StringBuilder sb, int BR, int[] Ns, fProxy density, int K)
        {
            var oi = new NativeArray<double>(2, Allocator.Persistent);

            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.fProxyRandomSparseSPD(nb, BR, density, 0x5A17u);
                var M = arena.fProxyBlockJacobi(in A);
                var ssor = arena.fProxySSOR(in A);
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
                var xs = arena.fProxyVec(N);
                var ssorJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xs, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = K, outInfo = oi };
                var ssorStat = Bench.Time(() => ssorJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR", ssorStat, Res(in A, in xs, in b), (int)oi[1], (int)oi[0]));

                // Iterations-to-CONVERGENCE (real tol, generous maxIter) -- ONE size only
                // (largest N: the trend is visible at one size; budget discipline). This is the
                // row set where SSOR's iteration-count win is visible; the fixed-K/tol=0 rows
                // above cannot show it (every solver there runs the full K by construction).
                if (N == Ns[Ns.Length - 1])
                {
                    fProxy convTol = Consts.fProxySqrtEps;
                    int convMaxIter = 8 * N;

                    var xc1 = arena.fProxyVec(N);
                    var cgConvJob = new SpCgJobFProxy { A = A, b = b, x = xc1, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), K = convMaxIter, tol = convTol, outInfo = oi };
                    var cgConvStat = Bench.Time(() => cgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG@tol", cgConvStat, Res(in A, in xc1, in b), (int)oi[1], (int)oi[0]));

                    var xc2 = arena.fProxyVec(N);
                    var pcgConvJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xc2, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = convMaxIter, tol = convTol, outInfo = oi };
                    var pcgConvStat = Bench.Time(() => pcgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi@tol", pcgConvStat, Res(in A, in xc2, in b), (int)oi[1], (int)oi[0]));

                    var xc3 = arena.fProxyVec(N);
                    var ssorConvJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xc3, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = convMaxIter, tol = convTol, outInfo = oi };
                    var ssorConvStat = Bench.Time(() => ssorConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR@tol", ssorConvStat, Res(in A, in xc3, in b), (int)oi[1], (int)oi[0]));
                }

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
        // Only the SPD-compatible solvers run here (CG/PCG-Jacobi/PCG-SSOR/MINRES) -- BiCGStab
        // needs a non-symmetric operator and CGLS/LSQR/LSMR need a rectangular one, neither of
        // which this generator produces.
        //
        // Krylov R3: gained PCG-SSOR (fixed-K row + the "@tol" convergence-comparison rows at the
        // largest N, same reasoning as BenchKrylovFProxy's own comment) -- PAID FOR by dropping
        // N=5120 from this section's Ns (caller now passes a single-element array): net row count
        // for the fixed-K table goes from 3 solvers x 2 Ns to 4 solvers x 1 N, i.e. DOWN despite
        // adding a whole new preconditioner (Q7 budget ruling).
        static void BenchStencilFProxy(StringBuilder sb, int[] Ns, int K)
        {
            var oi = new NativeArray<double>(2, Allocator.Persistent);

            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyLaplacian2D(1, N);
                var M = arena.fProxyBlockJacobi(in A);
                var ssor = arena.fProxySSOR(in A);
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
                var xs = arena.fProxyVec(N);
                var ssorJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xs, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = K, outInfo = oi };
                var ssorStat = Bench.Time(() => ssorJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR", ssorStat, Res(in A, in xs, in b), (int)oi[1], (int)oi[0]));
                var xm = arena.fProxyVec(N);
                var mrJob = new SpMinresJobFProxy { A = A, b = b, x = xm, y = arena.fProxyVec(N), r1 = arena.fProxyVec(N), r2 = arena.fProxyVec(N), v = arena.fProxyVec(N), w = arena.fProxyVec(N), w1 = arena.fProxyVec(N), w2 = arena.fProxyVec(N), K = K, outInfo = oi };
                var mrStat = Bench.Time(() => mrJob.Run());
                sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "MINRES", mrStat, Res(in A, in xm, in b), (int)oi[1], (int)oi[0]));

                if (N == Ns[Ns.Length - 1])
                {
                    fProxy convTol = Consts.fProxySqrtEps;
                    int convMaxIter = 8 * N;

                    var xc1 = arena.fProxyVec(N);
                    var cgConvJob = new SpCgJobFProxy { A = A, b = b, x = xc1, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), K = convMaxIter, tol = convTol, outInfo = oi };
                    var cgConvStat = Bench.Time(() => cgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "CG@tol", cgConvStat, Res(in A, in xc1, in b), (int)oi[1], (int)oi[0]));

                    var xc2 = arena.fProxyVec(N);
                    var pcgConvJob = new SpPcgJobFProxy { A = A, M = M, b = b, x = xc2, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = convMaxIter, tol = convTol, outInfo = oi };
                    var pcgConvStat = Bench.Time(() => pcgConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-Jacobi@tol", pcgConvStat, Res(in A, in xc2, in b), (int)oi[1], (int)oi[0]));

                    var xc3 = arena.fProxyVec(N);
                    var ssorConvJob = new SpPcgSSORJobFProxy { A = A, M = ssor, b = b, x = xc3, r = arena.fProxyVec(N), p = arena.fProxyVec(N), Ap = arena.fProxyVec(N), z = arena.fProxyVec(N), K = convMaxIter, tol = convTol, outInfo = oi };
                    var ssorConvStat = Bench.Time(() => ssorConvJob.Run());
                    sb.AppendLine(LargeSparseFmt.Row("fProxy", sz, "PCG-SSOR@tol", ssorConvStat, Res(in A, in xc3, in b), (int)oi[1], (int)oi[0]));
                }

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

        // Krylov R3b budget trade (spec §3b, disclosed): the "none"+guard row is DROPPED here to
        // pay for the new "SSOR"+guard=0 row -- row count per grid/dtype stays at 4. The dropped
        // combination isn't lost information: none/guard0 -> blockJac/guard0 -> blockJac/guardG
        // still shows both levers (precond alone, then precond+guard stacking); "guard alone" (the
        // dropped point) was never this round's question. Every row now also carries wall-clock
        // (Bench.Time, 1 warmup + 4 timed) alongside iterations -- see SpLobpcgJobFProxy's comment
        // for why ws.X must be re-zeroed every Execute() for that to be a fair repeated measurement.
        static void BenchLobpcgFProxy(StringBuilder sb, int[] eigGrids, int lobpcgK, int lobpcgGuard, int lobpcgMaxIter)
        {
            foreach (int g in eigGrids)
            {
                var arena = new Arena(Allocator.Persistent);
                int n = g * g;
                var A = arena.fProxyLaplacian2D(g, g);
                var M = arena.fProxyBlockJacobi(in A);
                var ssor = arena.fProxySSOR(in A);
                string grid = g + "x" + g + "(" + n + ")";
                var oi = new NativeArray<double>(5, Allocator.Persistent);
                fProxy tol = Consts.fProxySqrtEps;

                var noneJob = new SpLobpcgJobFProxy { A = A, ws = arena.fProxyLOBPCGCache(n, lobpcgK), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var noneStat = Bench.Time(() => noneJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "none", 0, noneStat, LargeSparseFmt.Snap(oi));

                var jacJob = new SpLobpcgPrecJobFProxy { A = A, M = M, ws = arena.fProxyLOBPCGCache(n, lobpcgK), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var jacStat = Bench.Time(() => jacJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "blockJac", 0, jacStat, LargeSparseFmt.Snap(oi));

                var ssorJob = new SpLobpcgSSORJobFProxy { A = A, M = ssor, ws = arena.fProxyLOBPCGCache(n, lobpcgK), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var ssorStat = Bench.Time(() => ssorJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "SSOR", 0, ssorStat, LargeSparseFmt.Snap(oi));

                var jacGuardJob = new SpLobpcgPrecJobFProxy { A = A, M = M, ws = arena.fProxyLOBPCGCache(n, lobpcgK + lobpcgGuard), k = lobpcgK, tol = tol, maxIter = lobpcgMaxIter, outInfo = oi };
                var jacGuardStat = Bench.Time(() => jacGuardJob.Run());
                LargeSparseFmt.LobRow(sb, "fProxy", grid, "blockJac", lobpcgGuard, jacGuardStat, LargeSparseFmt.Snap(oi));

                oi.Dispose();
                arena.Dispose();
            }
        }
    }
}
