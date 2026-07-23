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
    // GENERATED per-dtype half of PCGBenchmark (timed IJobs + system builder + residual + measure).
    // The dtype-agnostic harness (BR/NB/K constants, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/PCGBenchmark.cs.
    //
    // NOTE: the block-tridiagonal builder intentionally draws its noise with rng.NextFloat in BOTH
    // dtypes (not NextFProxy) so the float and double systems are seeded from the identical stream;
    // do not "correct" those to NextFProxy.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgBsrJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyBlockJacobi M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CgBsrJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }


    // Plain CG (no preconditioner) solve-to-tolerance, reporting iteration count -- the baseline the
    // preconditioned rows are measured against.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CgTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x, r, p, Ap;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgSsorTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxySSOR M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgIC0TolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyIC0 M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgJacobiTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyBlockJacobi M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgSchwarzTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyAdditiveSchwarz M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgChebyshevTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyChebyshev M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgFSAITolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyFSAI M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgAMGTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyAMGPreconditioner M;
        public fProxyN b, x, r, p, Ap, z;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FcgAMGKTolJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyAMGPreconditioner M;
        public fProxyN b, x, r, p, Ap, z, rOld;
        public int K;
        public fProxy Tol;
        public Indices Iters;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.fcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, ref rOld, K, Tol);
            Iters[0] = info.iterations;
        }
    }

    public static partial class PCGBenchmark
    {
        // Preconditioner face-off: solve-to-tolerance. Reports wall-clock AND iteration count --
        // a preconditioner must win TIME, not just iterations, to earn its apply cost. NOTE: the
        // Laplacian2D gallery is block-TRIDIAGONAL (the 2D stencil lives inside the blocks), a
        // fill-free pattern where IC(0) is the exact factorization (1 iteration) -- the
        // random-sparse-SPD rows are the genuinely-incomplete case.
        static string BenchPrecondFProxy(int gridX, int gridY)
            => BenchPrecondCoreFProxy(gridX, gridY, 0, 0);

        static string BenchPrecondRandomFProxy(int nb, int bs, float density, uint seed)
            => BenchPrecondCoreFProxy(nb, bs, 1, density, seed);

        static string BenchPrecondScalarPoissonFProxy(int gridX, int gridY)
            => BenchPrecondCoreFProxy(gridX, gridY, 2, 0);

        // Scalar 5-point 2D Poisson (BR=1): unlike the block-tridiagonal gallery Laplacian2D, IC(0)
        // is a GENUINELY incomplete factorization here, so every point-preconditioner's iteration
        // count grows ~O(sqrt(N)) while AMG stays ~flat -- the fair grid-independence comparison.
        static fProxyBSR ScalarPoisson2DFProxy(int gx, int gy)
        {
            int n = gx * gy;
            var bld = new fProxyBSRBuilder(n, n, 1, 1, Allocator.Persistent, 5 * n);
            for (int y = 0; y < gy; y++)
                for (int x = 0; x < gx; x++)
                {
                    int i = y * gx + x;
                    bld.AddValue(i, i, (fProxy)4);
                    if (x > 0) bld.AddValue(i, i - 1, (fProxy)(-1));
                    if (x < gx - 1) bld.AddValue(i, i + 1, (fProxy)(-1));
                    if (y > 0) bld.AddValue(i, i - gx, (fProxy)(-1));
                    if (y < gy - 1) bld.AddValue(i, i + gx, (fProxy)(-1));
                }
            var result = bld.ToBSR(Allocator.Persistent);
            bld.Dispose();
            return result;
        }

        // kind: 0 = block Laplacian2D gallery, 1 = random-sparse SPD, 2 = scalar 5-point Poisson.
        static string BenchPrecondCoreFProxy(int p1, int p2, int kind, float density, uint seed = 0)
        {
            const string fmt = "{0,-7} {1,-6} {2,-12} {3,11:F4} {4,11:F4} {5,7} {6,14:E3}";
            var A = kind == 0 ? fProxyGallery.fProxyLaplacian2D(p1, p2, Allocator.Persistent)
                  : kind == 1 ? fProxyGallery.fProxyRandomSparseSPD(p1, p2, (fProxy)density, seed, Allocator.Persistent)
                              : ScalarPoisson2DFProxy(p1, p2);
            int n = A.M_Rows;
            var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0xC002Du, Allocator.Persistent);
            fProxy tol = Consts.fProxySqrtEps;
            int cap = 8 * n;
            var iters = new Indices(1, Allocator.Persistent);
            var sb = new StringBuilder();

            var x = new fProxyN(n, Allocator.Persistent); var r = new fProxyN(n, Allocator.Persistent); var p = new fProxyN(n, Allocator.Persistent);
            var Ap = new fProxyN(n, Allocator.Persistent); var z = new fProxyN(n, Allocator.Persistent);

            var cgJob = new CgTolJobFProxy { A = A, b = b, x = x, r = r, p = p, Ap = Ap, K = cap, Tol = tol, Iters = iters };
            var cgStat = Bench.Time(() => cgJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "CG(plain)", cgStat.Median, cgStat.Min, iters[0], Residual(in A, in x, in b)));

            var mJ = new fProxyBlockJacobi(in A, Allocator.Persistent);
            var jJob = new PcgJacobiTolJobFProxy { A = A, M = mJ, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var jStat = Bench.Time(() => jJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-Jacobi", jStat.Median, jStat.Min, iters[0], Residual(in A, in x, in b)));

            var mS = new fProxySSOR(in A, Allocator.Persistent);
            var sJob = new PcgSsorTolJobFProxy { A = A, M = mS, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var sStat = Bench.Time(() => sJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-SSOR", sStat.Median, sStat.Min, iters[0], Residual(in A, in x, in b)));

            var mI = new fProxyIC0(in A, Allocator.Persistent);
            var iJob = new PcgIC0TolJobFProxy { A = A, M = mI, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var iStat = Bench.Time(() => iJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-IC0", iStat.Median, iStat.Min, iters[0], Residual(in A, in x, in b)));

            var mW = new fProxyAdditiveSchwarz(in A, Allocator.Persistent);
            var wJob = new PcgSchwarzTolJobFProxy { A = A, M = mW, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var wStat = Bench.Time(() => wJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-Schwarz", wStat.Median, wStat.Min, iters[0], Residual(in A, in x, in b)));

            var mC = new fProxyChebyshev(in A, Allocator.Persistent);
            var cJob = new PcgChebyshevTolJobFProxy { A = A, M = mC, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var cStat = Bench.Time(() => cJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-Cheby", cStat.Median, cStat.Min, iters[0], Residual(in A, in x, in b)));

            var mF = new fProxyFSAI(in A, Allocator.Persistent);
            var fJob = new PcgFSAITolJobFProxy { A = A, M = mF, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var fStat = Bench.Time(() => fJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-FSAI", fStat.Median, fStat.Min, iters[0], Residual(in A, in x, in b)));

            // AMG hierarchy built once outside the timed solve (mirrors the other build-then-solve
            // rows); the solve times one V-cycle-preconditioned CG. Setup cost is not in this number.
            var amgH = new fProxyAMG(in A, out _, Allocator.Persistent);
            var mAmg = new fProxyAMGPreconditioner(in amgH);
            var aJob = new PcgAMGTolJobFProxy { A = A, M = mAmg, b = b, x = x, r = r, p = p, Ap = Ap, z = z, K = cap, Tol = tol, Iters = iters };
            var aStat = Bench.Time(() => aJob.Run());
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-AMG-V", aStat.Median, aStat.Min, iters[0], Residual(in A, in x, in b)));

            // K-cycle AMG: per-level Krylov acceleration, driven by fcg (the K-cycle is a variable
            // operator). Compare its iteration count against PCG-AMG-V above.
            var rOld = new fProxyN(n, Allocator.Persistent);
            var amgK = new fProxyAMG(in A, new AMGOptions { cycle = MGCycle.K, theta = 0, pre = 1, post = 1, coarseMax = 48, maxLevels = 20 }, out _, Allocator.Persistent);
            var mAmgK = new fProxyAMGPreconditioner(in amgK);
            var akJob = new FcgAMGKTolJobFProxy { A = A, M = mAmgK, b = b, x = x, r = r, p = p, Ap = Ap, z = z, rOld = rOld, K = cap, Tol = tol, Iters = iters };
            var akStat = Bench.Time(() => akJob.Run());
            sb.Append(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "FCG-AMG-K", akStat.Median, akStat.Min, iters[0], Residual(in A, in x, in b)));

            amgK.Dispose();
            amgH.Dispose();
            A.Dispose();
            b.Dispose();
            iters.Dispose();
            x.Dispose(); r.Dispose(); p.Dispose(); Ap.Dispose(); z.Dispose();
            rOld.Dispose();
            mJ.Dispose(); mS.Dispose(); mI.Dispose(); mW.Dispose(); mC.Dispose(); mF.Dispose();
            return sb.ToString();
        }

        static void BuildTridiagBlockSPDFProxy(int NB, int BR, out fProxyBSR sparse, out int n)
        {
            n = NB * BR;
            int nnzb = NB + 2 * (NB - 1);
            var builder = new fProxyBSRBuilder(NB, NB, BR, BR, Allocator.Persistent, nnzb);
            var rng = new Random(0x51ED270Bu);

            for (int i = 0; i < NB; i++)
            {
                // Diagonal block must be SYMMETRIC (mirror the noise) or the assembled matrix
                // is not actually SPD and the residual column loses its meaning.
                var Di = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = r; c < BR; c++)
                    {
                        fProxy v = (r == c ? BR * 8f : 0f) + rng.NextFloat(-0.1f, 0.1f);
                        Di[r, c] = v;
                        Di[c, r] = v;
                    }
                builder.AddBlock(i, i, in Di);
                Di.Dispose();

                if (i > 0)
                {
                    var off = new fProxyMxN(BR, BR, Allocator.Persistent);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            off[r, c] = rng.NextFloat(-0.3f, 0.3f);
                    builder.AddBlock(i, i - 1, in off);

                    var offT = new fProxyMxN(BR, BR, Allocator.Persistent);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            offT[r, c] = off[c, r];
                    builder.AddBlock(i - 1, i, in offT);
                    off.Dispose();
                    offT.Dispose();
                }
            }

            sparse = builder.ToBSR(Allocator.Persistent);
            builder.Dispose();
        }

        static double Residual(in fProxyBSR A, in fProxyN x, in fProxyN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double diff = (double)Ax[i] - (double)b[i];
                num += diff * diff;
                den += (double)b[i] * (double)b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
        public struct JacobiBuildJobFProxy : IJob
        {
            public fProxyBSR A;

            public void Execute()
            {
                var M = new fProxyBlockJacobi(in A, Allocator.Temp, out PreconditionerInfo info);
                M.Dispose();
            }
        }

        // Build-cost row: times ONLY the BlockJacobi construction (per-diagonal-block LU inversion)
        // over the block-tridiagonal system — the allocator-traffic-sensitive path.
        static string BenchJacobiBuildFProxy(int BR, int NB)
        {
            const string fmt = "{0,-7} {1,-6} {2,-3} {3,11:F4} {4,11:F4}";
            BuildTridiagBlockSPDFProxy(NB, BR, out var A, out int n);

            var job = new JacobiBuildJobFProxy { A = A };
            var stat = Bench.Time(() => job.Run());

            A.Dispose();
            return string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, BR, stat.Median, stat.Min);
        }

        static string BenchFProxy(int BR, int NB, int K)
        {
            const string fmt = "{0,-7} {1,-6} {2,-12} {3,11:F4} {4,11:F4} {5,14:E3}";
            BuildTridiagBlockSPDFProxy(NB, BR, out var A, out int n);
            var M = new fProxyBlockJacobi(in A, Allocator.Persistent);
            var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0xC001Du, Allocator.Persistent);

            var xCg = new fProxyN(n, Allocator.Persistent); var rCg = new fProxyN(n, Allocator.Persistent); var pCg = new fProxyN(n, Allocator.Persistent); var ApCg = new fProxyN(n, Allocator.Persistent);
            var cgJob = new CgBsrJobFProxy { A = A, b = b, x = xCg, r = rCg, p = pCg, Ap = ApCg, K = K };
            var cgStat = Bench.Time(() => cgJob.Run());
            var sb = new StringBuilder();
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "CG", cgStat.Median, cgStat.Min, Residual(in A, in xCg, in b)));

            var xPcg = new fProxyN(n, Allocator.Persistent); var rPcg = new fProxyN(n, Allocator.Persistent); var pPcg = new fProxyN(n, Allocator.Persistent); var ApPcg = new fProxyN(n, Allocator.Persistent); var zPcg = new fProxyN(n, Allocator.Persistent);
            var pcgJob = new PcgBsrJobFProxy { A = A, M = M, b = b, x = xPcg, r = rPcg, p = pPcg, Ap = ApPcg, z = zPcg, K = K };
            var pcgStat = Bench.Time(() => pcgJob.Run());
            sb.Append(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "PCG-Jacobi", pcgStat.Median, pcgStat.Min, Residual(in A, in xPcg, in b)));

            A.Dispose();
            M.Dispose();
            b.Dispose();
            xCg.Dispose(); rCg.Dispose(); pCg.Dispose(); ApCg.Dispose();
            xPcg.Dispose(); rPcg.Dispose(); pPcg.Dispose(); ApPcg.Dispose(); zPcg.Dispose();
            return sb.ToString();
        }
    }
}
