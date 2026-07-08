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
    // LARGE sparse solvers at the scale where they are the only option: N up to 10240 at ~1.5% block
    // fill, built via the sparse gallery (fProxyRandomSparseSPD / fProxyRandomSparse) with NO dense twin
    // (a dense 10240x10240 float matrix is ~420 MB and its O(N^3) factor/eig is minutes -- this is exactly
    // the regime where the direct dense solvers are not an option and iterative sparse solvers earn their
    // keep). Every Krylov timing is a FIXED K iterations at tol=0 (deterministic timing; the residual
    // column shows how converged, not just how fast). All workspace is pre-allocated ONCE and reused
    // across timed samples (no per-sample arena growth); every solve runs inside a [BurstCompile] IJob.
    //   1. spMV throughput; 2. square SPD (cg/pcg/minres); 3. square non-symmetric (biCGStab);
    //   4. tall rectangular least-squares m=2n (cgls/lsqr/lsmr); 5. Lanczos (throughput).
    // NOTE on eigensolvers: the random SPD generator is DIAGONALLY DOMINANT -- great for Krylov (fast,
    // well-conditioned) but its spectrum is CLUSTERED, so the k smallest eigenpairs are near-degenerate.
    // Lanczos still measures matvec throughput honestly (it runs a fixed number of steps regardless), but
    // LOBPCG's smallest-k iteration breaks down on a clustered spectrum, so it is deliberately NOT
    // benchmarked here -- a meaningful smallest-eigenpair bench needs a SPREAD-spectrum sparse matrix
    // (e.g. a sparse Laplacian / 2D-grid gallery entry), which is future work.

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpmvJobFloat : IJob { public floatBSR A; public floatN x, y; public int reps;
        public void Execute() { for (int k = 0; k < reps; k++) { if ((k & 1) == 0) BSR.spMV(in A, in x, ref y); else BSR.spMV(in A, in y, ref x); } } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpmvJobDouble : IJob { public doubleBSR A; public doubleN x, y; public int reps;
        public void Execute() { for (int k = 0; k < reps; k++) { if ((k & 1) == 0) BSR.spMV(in A, in x, ref y); else BSR.spMV(in A, in y, ref x); } } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCgJobFloat : IJob { public floatBSR A; public floatN b, x, r, p, Ap; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCgJobDouble : IJob { public doubleBSR A; public doubleN b, x, r, p, Ap; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0.0); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpPcgJobFloat : IJob { public floatBSR A; public floatBlockJacobi M; public floatN b, x, r, p, Ap, z; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpPcgJobDouble : IJob { public doubleBSR A; public doubleBlockJacobi M; public doubleN b, x, r, p, Ap, z; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0.0); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpMinresJobFloat : IJob { public floatBSR A; public floatN b, x, y, r1, r2, v, w, w1, w2; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpMinresJobDouble : IJob { public doubleBSR A; public doubleN b, x, y, r1, r2, v, w, w1, w2; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0.0); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpBicgJobFloat : IJob { public floatBSR A; public floatN b, x, r, rHat0, p, v, t; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpBicgJobDouble : IJob { public doubleBSR A; public doubleN b, x, r, rHat0, p, v, t; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0.0); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCglsJobFloat : IJob { public floatBSR A; public floatN b, x, r, s, p, q; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpCglsJobDouble : IJob { public doubleBSR A; public doubleN b, x, r, s, p, q; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0.0); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLsqrJobFloat : IJob { public floatBSR A; public floatN b, x, u, v, w, tmpM, tmpN; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLsqrJobDouble : IJob { public doubleBSR A; public doubleN b, x, u, v, w, tmpM, tmpN; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0.0); } }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLsmrJobFloat : IJob { public floatBSR A; public floatN b, x, u, v, h, hbar, tmpM, tmpN; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0f; Krylov.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, K, 0f); } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLsmrJobDouble : IJob { public doubleBSR A; public doubleN b, x, u, v, h, hbar, tmpM, tmpN; public int K;
        public void Execute() { for (int i = 0; i < x.N; i++) x[i] = 0.0; Krylov.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, K, 0.0); } }

    // Eigen jobs write [iterations, converged, residual] into outInfo (reference-backed, visible after Run).
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLanczosJobFloat : IJob { public floatBSR A; public floatLanczosCache ws; public floatN vals; public int steps; public NativeArray<double> outInfo;
        public void Execute() { var info = Eigen.lanczos(in A, ref ws, ref vals, steps); outInfo[0] = info.produced; outInfo[1] = info.Solved ? 1 : 0; outInfo[2] = 0; } }
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Default)]
    public struct SpLanczosJobDouble : IJob { public doubleBSR A; public doubleLanczosCache ws; public doubleN vals; public int steps; public NativeArray<double> outInfo;
        public void Execute() { var info = Eigen.lanczos(in A, ref ws, ref vals, steps); outInfo[0] = info.produced; outInfo[1] = info.Solved ? 1 : 0; outInfo[2] = 0; } }

    public static class LargeSparseBenchmark
    {
        const int BR = 4;
        static readonly int[] Ns = { 2048, 5120, 10240 };
        const float Density = 0.015f;
        const int K = 40;
        const int SpmvReps = 50;
        const int LanczosSteps = 32;

        public static void Run() => Bench.WriteReport("benchmark-largesparse.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== LARGE sparse solvers (BSR, ~1.5% block fill, b=4), N up to 10240 -- no dense form ===");
            sb.AppendLine("Krylov rows: K=" + K + " fixed iterations, tol=0 (deterministic timing); residual = ||Ax-b||/||b|| after K.");
            sb.AppendLine("At N=10240 a dense matrix is ~420 MB (float) and O(N^3) dense factor/eig is minutes -- these solvers");
            sb.AppendLine("touch only the ~1.5% nonzero blocks, so this scale is exactly where dense is not an option.");
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-12} {3,11} {4,11} {5,14}", "dtype", "size", "solver", "med(ms)", "min(ms)", "residual"));
            BenchKrylovFloat(sb);
            BenchKrylovDouble(sb);
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-12} {3,11} {4,11} {5,8} {6,10} {7,14}", "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "converged", "maxResid"));
            BenchEigenFloat(sb);
            BenchEigenDouble(sb);
            sb.AppendLine();
        }

        static string Row(string dtype, string size, string solver, Bench.Stat st, double residual) =>
            string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-12} {3,11:F4} {4,11:F4} {5,14:E3}", dtype, size, solver, st.Median, st.Min, residual);
        static string EigRow(string dtype, string size, string solver, Bench.Stat st, double[] info) =>
            string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-12} {3,11:F4} {4,11:F4} {5,8} {6,10} {7,14:E3}", dtype, size, solver, st.Median, st.Min, (int)info[0], (int)info[1], info[2]);

        static void BenchKrylovFloat(StringBuilder sb)
        {
            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.floatRandomSparseSPD(nb, BR, Density, 0x5A17u);
                var M = arena.floatBlockJacobi(in A);
                var xKnown = arena.floatRandomVec(N, 0.5f, 1.5f, 0xB0Bu);
                var b = arena.floatVec(N); BSR.spMV(in A, in xKnown, ref b);
                string sz = N.ToString();

                var sx = arena.floatRandomVec(N, -1f, 1f, 7u); var sy = arena.floatVec(N);
                var spmvJob = new SpmvJobFloat { A = A, x = sx, y = sy, reps = SpmvReps };
                sb.AppendLine(Row("float", sz, "spMV x" + SpmvReps, Bench.Time(() => spmvJob.Run()), 0.0));

                var x = arena.floatVec(N);
                var cgJob = new SpCgJobFloat { A = A, b = b, x = x, r = arena.floatVec(N), p = arena.floatVec(N), Ap = arena.floatVec(N), K = K };
                sb.AppendLine(Row("float", sz, "CG", Bench.Time(() => cgJob.Run()), Res(in A, in x, in b)));
                var xp = arena.floatVec(N);
                var pcgJob = new SpPcgJobFloat { A = A, M = M, b = b, x = xp, r = arena.floatVec(N), p = arena.floatVec(N), Ap = arena.floatVec(N), z = arena.floatVec(N), K = K };
                sb.AppendLine(Row("float", sz, "PCG-Jacobi", Bench.Time(() => pcgJob.Run()), Res(in A, in xp, in b)));
                var xm = arena.floatVec(N);
                var mrJob = new SpMinresJobFloat { A = A, b = b, x = xm, y = arena.floatVec(N), r1 = arena.floatVec(N), r2 = arena.floatVec(N), v = arena.floatVec(N), w = arena.floatVec(N), w1 = arena.floatVec(N), w2 = arena.floatVec(N), K = K };
                sb.AppendLine(Row("float", sz, "MINRES", Bench.Time(() => mrJob.Run()), Res(in A, in xm, in b)));

                var An = arena.floatRandomSparse(nb, nb, BR, Density, 0x1234u);
                var bn = arena.floatVec(N); BSR.spMV(in An, in xKnown, ref bn);
                var xn = arena.floatVec(N);
                var bicgJob = new SpBicgJobFloat { A = An, b = bn, x = xn, r = arena.floatVec(N), rHat0 = arena.floatVec(N), p = arena.floatVec(N), v = arena.floatVec(N), t = arena.floatVec(N), K = K };
                sb.AppendLine(Row("float", sz, "BiCGStab", Bench.Time(() => bicgJob.Run()), Res(in An, in xn, in bn)));

                int mb = 2 * nb, m = mb * BR;
                var At = arena.floatRandomSparse(mb, nb, BR, Density, 0xC0DEu);
                var bt = arena.floatVec(m); BSR.spMV(in At, in xKnown, ref bt);
                string rsz = m + "x" + N;
                var xc = arena.floatVec(N);
                var cglsJob = new SpCglsJobFloat { A = At, b = bt, x = xc, r = arena.floatVec(m), s = arena.floatVec(N), p = arena.floatVec(N), q = arena.floatVec(m), K = K };
                sb.AppendLine(Row("float", rsz, "CGLS", Bench.Time(() => cglsJob.Run()), Res(in At, in xc, in bt)));
                var xl = arena.floatVec(N);
                var lsqrJob = new SpLsqrJobFloat { A = At, b = bt, x = xl, u = arena.floatVec(m), v = arena.floatVec(N), w = arena.floatVec(N), tmpM = arena.floatVec(m), tmpN = arena.floatVec(N), K = K };
                sb.AppendLine(Row("float", rsz, "LSQR", Bench.Time(() => lsqrJob.Run()), Res(in At, in xl, in bt)));
                var xr = arena.floatVec(N);
                var lsmrJob = new SpLsmrJobFloat { A = At, b = bt, x = xr, u = arena.floatVec(m), v = arena.floatVec(N), h = arena.floatVec(N), hbar = arena.floatVec(N), tmpM = arena.floatVec(m), tmpN = arena.floatVec(N), K = K };
                sb.AppendLine(Row("float", rsz, "LSMR", Bench.Time(() => lsmrJob.Run()), Res(in At, in xr, in bt)));

                arena.Dispose();
            }
        }

        static void BenchKrylovDouble(StringBuilder sb)
        {
            sb.AppendLine();
            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.doubleRandomSparseSPD(nb, BR, Density, 0x5A17u);
                var M = arena.doubleBlockJacobi(in A);
                var xKnown = arena.doubleRandomVec(N, 0.5, 1.5, 0xB0Bu);
                var b = arena.doubleVec(N); BSR.spMV(in A, in xKnown, ref b);
                string sz = N.ToString();

                var sx = arena.doubleRandomVec(N, -1.0, 1.0, 7u); var sy = arena.doubleVec(N);
                var spmvJob = new SpmvJobDouble { A = A, x = sx, y = sy, reps = SpmvReps };
                sb.AppendLine(Row("double", sz, "spMV x" + SpmvReps, Bench.Time(() => spmvJob.Run()), 0.0));

                var x = arena.doubleVec(N);
                var cgJob = new SpCgJobDouble { A = A, b = b, x = x, r = arena.doubleVec(N), p = arena.doubleVec(N), Ap = arena.doubleVec(N), K = K };
                sb.AppendLine(Row("double", sz, "CG", Bench.Time(() => cgJob.Run()), Res(in A, in x, in b)));
                var xp = arena.doubleVec(N);
                var pcgJob = new SpPcgJobDouble { A = A, M = M, b = b, x = xp, r = arena.doubleVec(N), p = arena.doubleVec(N), Ap = arena.doubleVec(N), z = arena.doubleVec(N), K = K };
                sb.AppendLine(Row("double", sz, "PCG-Jacobi", Bench.Time(() => pcgJob.Run()), Res(in A, in xp, in b)));
                var xm = arena.doubleVec(N);
                var mrJob = new SpMinresJobDouble { A = A, b = b, x = xm, y = arena.doubleVec(N), r1 = arena.doubleVec(N), r2 = arena.doubleVec(N), v = arena.doubleVec(N), w = arena.doubleVec(N), w1 = arena.doubleVec(N), w2 = arena.doubleVec(N), K = K };
                sb.AppendLine(Row("double", sz, "MINRES", Bench.Time(() => mrJob.Run()), Res(in A, in xm, in b)));

                var An = arena.doubleRandomSparse(nb, nb, BR, Density, 0x1234u);
                var bn = arena.doubleVec(N); BSR.spMV(in An, in xKnown, ref bn);
                var xn = arena.doubleVec(N);
                var bicgJob = new SpBicgJobDouble { A = An, b = bn, x = xn, r = arena.doubleVec(N), rHat0 = arena.doubleVec(N), p = arena.doubleVec(N), v = arena.doubleVec(N), t = arena.doubleVec(N), K = K };
                sb.AppendLine(Row("double", sz, "BiCGStab", Bench.Time(() => bicgJob.Run()), Res(in An, in xn, in bn)));

                int mb = 2 * nb, m = mb * BR;
                var At = arena.doubleRandomSparse(mb, nb, BR, Density, 0xC0DEu);
                var bt = arena.doubleVec(m); BSR.spMV(in At, in xKnown, ref bt);
                string rsz = m + "x" + N;
                var xc = arena.doubleVec(N);
                var cglsJob = new SpCglsJobDouble { A = At, b = bt, x = xc, r = arena.doubleVec(m), s = arena.doubleVec(N), p = arena.doubleVec(N), q = arena.doubleVec(m), K = K };
                sb.AppendLine(Row("double", rsz, "CGLS", Bench.Time(() => cglsJob.Run()), Res(in At, in xc, in bt)));
                var xl = arena.doubleVec(N);
                var lsqrJob = new SpLsqrJobDouble { A = At, b = bt, x = xl, u = arena.doubleVec(m), v = arena.doubleVec(N), w = arena.doubleVec(N), tmpM = arena.doubleVec(m), tmpN = arena.doubleVec(N), K = K };
                sb.AppendLine(Row("double", rsz, "LSQR", Bench.Time(() => lsqrJob.Run()), Res(in At, in xl, in bt)));
                var xr = arena.doubleVec(N);
                var lsmrJob = new SpLsmrJobDouble { A = At, b = bt, x = xr, u = arena.doubleVec(m), v = arena.doubleVec(N), h = arena.doubleVec(N), hbar = arena.doubleVec(N), tmpM = arena.doubleVec(m), tmpN = arena.doubleVec(N), K = K };
                sb.AppendLine(Row("double", rsz, "LSMR", Bench.Time(() => lsmrJob.Run()), Res(in At, in xr, in bt)));

                arena.Dispose();
            }
        }

        static void BenchEigenFloat(StringBuilder sb)
        {
            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.floatRandomSparseSPD(nb, BR, Density, 0x5A17u);
                string sz = N.ToString();
                var outInfo = new NativeArray<double>(3, Allocator.Persistent);

                var lws = arena.floatLanczosCache(N, LanczosSteps);
                var lvals = arena.floatVec(LanczosSteps);
                var lanJob = new SpLanczosJobFloat { A = A, ws = lws, vals = lvals, steps = LanczosSteps, outInfo = outInfo };
                var lanStat = Bench.Time(() => lanJob.Run());
                sb.AppendLine(EigRow("float", sz, "Lanczos s=" + LanczosSteps, lanStat, new[] { outInfo[0], outInfo[1], outInfo[2] }));

                outInfo.Dispose();
                arena.Dispose();
            }
        }

        static void BenchEigenDouble(StringBuilder sb)
        {
            sb.AppendLine();
            foreach (int N in Ns)
            {
                var arena = new Arena(Allocator.Persistent);
                int nb = N / BR;
                var A = arena.doubleRandomSparseSPD(nb, BR, Density, 0x5A17u);
                string sz = N.ToString();
                var outInfo = new NativeArray<double>(3, Allocator.Persistent);

                var lws = arena.doubleLanczosCache(N, LanczosSteps);
                var lvals = arena.doubleVec(LanczosSteps);
                var lanJob = new SpLanczosJobDouble { A = A, ws = lws, vals = lvals, steps = LanczosSteps, outInfo = outInfo };
                var lanStat = Bench.Time(() => lanJob.Run());
                sb.AppendLine(EigRow("double", sz, "Lanczos s=" + LanczosSteps, lanStat, new[] { outInfo[0], outInfo[1], outInfo[2] }));

                outInfo.Dispose();
                arena.Dispose();
            }
        }

        static double Res(in floatBSR A, in floatN x, in floatN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { double d = (double)Ax[i] - (double)b[i]; num += d * d; den += (double)b[i] * (double)b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }
        static double Res(in doubleBSR A, in doubleN x, in doubleN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { double d = Ax[i] - b[i]; num += d * d; den += b[i] * b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }
    }
}
