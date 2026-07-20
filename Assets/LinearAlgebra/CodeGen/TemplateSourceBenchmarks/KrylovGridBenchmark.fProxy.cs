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
    // GENERATED per-dtype half of KrylovGridBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (gallery/solver constants, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/KrylovGridBenchmark.cs.
    //
    // Every job zeroes x then calls a solver's fixed-iteration BSR entry point at tol=0 -- every
    // timed sample runs exactly K iterations (deterministic timing, mirrors IterativeBenchmark /
    // SparseSolverBenchmark / PCGBenchmark's convention). fcg is excluded: it has no unpreconditioned
    // BSR entry point (see fProxyFcgInvoker in KrylovBattery.Invokers.fProxy.cs), so a plain-Identity
    // timing here would need an explicit identity preconditioner instance, not a "clean" fixed-iter
    // call like the other nine.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CgGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.minres(in A, in b, ref x, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresQLPGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.minresQLP(in A, in b, ref x, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.biCGStab(in A, in b, ref x, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GmresGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int Restart;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.gmres(in A, in b, ref x, Restart, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FgmresGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int Restart;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.fgmres(in A, in b, ref x, Restart, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IdrGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int S;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.idr(in A, in b, ref x, S, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TfqmrGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.tfqmr(in A, in b, ref x, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GcrodrGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int Restart;
        public int Recycle;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.gcrodr(in A, in b, ref x, Restart, Recycle, K, 0f);
        }
    }

    public static partial class KrylovGridBenchmark
    {
        static double ResidualFProxy(in fProxyBSR A, in fProxyN x, in fProxyN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double d = (double)Ax[i] - (double)b[i];
                num += d * d;
                den += (double)b[i] * (double)b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        static string RowFProxy(string gallery, string solver, int n, Bench.Stat st, double residual)
        {
            const string fmt = "{0,-7} {1,-6} {2,-10} {3,-12} {4,11:F4} {5,11:F4} {6,14:E3}";
            return string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, gallery, solver, st.Median, st.Min, residual);
        }

        // Every square solver applies on an SPD gallery -- times all nine.
        static string BenchSpdFProxy(int restart, int s, int recycle, int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(16, 16);   // same construction as GalleryBSRMatrix.Laplacian2D_16x16
            int n = A.M_Rows;
            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xD100u);
            var x = arena.fProxyVec(n);
            var sb = new StringBuilder();

            var cgJob = new CgGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("SPD", "cg", n, Bench.Time(() => cgJob.Run()), ResidualFProxy(in A, in x, in b)));

            var minresJob = new MinresGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("SPD", "minres", n, Bench.Time(() => minresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var minresQlpJob = new MinresQLPGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("SPD", "minresQLP", n, Bench.Time(() => minresQlpJob.Run()), ResidualFProxy(in A, in x, in b)));

            var biCGStabJob = new BiCGStabGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("SPD", "biCGStab", n, Bench.Time(() => biCGStabJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gmresJob = new GmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k };
            sb.AppendLine(RowFProxy("SPD", "gmres", n, Bench.Time(() => gmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var fgmresJob = new FgmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k };
            sb.AppendLine(RowFProxy("SPD", "fgmres", n, Bench.Time(() => fgmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var idrJob = new IdrGridJobFProxy { A = A, b = b, x = x, S = s, K = k };
            sb.AppendLine(RowFProxy("SPD", "idr", n, Bench.Time(() => idrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var tfqmrJob = new TfqmrGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("SPD", "tfqmr", n, Bench.Time(() => tfqmrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gcrodrJob = new GcrodrGridJobFProxy { A = A, b = b, x = x, Restart = restart, Recycle = recycle, K = k };
            sb.Append(RowFProxy("SPD", "gcrodr", n, Bench.Time(() => gcrodrJob.Run()), ResidualFProxy(in A, in x, in b)));

            arena.Dispose();
            return sb.ToString();
        }

        // Only the general-square solvers apply on a nonsymmetric gallery (cg/minres/minresQLP
        // require/forbid symmetry -- see MatrixProfile.Nonsymmetric in KrylovBatteryProfile.cs).
        static string BenchNonsymFProxy(int restart, int s, int recycle, int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomSparse(80, 80, 1, (fProxy)0.1, 0x5EED1u);   // same construction as GalleryBSRMatrix.RandomSparseNonsym_80
            int n = A.M_Rows;
            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xD101u);
            var x = arena.fProxyVec(n);
            var sb = new StringBuilder();

            var biCGStabJob = new BiCGStabGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("Nonsym", "biCGStab", n, Bench.Time(() => biCGStabJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gmresJob = new GmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k };
            sb.AppendLine(RowFProxy("Nonsym", "gmres", n, Bench.Time(() => gmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var fgmresJob = new FgmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k };
            sb.AppendLine(RowFProxy("Nonsym", "fgmres", n, Bench.Time(() => fgmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var idrJob = new IdrGridJobFProxy { A = A, b = b, x = x, S = s, K = k };
            sb.AppendLine(RowFProxy("Nonsym", "idr", n, Bench.Time(() => idrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var tfqmrJob = new TfqmrGridJobFProxy { A = A, b = b, x = x, K = k };
            sb.AppendLine(RowFProxy("Nonsym", "tfqmr", n, Bench.Time(() => tfqmrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gcrodrJob = new GcrodrGridJobFProxy { A = A, b = b, x = x, Restart = restart, Recycle = recycle, K = k };
            sb.Append(RowFProxy("Nonsym", "gcrodr", n, Bench.Time(() => gcrodrJob.Run()), ResidualFProxy(in A, in x, in b)));

            arena.Dispose();
            return sb.ToString();
        }
    }
}
