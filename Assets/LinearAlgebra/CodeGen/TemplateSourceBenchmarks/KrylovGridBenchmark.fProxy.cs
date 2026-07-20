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
    // Two regimes over the same jobs:
    //   FIXED-K   -- Tol = 0, so every timed sample runs exactly K iterations (deterministic
    //                per-iteration cost, mirrors IterativeBenchmark/SparseSolverBenchmark/PCGBenchmark).
    //   CONVERGE  -- Tol = sqrt(eps), K = generous cap, so each solver runs to its own stopping test;
    //                reports iterations-to-converge + status + time-to-solution (the comparison that
    //                ranks solvers -- fewer/cheaper iterations, not a forced fixed count).
    // Each job writes info.iterations/info.status into Out (when created) for the converge regime.
    // fcg is excluded: it has no unpreconditioned BSR entry point (see fProxyFcgInvoker in
    // KrylovBattery.Invokers.fProxy.cs), so a plain-Identity call here would need an explicit identity
    // preconditioner instance, not a "clean" call like the other nine.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CgGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.cg(in A, in b, ref x, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.minres(in A, in b, ref x, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresQLPGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.minresQLP(in A, in b, ref x, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.biCGStab(in A, in b, ref x, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct GmresGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int Restart;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.gmres(in A, in b, ref x, Restart, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FgmresGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int Restart;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.fgmres(in A, in b, ref x, Restart, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IdrGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int S;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.idr(in A, in b, ref x, S, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TfqmrGridJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public int K;
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.tfqmr(in A, in b, ref x, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
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
        public fProxy Tol;
        public NativeArray<int> Out;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            var info = Krylov.gcrodr(in A, in b, ref x, Restart, Recycle, K, Tol);
            if (Out.IsCreated) { Out[0] = info.iterations; Out[1] = (int)info.status; }
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

        // Converge-regime row: iterations-to-converge + status alongside the time and true residual.
        static string RowConvFProxy(string gallery, string solver, int n, int iters, string status, Bench.Stat st, double residual)
        {
            const string fmt = "{0,-7} {1,-6} {2,-10} {3,-12} {4,7} {5,-13} {6,11:F4} {7,11:F4} {8,14:E3}";
            return string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, gallery, solver, iters, status, st.Median, st.Min, residual);
        }

        // fProxy-tokened purely so codegen renames it per dtype (StatusNameFloat/StatusNameDouble),
        // avoiding a duplicate-member clash the way RowFProxy does -- the body is dtype-agnostic.
        static string StatusNameFProxy(int code) => ((IterativeSolveStatus)code).ToString();

        // Every square solver applies on an SPD gallery -- times all nine.
        static string BenchSpdFProxy(int restart, int s, int recycle, int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(16, 16);   // same construction as GalleryBSRMatrix.Laplacian2D_16x16
            int n = A.M_Rows;
            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xD100u);
            var x = arena.fProxyVec(n);
            var o = new NativeArray<int>(2, Allocator.Persistent);   // required construction; unread in fixed-K
            var sb = new StringBuilder();

            var cgJob = new CgGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "cg", n, Bench.Time(() => cgJob.Run()), ResidualFProxy(in A, in x, in b)));

            var minresJob = new MinresGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "minres", n, Bench.Time(() => minresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var minresQlpJob = new MinresQLPGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "minresQLP", n, Bench.Time(() => minresQlpJob.Run()), ResidualFProxy(in A, in x, in b)));

            var biCGStabJob = new BiCGStabGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "biCGStab", n, Bench.Time(() => biCGStabJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gmresJob = new GmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "gmres", n, Bench.Time(() => gmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var fgmresJob = new FgmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "fgmres", n, Bench.Time(() => fgmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var idrJob = new IdrGridJobFProxy { A = A, b = b, x = x, S = s, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "idr", n, Bench.Time(() => idrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var tfqmrJob = new TfqmrGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("SPD", "tfqmr", n, Bench.Time(() => tfqmrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gcrodrJob = new GcrodrGridJobFProxy { A = A, b = b, x = x, Restart = restart, Recycle = recycle, K = k, Tol = (fProxy)0, Out = o };
            sb.Append(RowFProxy("SPD", "gcrodr", n, Bench.Time(() => gcrodrJob.Run()), ResidualFProxy(in A, in x, in b)));

            o.Dispose();
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
            var o = new NativeArray<int>(2, Allocator.Persistent);   // required construction; unread in fixed-K
            var sb = new StringBuilder();

            var biCGStabJob = new BiCGStabGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("Nonsym", "biCGStab", n, Bench.Time(() => biCGStabJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gmresJob = new GmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("Nonsym", "gmres", n, Bench.Time(() => gmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var fgmresJob = new FgmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("Nonsym", "fgmres", n, Bench.Time(() => fgmresJob.Run()), ResidualFProxy(in A, in x, in b)));

            var idrJob = new IdrGridJobFProxy { A = A, b = b, x = x, S = s, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("Nonsym", "idr", n, Bench.Time(() => idrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var tfqmrJob = new TfqmrGridJobFProxy { A = A, b = b, x = x, K = k, Tol = (fProxy)0, Out = o };
            sb.AppendLine(RowFProxy("Nonsym", "tfqmr", n, Bench.Time(() => tfqmrJob.Run()), ResidualFProxy(in A, in x, in b)));

            var gcrodrJob = new GcrodrGridJobFProxy { A = A, b = b, x = x, Restart = restart, Recycle = recycle, K = k, Tol = (fProxy)0, Out = o };
            sb.Append(RowFProxy("Nonsym", "gcrodr", n, Bench.Time(() => gcrodrJob.Run()), ResidualFProxy(in A, in x, in b)));

            o.Dispose();
            arena.Dispose();
            return sb.ToString();
        }

        // CONVERGE regime, SPD gallery: run each of the nine to its own stopping test (real tol,
        // generous cap) and report iterations + status + time-to-solution.
        static string BenchSpdConvergeFProxy(int restart, int s, int recycle)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(16, 16);
            int n = A.M_Rows;
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;
            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xD100u);
            var x = arena.fProxyVec(n);
            var o = new NativeArray<int>(2, Allocator.Persistent);
            var sb = new StringBuilder();

            var cgJob = new CgGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            var st = Bench.Time(() => cgJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "cg", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var minresJob = new MinresGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => minresJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "minres", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var minresQlpJob = new MinresQLPGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => minresQlpJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "minresQLP", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var biCGStabJob = new BiCGStabGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => biCGStabJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "biCGStab", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var gmresJob = new GmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => gmresJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "gmres", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var fgmresJob = new FgmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => fgmresJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "fgmres", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var idrJob = new IdrGridJobFProxy { A = A, b = b, x = x, S = s, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => idrJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "idr", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var tfqmrJob = new TfqmrGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => tfqmrJob.Run());
            sb.AppendLine(RowConvFProxy("SPD", "tfqmr", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var gcrodrJob = new GcrodrGridJobFProxy { A = A, b = b, x = x, Restart = restart, Recycle = recycle, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => gcrodrJob.Run());
            sb.Append(RowConvFProxy("SPD", "gcrodr", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            o.Dispose();
            arena.Dispose();
            return sb.ToString();
        }

        // CONVERGE regime, nonsymmetric gallery: only the general-square solvers apply.
        static string BenchNonsymConvergeFProxy(int restart, int s, int recycle)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomSparse(80, 80, 1, (fProxy)0.1, 0x5EED1u);
            int n = A.M_Rows;
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;
            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xD101u);
            var x = arena.fProxyVec(n);
            var o = new NativeArray<int>(2, Allocator.Persistent);
            var sb = new StringBuilder();

            var biCGStabJob = new BiCGStabGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            var st = Bench.Time(() => biCGStabJob.Run());
            sb.AppendLine(RowConvFProxy("Nonsym", "biCGStab", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var gmresJob = new GmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => gmresJob.Run());
            sb.AppendLine(RowConvFProxy("Nonsym", "gmres", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var fgmresJob = new FgmresGridJobFProxy { A = A, b = b, x = x, Restart = restart, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => fgmresJob.Run());
            sb.AppendLine(RowConvFProxy("Nonsym", "fgmres", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var idrJob = new IdrGridJobFProxy { A = A, b = b, x = x, S = s, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => idrJob.Run());
            sb.AppendLine(RowConvFProxy("Nonsym", "idr", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var tfqmrJob = new TfqmrGridJobFProxy { A = A, b = b, x = x, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => tfqmrJob.Run());
            sb.AppendLine(RowConvFProxy("Nonsym", "tfqmr", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            var gcrodrJob = new GcrodrGridJobFProxy { A = A, b = b, x = x, Restart = restart, Recycle = recycle, K = maxIter, Tol = tol, Out = o };
            st = Bench.Time(() => gcrodrJob.Run());
            sb.Append(RowConvFProxy("Nonsym", "gcrodr", n, o[0], StatusNameFProxy(o[1]), st, ResidualFProxy(in A, in x, in b)));

            o.Dispose();
            arena.Dispose();
            return sb.ToString();
        }
    }
}
