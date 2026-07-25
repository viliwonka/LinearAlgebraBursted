using System.Text;

namespace BULA.Benchmarks
{
    // Two views of every single-RHS square Krylov solver that has a clean unpreconditioned BSR entry
    // point, over two BSR galleries (an SPD gallery where all nine apply, a nonsymmetric gallery where
    // only the general-square solvers apply):
    //   FIXED-K  -- tol=0, exactly K iterations: per-iteration cost, isolating kernel throughput
    //               (task #58's optimization target; mirrors IterativeBenchmark/PCGBenchmark).
    //   CONVERGE -- real tol, generous cap: iterations-to-converge + status + time-to-solution, the
    //               comparison that actually ranks solvers (fixed-K hides convergence rate, which is
    //               the whole point of methods like gcrodr).
    // See KrylovGridTests.fProxy.cs for the full gallery x preconditioner x solver compatibility grid.
    //
    // Hand-written harness half. The timed IJobs and the per-gallery build+measure methods are
    // code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KrylovGridBenchmark.fProxy.cs.
    public static partial class KrylovGridBenchmark
    {
        const int K = 100;         // fixed iteration budget, tol=0 (deterministic per-iteration timing)
        const int Restart = 20;    // gmres/fgmres/gcrodr restart length (short enough that restarted
                                    // gmres stagnates on the hard ConvDiff gallery, letting gcrodr's
                                    // recycled-subspace deflation show)
        const int S = 4;           // idr shadow-space depth
        const int Recycle = 10;    // gcrodr recycled subspace size

        public static void Run() => Bench.WriteReport("benchmark-krylov-grid.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine(string.Format("=== Krylov solver per-iteration cost, BSR galleries (fixed K={0}, tol=0) ===", K));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-10} {3,-12} {4,11} {5,11} {6,14}",
                "dtype", "N", "gallery", "solver", "med(ms)", "min(ms)", "residual"));
            sb.AppendLine(BenchSpdFloat(Restart, S, Recycle, K));
            sb.AppendLine(BenchSpdDouble(Restart, S, Recycle, K));
            sb.AppendLine(BenchNonsymFloat(Restart, S, Recycle, K));
            sb.AppendLine(BenchNonsymDouble(Restart, S, Recycle, K));

            sb.AppendLine("=== Krylov solver convergence, BSR galleries (run to tol=sqrt(eps), cap 4N iters) ===");
            sb.AppendLine("    iters/status/time to SOLUTION -- the solver ranking fixed-K cannot show.");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-10} {3,-12} {4,7} {5,-13} {6,11} {7,11} {8,11} {9,14}",
                "dtype", "N", "gallery", "solver", "iters", "status", "med(ms)", "min(ms)", "ms/iter", "residual"));
            sb.AppendLine(BenchSpdConvergeFloat(Restart, S, Recycle));
            sb.AppendLine(BenchSpdConvergeDouble(Restart, S, Recycle));
            sb.AppendLine(BenchNonsymConvergeFloat(Restart, S, Recycle));
            sb.AppendLine(BenchNonsymConvergeDouble(Restart, S, Recycle));
            sb.AppendLine(BenchHardConvergeFloat(Restart, S, Recycle));
            sb.AppendLine(BenchHardConvergeDouble(Restart, S, Recycle));
        }
    }
}
