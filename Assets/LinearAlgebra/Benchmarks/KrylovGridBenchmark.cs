using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Per-iteration cost of every single-RHS square Krylov solver that has a clean unpreconditioned
    // BSR entry point (task #58's optimization target) -- fixed iteration budget, tol=0, so every
    // timed sample runs exactly K iterations (deterministic timing, mirrors IterativeBenchmark /
    // SparseSolverBenchmark / PCGBenchmark's convention). Two BSR galleries: an SPD gallery (every
    // solver applies) and a nonsymmetric gallery (only the general-square solvers apply). See
    // KrylovGridTests.fProxy.cs for the full gallery x preconditioner x solver compatibility grid
    // this benchmark's per-solver timing feeds.
    //
    // Hand-written harness half. The timed IJobs and the per-gallery build+measure methods are
    // code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KrylovGridBenchmark.fProxy.cs.
    public static partial class KrylovGridBenchmark
    {
        const int K = 100;         // fixed iteration budget, tol=0 (deterministic timing)
        const int Restart = 30;    // gmres/fgmres/gcrodr restart length
        const int S = 4;           // idr shadow-space depth
        const int Recycle = 10;    // gcrodr recycled subspace size

        public static void Run() => Bench.WriteReport("benchmark-krylov-grid.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine(string.Format("=== Krylov solver per-iteration cost, BSR galleries (K={0}, tol=0) ===", K));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-10} {3,-12} {4,11} {5,11} {6,14}",
                "dtype", "N", "gallery", "solver", "med(ms)", "min(ms)", "residual"));
            sb.AppendLine(BenchSpdFloat(Restart, S, Recycle, K));
            sb.AppendLine(BenchSpdDouble(Restart, S, Recycle, K));
            sb.AppendLine(BenchNonsymFloat(Restart, S, Recycle, K));
            sb.AppendLine(BenchNonsymDouble(Restart, S, Recycle, K));
        }
    }
}
