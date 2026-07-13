using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Block-Jacobi Preconditioned Conjugate Gradient (Krylov.pcg) over a representative BSR system —
    // the one square iterative solver SparseSolverBenchmark.cs doesn't already cover (that file
    // benchmarks plain cg/minres/biCGStab/cgls/lsqr but predates pcg). The system is a block-tridiagonal
    // SPD matrix (block size BR, a common 1D FEM/heat-equation stencil): diagonally-dominant diagonal
    // blocks + small symmetric off-diagonal coupling to the immediate neighbor block only, so it is
    // genuinely sparse (nnzb = 3*nb-2) without needing SparseSolverBenchmark's randomized block-pattern
    // machinery. maxIter is FIXED with tol=0 (mirrors SparseSolverBenchmark's convention),
    // so every timed sample runs exactly K iterations — deterministic timing; the residual column shows
    // convergence, not just speed. Plain cg (unpreconditioned) on the SAME system is included alongside
    // for a direct preconditioning-overhead-vs-iteration-savings comparison.
    //
    // Hand-written harness half. The timed IJobs (Pcg/Cg BsrJob {Float,Double}), the block-tridiagonal
    // builder, the residual, and the build+measure method (Bench{Float,Double}) are code-generated per
    // dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/PCGBenchmark.fProxy.cs.
    public static partial class PCGBenchmark
    {
        const int BR = 3;         // block size (matches SparseSolverBenchmark's FEM/cloth/PD workhorse)
        const int NB = 256;       // number of blocks -> N = 768
        const int K = 40;         // fixed iteration budget, tol=0 (deterministic timing)

        public static void Run() => Bench.WriteReport("benchmark-pcg.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine(string.Format("=== Block-Jacobi PCG vs plain CG, block-tridiagonal SPD BSR (b={0}, nb={1}, K={2}, tol=0) ===", BR, NB, K));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "residual"));
            sb.AppendLine(BenchFloat(BR, NB, K));
            sb.AppendLine(BenchDouble(BR, NB, K));
            sb.AppendLine();

            sb.AppendLine("=== Preconditioner face-off, 2D Laplacian BSR, solve to tol=sqrt(eps) ===");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,7} {6,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "residual"));
            sb.AppendLine(BenchPrecondFloat(4, 256));
            sb.AppendLine(BenchPrecondDouble(4, 256));
            sb.AppendLine(BenchPrecondFloat(4, 1024));
            sb.AppendLine(BenchPrecondDouble(4, 1024));
            sb.AppendLine();

            sb.AppendLine("=== Preconditioner face-off, random sparse SPD BSR (genuine fill; IC(0) incomplete), tol=sqrt(eps) ===");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,7} {6,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "residual"));
            sb.AppendLine(BenchPrecondRandomFloat(120, 3, 0.30f, 0xC003Du));
            sb.AppendLine(BenchPrecondRandomDouble(120, 3, 0.30f, 0xC003Du));
            sb.AppendLine();
        }
    }
}
