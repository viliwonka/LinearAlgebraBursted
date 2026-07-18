using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Block-Jacobi preconditioned Conjugate Gradient (Krylov.cg with a preconditioner) over a
    // representative BSR system — the one square iterative solver SparseSolverBenchmark.cs doesn't
    // already cover (that file benchmarks plain cg/minres/biCGStab/lsqr). The system is a block-tridiagonal
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

            // SQUARE grids (gridX == gridY): the 2D Laplacian condition number grows ~O(N), so plain-CG
            // iterations grow ~O(sqrt(N)) while a good preconditioner stays ~flat -- the convergence-at-
            // scale story. (Elongated 4xN grids hide this: iters are set by the short side, flat in N.)
            sb.AppendLine("=== Preconditioner face-off, SQUARE 2D Laplacian BSR, solve to tol=sqrt(eps) ===");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,7} {6,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "residual"));
            sb.AppendLine(BenchPrecondFloat(32, 32));      // N = 1024
            sb.AppendLine(BenchPrecondDouble(32, 32));
            sb.AppendLine(BenchPrecondFloat(64, 64));      // N = 4096
            sb.AppendLine(BenchPrecondDouble(64, 64));
            sb.AppendLine(BenchPrecondFloat(101, 101));    // N = 10201
            sb.AppendLine(BenchPrecondDouble(101, 101));
            sb.AppendLine();

            sb.AppendLine("=== Preconditioner face-off, random sparse SPD BSR (genuine fill; IC(0) incomplete), tol=sqrt(eps) ===");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,7} {6,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "residual"));
            sb.AppendLine(BenchPrecondRandomFloat(120, 3, 0.30f, 0xC003Du));
            sb.AppendLine(BenchPrecondRandomDouble(120, 3, 0.30f, 0xC003Du));
            sb.AppendLine();

            // SCALAR 5-point Poisson (BR=1): the fair grid-independence case. IC(0) is genuinely
            // incomplete here (unlike the block-tridiagonal gallery Laplacian, where it is exact), so
            // point-preconditioner iteration counts grow ~O(sqrt(N)) while AMG's stay ~flat. Watch the
            // iters column across sizes.
            sb.AppendLine("=== Preconditioner face-off, SCALAR 5-point 2D Poisson (BR=1; IC0 genuinely incomplete), tol=sqrt(eps) ===");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,7} {6,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "residual"));
            sb.AppendLine(BenchPrecondScalarPoissonFloat(48, 48));      // N = 2304
            sb.AppendLine(BenchPrecondScalarPoissonDouble(48, 48));
            sb.AppendLine(BenchPrecondScalarPoissonFloat(96, 96));      // N = 9216
            sb.AppendLine(BenchPrecondScalarPoissonDouble(96, 96));
            sb.AppendLine(BenchPrecondScalarPoissonFloat(144, 144));    // N = 20736
            sb.AppendLine(BenchPrecondScalarPoissonDouble(144, 144));
            sb.AppendLine();

            sb.AppendLine("=== BlockJacobi build cost (ctor only, block-tridiagonal SPD) ===");
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-3} {3,11} {4,11}",
                "dtype", "N", "b", "med(ms)", "min(ms)"));
            sb.AppendLine(BenchJacobiBuildFloat(3, 4096));
            sb.AppendLine(BenchJacobiBuildDouble(3, 4096));
            sb.AppendLine(BenchJacobiBuildFloat(4, 2048));
            sb.AppendLine(BenchJacobiBuildDouble(4, 2048));
            sb.AppendLine();
        }
    }
}
