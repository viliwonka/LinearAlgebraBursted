using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Randomized SVD (top-k, k=16) and the two pseudo-inverse solvers that ride the Golub-Kahan SVD.
    // All three are time-only: the cost is dominated by iterative convergence, so GFLOP/s would be
    // misleading. A is never modified; workspaces are built once outside the timing loop.
    //
    // Hand-written harness half. The timed IJobs (SvdRandomized/PinvSolve/PseudoInverse Job {Float,Double})
    // and build+measure methods (SvdRand/Pinv/PseudoInv {Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SvdSolversBenchmark.fProxy.cs.
    public static partial class SvdSolversBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-svdsolvers.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== randomized (Halko-Martinsson-Tropp, top-k singular triplets, k=16; ms) ===");
            sb.AppendLine("    Randomized low-rank path: GEMM sketch -> QR range-finder -> small exact SVD.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdRandFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdRandDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== pinvSolve (Moore-Penrose minimum-norm LS solve via Golub-Kahan SVD; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(PinvFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(PinvDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== pseudoInverse (Moore-Penrose pseudo-inverse matrix via Golub-Kahan SVD; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(PseudoInvFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(PseudoInvDouble(n));
            sb.AppendLine();
        }
    }
}
