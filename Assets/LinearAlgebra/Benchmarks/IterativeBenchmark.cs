using System.Text;

// NOTE: PCG (preconditioned Conjugate Gradient) is not yet implemented in the library;
// only plain CG is benchmarked here.

namespace BULA.Benchmarks
{
    // Conjugate Gradient solver for dense SPD systems. The cost per iteration is one dense GEMV
    // (A·p) plus vector ops — all O(n²). With maxIter = 100 and tol = 0 every timed
    // sample runs exactly 100 iterations for a deterministic, representative measurement.
    //
    // SPD construction: A = MᵀM + I (M random n×n in [-1,1]). This is guaranteed SPD (all
    // eigenvalues >= 1), and the Frobenius-norm-n random M makes the condition number grow with n,
    // so 100 iterations do not converge early even at small sizes — timing is iteration-count-bounded.
    //
    // A and b are built once (A is `in` — CG does not modify it). x is zeroed at the start of each
    // Execute so every timed sample begins from the same zero initial guess.
    //
    // Hand-written harness half. The timed IJob (CGJob{Float,Double}) and build+measure method
    // (Bench{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/IterativeBenchmark.fProxy.cs.
    public static partial class IterativeBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-iterative.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Conjugate Gradient (dense SPD A = MᵀM + I; 100 iterations, tol=0; ms) ===");
            sb.AppendLine("    Each iteration: one dense GEMV (A·p) + vector ops. " +
                          "tol=0 forces all 100 iterations for deterministic timing.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();
        }
    }
}
