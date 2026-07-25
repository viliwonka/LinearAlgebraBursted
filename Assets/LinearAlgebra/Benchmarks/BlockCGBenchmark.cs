using System.Text;

namespace BULA.Benchmarks
{
    // Block-CG (multi-RHS) vs the scalar loop of s independent cg solves, on the SAME dense SPD system
    // (A = MᵀM + nI, condition grows with n) and the SAME s right-hand sides, solved to the same
    // tol = sqrt(eps). The wall-clock ratio is the true block-vs-scalar payoff: block-CG shares one
    // Krylov subspace (fewer iterations) and streams A over the whole block once per iteration via
    // ApplyBlock (one GEMM instead of s GEMVs), against O(s²n) block-update overhead + a tiny sxs
    // Cholesky per iteration. s = number of right-hand sides. `iters` is block iterations for the
    // block row, and the SUM over the s columns for the scalar row.
    //
    // Hand-written harness half. The timed IJobs and build+measure method (Bench{Float,Double}) are
    // code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/BlockCGBenchmark.fProxy.cs.
    public static partial class BlockCGBenchmark
    {
        static readonly int[] Ns = { 128, 256, 512 };
        static readonly int[] Ss = { 1, 2, 4, 8, 16 };

        public static void Run() => Bench.WriteReport("benchmark-blockcg.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Block-CG vs scalar-loop (dense SPD A = MᵀM + nI, s RHS, solve to sqrt(eps)) ===");
            sb.AppendLine(string.Format("{0,-7}{1,-7}{2,-4}{3,-14}{4,10}{5,12}{6,8}",
                "dtype", "N", "s", "method", "med(ms)", "min(ms)", "iters"));
            foreach (var n in Ns)
                foreach (var s in Ss)
                    sb.Append(BenchFloat(n, s));
            foreach (var n in Ns)
                foreach (var s in Ss)
                    sb.Append(BenchDouble(n, s));
            sb.AppendLine();
        }
    }
}
