using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Ridge block-CG (bcg) and deflating block-CG (bcgrq) vs the scalar loop of s independent cg solves,
    // over a BSR 2D-Poisson operator (5-point stencil, ~5 nonzeros/row, n = grid²), solved to sqrt(eps)
    // — SPARSE is block-CG's real use case. A "spMM x50 / spMVx s x50" pair per row is a matvec-only
    // probe: 50 block spMM(s) calls vs 50*s single-vector spMV calls (same total matvec work), whose
    // wall-clock ratio isolates the s×n multivector layout cost from the O(s²n) solver bookkeeping.
    // s = number of right-hand sides; `iters` is block iterations for the block-CG rows, summed column
    // iterations for the scalar loop, and the rep count for the probes. `minActive` is the smallest
    // active block width each block-CG row reached (bcg always reports rhs; bcgrq's LQRP deflation can
    // report less).
    //
    // Hand-written harness half; the timed IJobs + build/measure (Bench{Float,Double}) are code-generated
    // per dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/BlockCGSparseBenchmark.fProxy.cs.
    public static partial class BlockCGSparseBenchmark
    {
        static readonly int[] Grids = { 32, 48, 64 };   // n = 1024, 2304, 4096
        static readonly int[] Ss = { 2, 4, 8, 16, 32 };

        public static void Run() => Bench.WriteReport("benchmark-blockcg-sparse.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Block-CG vs scalar-loop, BSR 2D Poisson (n=grid², independent RHS, tol=sqrt(eps)) ===");
            sb.AppendLine(string.Format("{0,-7}{1,-7}{2,-4}{3,-14}{4,10}{5,12}{6,8}{7,8}",
                "dtype", "N", "s", "method", "med(ms)", "min(ms)", "iters", "minAct"));
            foreach (var g in Grids)
                foreach (var s in Ss)
                    sb.Append(BenchFloat(g, s));
            foreach (var g in Grids)
                foreach (var s in Ss)
                    sb.Append(BenchDouble(g, s));
            sb.AppendLine();
        }
    }
}
