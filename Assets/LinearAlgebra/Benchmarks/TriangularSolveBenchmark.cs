using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic row formatter for TriangularSolveBenchmark. Public so the code-generated
    // per-dtype build method (in a separate template assembly) can call it. Time-only (no GFLOP/s):
    // the six rows per size have different flop shapes (vector vs k=8 multi-RHS vs single-pass kernel),
    // so a single throughput column across them would be misleading -- the time columns and the
    // forward/TransA ratio are the honest signal here (mirrors Bench.RowTime's rationale).
    public static class TriSolveFmt
    {
        public static string Header()
        {
            return string.Format("{0,-7} {1,-26} {2,-6} {3,11} {4,11} {5,11} {6,11}",
                "dtype", "kernel", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)");
        }

        public static string RowKernel(string dtype, string kernel, int n, Bench.Stat st)
        {
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-26} {2,-6} {3,11:F4} {4,11:F4} {5,11:F4} {6,11:F4}",
                dtype, kernel, n, st.Min, st.Median, st.Mean, st.Max);
        }
    }

    // Solve-ONLY isolation for LU.decompSolve vs decompSolveTransA (the getrs 'N'/'T' triangular pair),
    // plus the single-pass Blas kernels underneath (triUpperLU vs triUpperLUTransA). Motivation:
    // DirectSolveBenchmark's LU rows are FUSED factor+solve, and the O(n^3) factorization swamps the
    // O(n^2) triangular solve at every size it tests -- it cannot resolve a forward-vs-TransA solve gap
    // (a ~6ms difference at N=1024 float there could equally be factorization noise). Here the compact
    // LU factor is built ONCE per size, OUTSIDE every timed region (TriSolveFactorJob{Float,Double}.Run(),
    // not wrapped in Bench.Time), and every timed job below re-copies the pristine RHS from bSrc/BXsrc
    // before solving, since the solve overwrites b_to_x/B_to_X in place and iteration 2+ would otherwise
    // solve the PREVIOUS iteration's already-solved output instead of the intended RHS. The copy has the
    // same shape/cost in both directions (a plain element or row-block copy), so it cancels out of the
    // forward/TransA comparison and does not explain any gap that shows up.
    //
    // "vec" rows use the vector decompSolve/decompSolveTransA overloads; "mat k=8" rows use the
    // multi-RHS (TRSM-shaped) overloads with 8 right-hand sides; the "Blas tri*" rows call ONE
    // triangular pass directly (no pivot gather/scatter, no second pass), isolating the pivot-indirected
    // row-dot back-substitution (triUpperLU) from the pivot-indirected right-looking/axpy forward step
    // (triUpperLUTransA), and from the pivot-application cost the full LU-solve rows include.
    //
    // Hand-written harness half. The timed IJobs (TriSolve*Job{Float,Double}) and the build+measure
    // method (TriSolve{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/TriangularSolveBenchmark.fProxy.cs.
    public static partial class TriangularSolveBenchmark
    {
        // O(n^2) work per rep (a triangular solve, not a factorization) -- trivially fast at every size
        // here (sub-second total even at N=4096), so this whole section runs in well under a second,
        // let alone the 60s this task budgets for it.
        static readonly int[] Sizes = { 256, 1024, 4096 };

        public static void Run() => Bench.WriteReport("benchmark-triangularsolve.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Triangular solve ONLY (LU factored ONCE outside the timed region), forward vs TransA ===");
            sb.AppendLine(TriSolveFmt.Header());
            foreach (var n in Sizes) sb.AppendLine(TriSolveFloat(n));
            foreach (var n in Sizes) sb.AppendLine(TriSolveDouble(n));
            sb.AppendLine();
        }
    }
}
