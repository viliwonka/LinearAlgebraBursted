using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatters for LPBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes and row writers.
    public static class LPBenchmarkFmt
    {
        // LP.solve sizes: n variables, m = n/2 inequality constraints (a wide, interior feasible region).
        public static readonly int[] SolveVarsN = { 24, 48, 96 };

        // LAD sizes: m observations, NCoef coefficients. LAD-via-LP has m equality constraints, so its
        // tableau grows with m -- kept modest here precisely to show the scaling gap against IRLS.
        public static readonly int[] LadRowsM = { 48, 96, 192 };
        public const int NCoef = 4;

        public static string StatusName(LPStatus s) => s == LPStatus.Optimal ? "Optimal"
            : s == LPStatus.Infeasible ? "Infeasible" : s == LPStatus.Unbounded ? "Unbounded" : "MaxIter";

        public static string SolveHeader() => string.Format("{0,-7} {1,-6} {2,-6} {3,-14} {4,11} {5,11} {6,7} {7,14}",
            "dtype", "n", "m", "method", "med(ms)", "min(ms)", "iters", "objective");

        public static string SolveRow(string dtype, int n, int m, string method, Bench.Stat st, int iters, double obj) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-14} {4,11:F4} {5,11:F4} {6,7} {7,14:E4}",
                dtype, n, m, method, st.Median, st.Min, iters, obj);

        public static string LadHeader() => string.Format("{0,-7} {1,-6} {2,-6} {3,-16} {4,11} {5,11} {6,7} {7,14}",
            "dtype", "m", "n", "method", "med(ms)", "min(ms)", "iters", "L1 residual");

        public static string LadRow(string dtype, int m, int n, string method, Bench.Stat st, int iters, double l1) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-16} {4,11:F4} {5,11:F4} {6,7} {7,14:E4}",
                dtype, m, n, method, st.Median, st.Min, iters, l1);
    }

    // ================================================================================================
    // Linear programming + least-absolute-deviation benchmark.
    //
    //   Section 1 (LP.solve): random dense feasible+bounded LPs (min cᵀx s.t. A x <= b, x >= 0, A >= 0,
    //     b = A x0 + slack so x0 is feasible). Two backends compared head to head -- two-phase simplex
    //     vs Mehrotra interior point -- on the IDENTICAL problem; the objective column shows they agree.
    //
    //   Section 2 (LAD): random overdetermined regression b = A x_true + noise with periodic gross
    //     outliers. Exact L1 fit (LP.lad) via simplex vs via interior point vs the fast approximate
    //     IRLS (Optimize.ladIRLS). The L1-residual column shows all three reach essentially the same
    //     minimum; the timing shows LAD-via-LP grows with the number of observations (its constraint
    //     count) while IRLS -- a fixed-size normal-equation solve per iteration -- barely moves.
    //
    // Every solve runs inside a [BurstCompile] IJob; timing is IJob.Run() (native code, not Mono).
    // Hand-written harness half. The timed IJobs and the per-section build+measure methods are code-
    // generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class LPBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-lp.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Linear programming (simplex vs interior point) + least absolute deviation ===");
            sb.AppendLine("Section 1: random dense feasible/bounded LPs, min cx s.t. Ax<=b, x>=0 -- simplex vs");
            sb.AppendLine("interior point on the SAME problem (objective column shows they agree). Section 2: LAD");
            sb.AppendLine("(L1) regression with gross outliers -- exact LP.lad (simplex / interior point) vs fast");
            sb.AppendLine("approximate Optimize.ladIRLS. L1-residual column shows agreement; timing shows LAD-via-");
            sb.AppendLine("LP scales with observation count (its constraints) while IRLS stays a fixed n x n solve.");
            sb.AppendLine();

            SectionSolveFloat(sb);
            SectionSolveDouble(sb);
            SectionLadFloat(sb);
            SectionLadDouble(sb);
        }
    }
}
