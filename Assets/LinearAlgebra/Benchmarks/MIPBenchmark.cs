using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatters for MIPBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes/seeds and row writers.
    public static class MIPBenchmarkFmt
    {
        // Section 1 (MIPLIB oracles): a generous safety net, not expected to bind -- stein9/stein15/p0033
        // converge in a few hundred nodes at most (see MIPTests.fProxy.cs's own measured baselines: stein15
        // ~275 nodes, p0033 ~447 nodes).
        public const int OracleMaxNodes = 50000;

        // Section 2 (synthetic scaling): fixed-seed random "branchy" integer-box MIPs (the BuildBranchy12
        // recipe from MIPTests.fProxy.cs, generalized to n/m/seed) at three sizes. n=12's seed (424242)
        // exactly reproduces the test suite's own Branchy12 instance. maxNodes is a TIGHT safety cap here
        // (unlike the MIPLIB oracles above, these are untested instances with no known baseline) -- a
        // diverging cell reports its NodeLimit status honestly instead of burning the wall-clock budget.
        public static readonly int[] ScalingN = { 8, 12, 16 };
        public static readonly uint[] ScalingSeed = { 80808u, 424242u, 161616u };
        public const int ScalingMaxNodes = 20000;

        // Section 3 (gap-limit economics): p0033 (double only) at three relGap settings.
        public static readonly double[] GapRelGaps = { 0.0, 0.01, 0.05 };
        public const int GapMaxNodes = 50000;

        // Section 4 (warm-start accounting): one mid-size instance (stein15, double only).
        public const int WarmStartMaxNodes = 50000;

        // `status` crosses the hand-written/template assembly boundary as a raw int -- same CS0012 reason
        // as LPBenchmarkFmt.InfeasRow's own doc comment (MIPStatus is defined in MIP.Info.cs, a
        // "singularFile" that's ALSO part of the TemplateSource-firstpass compile, so the generated
        // TemplateSourceBenchmarks firstpass compile has its own LOCAL MIPStatus, distinct from this
        // hand-written assembly's) -- StatusName (this assembly's own real MIPStatus) belongs here, one
        // cast away from the raw int.
        public static string StatusName(MIPStatus s) => s switch
        {
            MIPStatus.Optimal => "Optimal",
            MIPStatus.Infeasible => "Infeasible",
            MIPStatus.Unbounded => "Unbounded",
            MIPStatus.GapLimit => "GapLimit",
            MIPStatus.NodeLimit => "NodeLimit",
            MIPStatus.MaxIterations => "MaxIter",
            _ => "Unknown",
        };

        public static string Header() => string.Format("{0,-7} {1,-9} {2,4} {3,4} {4,11} {5,11} {6,7} {7,9} {8,10} {9,14}",
            "dtype", "instance", "n", "m", "med(ms)", "min(ms)", "nodes", "lpIters", "status", "objective");

        public static string Row(string dtype, string instance, int n, int m, Bench.Stat st, int nodes, int lpIters, int status, double obj) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-9} {2,4} {3,4} {4,11:F4} {5,11:F4} {6,7} {7,9} {8,10} {9,14:E4}",
                dtype, instance, n, m, st.Median, st.Min, nodes, lpIters, StatusName((MIPStatus)status), obj);

        public static string GapHeader() => string.Format("{0,-7} {1,-9} {2,7} {3,11} {4,11} {5,7} {6,9} {7,10} {8,14} {9,10}",
            "dtype", "instance", "relGap", "med(ms)", "min(ms)", "nodes", "lpIters", "status", "objective", "gap");

        public static string GapRow(string dtype, string instance, double relGap, Bench.Stat st, int nodes, int lpIters, int status, double obj, double gap) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-9} {2,7:F2} {3,11:F4} {4,11:F4} {5,7} {6,9} {7,10} {8,14:E4} {9,10:E3}",
                dtype, instance, relGap, st.Median, st.Min, nodes, lpIters, StatusName((MIPStatus)status), obj, gap);

        // Warm-start summary line (Section 4): average simplex pivots per B&B node.
        public static string RatioLine(string dtype, int lpIters, int nodes)
        {
            double ratio = nodes > 0 ? (double)lpIters / nodes : double.NaN;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} lpIterations={1} / nodes={2} -> {3:F3} simplex pivots/node (warm-started dual simplex)",
                dtype, lpIters, nodes, ratio);
        }
    }

    // ================================================================================================
    // Mixed-integer programming benchmark (MIP.solve: branch & bound over the warm-started dual simplex
    // with pseudocost/reliability branching, best-bound+plunging, domain propagation, a rounding
    // heuristic, and absGap/relGap gap limits -- docs/draft-spec-mip.md).
    //
    //   Section 1 (MIPLIB oracles): the standard tiny known-answer set (stein9/stein15/p0033, same literal
    //     instance data as MIPTests.fProxy.cs, not re-derived). stein9 runs in both dtypes; stein15 and
    //     p0033 run DOUBLE ONLY -- float cannot prove optimality on these within a sane node budget (see
    //     MIPTests.fProxy.cs's own Stein15/P0033 doc comments). Columns: wall time, nodes, lpIterations,
    //     objective.
    //
    //   Section 2 (synthetic scaling): random "branchy" integer-box MIPs at n = 8/12/16 (the
    //     MIPTests.fProxy.cs BuildBranchy12 recipe, generalized), both dtypes, under a TIGHT maxNodes
    //     safety cap -- unlike the MIPLIB oracles these are untested instances with no known baseline, so
    //     a diverging cell (the known stein15-style float robustness risk) reports its own status (e.g.
    //     NodeLimit) rather than being hand-excluded or burning the wall-clock budget.
    //
    //   Section 3 (gap-limit economics): p0033 (double only) at relGap = 0 / 0.01 / 0.05 -- shows the time
    //     vs proof-quality tradeoff: the rounding heuristic finds the optimal incumbent (3089) early, and
    //     the remaining time is spent proving the gap by closing the dual bound.
    //
    //   Section 4 (warm-start accounting): stein15 (double only, mid-size), reporting lpIterations/nodes --
    //     the average number of simplex pivots the warm-started dual simplex needs per B&B node (already
    //     in MIPInfo, no extra instrumentation needed). A low ratio is the warm-start payoff: each node
    //     reuses the persistent LPBasis instead of cold-starting a fresh LP solve.
    //
    // Every solve runs inside a [BurstCompile] IJob; timing is IJob.Run() (native code, not Mono). Hand-
    // written harness half. The timed IJob and the per-section instance-builders + build+measure methods
    // are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/MIPBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class MIPBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-mip.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Mixed-integer programming (MIP.solve: B&B over the warm-started dual simplex) ===");
            sb.AppendLine("Section 1: MIPLIB oracles stein9/stein15/p0033 (same literal instance data as");
            sb.AppendLine("MIPTests.fProxy.cs) -- stein9 both dtypes, stein15/p0033 double only (float cannot prove");
            sb.AppendLine("optimality within a sane node budget). Section 2: synthetic scaling -- random branchy");
            sb.AppendLine("integer-box MIPs at n=8/12/16, both dtypes, under a tight maxNodes safety cap (a diverging");
            sb.AppendLine("cell reports its own status rather than burning the wall-clock budget). Section 3:");
            sb.AppendLine("gap-limit economics -- p0033 (double only) at relGap=0/0.01/0.05, showing the rounding");
            sb.AppendLine("heuristic finds the optimum early while the gap PROOF (closing the dual bound) is what");
            sb.AppendLine("costs. Section 4: warm-start accounting -- stein15 (double only), lpIterations/nodes as");
            sb.AppendLine("the average simplex pivots per warm-started B&B node.");
            sb.AppendLine();

            SectionOraclesFloat(sb);
            SectionOraclesDouble(sb);
            SectionScalingFloat(sb);
            SectionScalingDouble(sb);
            SectionGapFloat(sb);       // no-op: double-only section
            SectionGapDouble(sb);
            SectionWarmStartFloat(sb); // no-op: double-only section
            SectionWarmStartDouble(sb);
        }
    }
}
